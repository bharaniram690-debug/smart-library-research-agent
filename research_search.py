"""research_search.py - Academic paper discovery, caching, and multi-provider search module.

This module handles:
1. Input validation for academic queries and result counts.
2. Search query sanitization.
3. HTTP communication with the Semantic Scholar Graph API using documented authentication (`x-api-key`).
4. HTTP 429 rate limit handling with exponential backoff and `Retry-After` header respect (max 3 retries).
5. In-memory caching of successful queries to eliminate redundant API requests.
6. OpenAlex API fallback provider when Semantic Scholar is unavailable or rate-limited.
7. Deduplication, metadata extraction, abstract reconstruction, and missing abstract logging.
"""

import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests

# Endpoints
SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_ENDPOINT = "https://api.openalex.org/works"

# Requested fields for Semantic Scholar
REQUESTED_FIELDS = "paperId,title,abstract,authors,year,url,venue,citationCount,openAccessPdf"

# Timeout and Retry configuration
DEFAULT_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

# Fast backoff when running in automated test suite, standard in production
INITIAL_BACKOFF_SECONDS = 0.01 if "pytest" in sys.modules else 1.0
BACKOFF_FACTOR = 2.0
MAX_BACKOFF_SECONDS = 60.0

# Allowed range for maximum papers
MIN_PAPERS_ALLOWED = 1
MAX_PAPERS_ALLOWED = 20

# In-memory search cache: (query_lower, limit, provider) -> result dict
_SEARCH_CACHE: Dict[Tuple[str, int, str], Dict[str, Any]] = {}


# ==============================================================================
# CACHING FUNCTIONS
# ==============================================================================

def get_cache_key(query: str, limit: int, provider: str = "all") -> Tuple[str, int, str]:
    """Generates a normalized cache key for query caching."""
    return (clean_search_query(query).lower(), limit, provider.lower())


def get_cached_search(query: str, limit: int, provider: str = "all") -> Optional[Dict[str, Any]]:
    """Retrieves cached search results if present."""
    key = get_cache_key(query, limit, provider)
    cached = _SEARCH_CACHE.get(key)
    if cached:
        # Return a copy to avoid mutation
        copy_res = dict(cached)
        copy_res["cached"] = True
        return copy_res
    return None


def set_cached_search(query: str, limit: int, result: Dict[str, Any], provider: str = "all") -> None:
    """Stores successful search results in memory cache."""
    key = get_cache_key(query, limit, provider)
    _SEARCH_CACHE[key] = dict(result)


def clear_search_cache() -> None:
    """Clears the search cache (useful for testing and memory resets)."""
    _SEARCH_CACHE.clear()


# ==============================================================================
# VALIDATION & NORMALIZATION
# ==============================================================================

def validate_search_inputs(
    topic: str,
    max_papers: int,
    min_allowed: int = MIN_PAPERS_ALLOWED,
    max_allowed: int = MAX_PAPERS_ALLOWED,
) -> Tuple[bool, str]:
    """Validates user's research topic and paper retrieval count."""
    if not isinstance(topic, str):
        return False, "Research topic must be a text string."

    trimmed_topic = topic.strip()
    if not trimmed_topic:
        return False, "Research topic cannot be empty. Please enter a valid scientific topic."

    if not re.search(r"[a-zA-Z0-9]", trimmed_topic):
        return False, "Research topic must contain alphanumeric characters, not only symbols."

    if len(trimmed_topic) < 3:
        return False, "Research topic is too short. Please provide at least 3 characters for a meaningful search."

    if not isinstance(max_papers, int) or max_papers < min_allowed or max_papers > max_allowed:
        return (
            False,
            f"Paper count must be an integer between {min_allowed} and {max_allowed}. Received: {max_papers}.",
        )

    return True, ""


def clean_search_query(topic: str) -> str:
    """Prepares and normalizes the search query for academic search engines."""
    cleaned = topic.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip("\"'.,;:")
    return cleaned


def parse_retry_after(retry_after_header: Optional[str], default_backoff: float) -> float:
    """Parses Retry-After header value into seconds, respecting server limits."""
    if not retry_after_header:
        return default_backoff
    try:
        val = float(retry_after_header.strip())
        return max(0.1, min(val, MAX_BACKOFF_SECONDS))
    except (ValueError, TypeError):
        return default_backoff


def reconstruct_abstract_from_inverted_index(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstructs text abstract from OpenAlex's official abstract_inverted_index.

    OpenAlex indexes abstracts as { "word": [positions...] }.
    We sort word occurrences by position to faithfully reconstruct the author's original text.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return None

    word_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    word_positions.append((pos, word))

    if not word_positions:
        return None

    word_positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in word_positions)


# ==============================================================================
# PROVIDER 1: SEMANTIC SCHOLAR GRAPH API
# ==============================================================================

def get_semantic_scholar_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Builds HTTP headers for Semantic Scholar, securely sending API key if configured."""
    resolved_api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
    headers: Dict[str, str] = {
        "User-Agent": "SmartLibraryResearchAgent/1.0 (academic-research-tool)",
        "Accept": "application/json",
    }
    if resolved_api_key:
        headers["x-api-key"] = resolved_api_key
    return headers


def search_semantic_scholar(
    query: str,
    limit: int = 5,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
    sleep_func: Optional[Callable[[float], None]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Queries the Semantic Scholar Graph API with rate limit backoff and retries.

    Handles HTTP 429 with exponential backoff and respects `Retry-After` header.
    Limits retries to max_retries (default: 3).

    Returns:
        Dict with keys: 'status', 'data', 'total', 'error', 'source_provider', 'cached'
    """
    # Check cache
    if use_cache:
        cached = get_cached_search(query, limit, provider="semantic_scholar")
        if cached:
            return cached

    sleeper = sleep_func if sleep_func is not None else time.sleep
    headers = get_semantic_scholar_headers(api_key)
    params = {
        "query": query,
        "limit": limit,
        "fields": REQUESTED_FIELDS,
    }

    current_backoff = INITIAL_BACKOFF_SECONDS
    last_error_message = ""

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_ENDPOINT,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # Handle HTTP 429 Rate Limiting
            if response.status_code == 429:
                retry_after_hdr = response.headers.get("Retry-After")
                wait_seconds = parse_retry_after(retry_after_hdr, current_backoff)

                if attempt < max_retries:
                    sleeper(wait_seconds)
                    current_backoff = min(current_backoff * BACKOFF_FACTOR, MAX_BACKOFF_SECONDS)
                    continue
                else:
                    last_error_message = (
                        f"Semantic Scholar API rate limit exceeded (HTTP 429) after {max_retries} retries. "
                        "The service is currently congested or free-tier quota has been exhausted."
                    )
                    return {
                        "status": "error",
                        "data": [],
                        "total": 0,
                        "error": last_error_message,
                        "source_provider": "Semantic Scholar",
                        "rate_limited": True,
                    }

            if response.status_code == 403:
                return {
                    "status": "error",
                    "data": [],
                    "total": 0,
                    "error": "Semantic Scholar API access forbidden (HTTP 403). Check your API key or permissions.",
                    "source_provider": "Semantic Scholar",
                }

            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError:
                return {
                    "status": "error",
                    "data": [],
                    "total": 0,
                    "error": "Received malformed non-JSON response from Semantic Scholar API.",
                    "source_provider": "Semantic Scholar",
                }

            data_papers = payload.get("data")
            total_results = payload.get("total", 0)

            if not data_papers or len(data_papers) == 0:
                return {
                    "status": "empty",
                    "data": [],
                    "total": 0,
                    "error": f"No relevant academic research papers found for '{query}'.",
                    "source_provider": "Semantic Scholar",
                }

            # Tag each paper with its verified source provider
            for paper in data_papers:
                if isinstance(paper, dict):
                    paper["source_provider"] = "Semantic Scholar"

            result = {
                "status": "success",
                "data": data_papers,
                "total": total_results,
                "error": "",
                "source_provider": "Semantic Scholar",
                "cached": False,
            }

            if use_cache:
                set_cached_search(query, limit, result, provider="semantic_scholar")

            return result

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                sleeper(current_backoff)
                current_backoff = min(current_backoff * BACKOFF_FACTOR, MAX_BACKOFF_SECONDS)
                continue
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": f"Semantic Scholar request timed out after {timeout} seconds.",
                "source_provider": "Semantic Scholar",
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": "Failed to connect to Semantic Scholar API. Please verify network connectivity.",
                "source_provider": "Semantic Scholar",
            }
        except requests.exceptions.HTTPError as http_err:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": f"Semantic Scholar HTTP error: {http_err}",
                "source_provider": "Semantic Scholar",
            }
        except Exception as exc:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": f"Unexpected error querying Semantic Scholar: {str(exc)}",
                "source_provider": "Semantic Scholar",
            }

    return {
        "status": "error",
        "data": [],
        "total": 0,
        "error": last_error_message or "Semantic Scholar request failed after retries.",
        "source_provider": "Semantic Scholar",
        "rate_limited": True,
    }


# ==============================================================================
# PROVIDER 2: OPENALEX API (FALLBACK SEARCH PROVIDER)
# ==============================================================================

def search_openalex(
    query: str,
    limit: int = 5,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Queries OpenAlex API as an academic research fallback provider.

    Documented OpenAlex endpoint: https://api.openalex.org/works
    Reconstructs abstract from abstract_inverted_index without fabricating data.
    """
    if use_cache:
        cached = get_cached_search(query, limit, provider="openalex")
        if cached:
            return cached

    headers = {
        "User-Agent": "SmartLibraryResearchAgent/1.0 (mailto:library-agent@academic.org)",
        "Accept": "application/json",
    }
    params = {
        "search": query,
        "per-page": limit,
    }

    try:
        response = requests.get(
            OPENALEX_ENDPOINT,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        if response.status_code == 429:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": "OpenAlex API rate limit reached.",
                "source_provider": "OpenAlex",
            }

        if response.status_code != 200:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": f"OpenAlex API returned HTTP {response.status_code}.",
                "source_provider": "OpenAlex",
            }

        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            return {
                "status": "error",
                "data": [],
                "total": 0,
                "error": "Received non-JSON response from OpenAlex API.",
                "source_provider": "OpenAlex",
            }

        results = payload.get("results", [])
        total_count = payload.get("meta", {}).get("count", 0)

        if not results:
            return {
                "status": "empty",
                "data": [],
                "total": 0,
                "error": f"No relevant research papers found on OpenAlex for '{query}'.",
                "source_provider": "OpenAlex",
            }

        # Normalize OpenAlex records into standard paper format
        normalized_papers: List[Dict[str, Any]] = []
        for work in results:
            if not isinstance(work, dict):
                continue

            raw_title = work.get("title")
            if not raw_title:
                continue

            openalex_id = work.get("id", "")
            paper_id = openalex_id.replace("https://openalex.org/", "") if openalex_id else ""

            # Extract authors
            authorships = work.get("authorships", [])
            authors_list = []
            for item in authorships:
                if isinstance(item, dict) and item.get("author"):
                    auth_name = item["author"].get("display_name")
                    if auth_name:
                        authors_list.append({"name": auth_name})

            # Reconstruct abstract from inverted index
            abstract_text = reconstruct_abstract_from_inverted_index(work.get("abstract_inverted_index"))

            # Primary location venue and URL
            primary_loc = work.get("primary_location") or {}
            source_obj = primary_loc.get("source") or {}
            venue_name = source_obj.get("display_name") or "OpenAlex Index"

            # Prefer DOI link, fallback to landing page or openalex ID
            doi_url = work.get("doi")
            landing_url = primary_loc.get("landing_page_url")
            final_url = doi_url or landing_url or openalex_id or "https://openalex.org"

            citation_count = work.get("cited_by_count", 0)
            year = work.get("publication_year")

            normalized_papers.append({
                "paperId": paper_id or f"openalex_{len(normalized_papers)}",
                "title": raw_title,
                "authors": authors_list,
                "year": year,
                "venue": venue_name,
                "citationCount": citation_count,
                "abstract": abstract_text,
                "url": final_url,
                "source_provider": "OpenAlex",
            })

        result = {
            "status": "success",
            "data": normalized_papers,
            "total": total_count,
            "error": "",
            "source_provider": "OpenAlex",
            "cached": False,
        }

        if use_cache:
            set_cached_search(query, limit, result, provider="openalex")

        return result

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "data": [],
            "total": 0,
            "error": f"OpenAlex API request timed out after {timeout} seconds.",
            "source_provider": "OpenAlex",
        }
    except Exception as exc:
        return {
            "status": "error",
            "data": [],
            "total": 0,
            "error": f"OpenAlex API error: {str(exc)}",
            "source_provider": "OpenAlex",
        }


# ==============================================================================
# ORCHESTRATOR: UNIFIED ACADEMIC SEARCH WITH RETRIES & FALLBACK
# ==============================================================================

def search_academic_papers(
    query: str,
    limit: int = 5,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
    sleep_func: Optional[Callable[[float], None]] = None,
    enable_fallback: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Orchestrates multi-provider academic search with caching and fallback.

    Pipeline:
    1. Check query cache (avoids repeated API calls).
    2. Try Semantic Scholar with up to 3 retries on HTTP 429.
    3. If Semantic Scholar fails and fallback is enabled: query OpenAlex.
    4. If both fail: return friendly error guidance.
    """
    # 1. Global Cache Check
    if use_cache:
        cached_result = get_cached_search(query, limit, provider="all")
        if cached_result:
            return cached_result

    # 2. Primary Provider: Semantic Scholar
    s2_result = search_semantic_scholar(
        query=query,
        limit=limit,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        sleep_func=sleep_func,
        use_cache=use_cache,
    )

    if s2_result["status"] == "success":
        s2_result["fallback_used"] = False
        if use_cache:
            set_cached_search(query, limit, s2_result, provider="all")
        return s2_result

    # If Semantic Scholar returned clean empty results, return without error
    if s2_result["status"] == "empty":
        return s2_result

    # 3. Fallback Provider: OpenAlex (if Semantic Scholar failed or hit 429)
    if enable_fallback:
        openalex_result = search_openalex(
            query=query,
            limit=limit,
            timeout=timeout,
            use_cache=use_cache,
        )

        if openalex_result["status"] == "success":
            openalex_result["fallback_used"] = True
            openalex_result["fallback_notice"] = (
                "Semantic Scholar was unavailable or rate-limited. "
                "Successfully retrieved research papers using OpenAlex."
            )
            if use_cache:
                set_cached_search(query, limit, openalex_result, provider="all")
            return openalex_result

        if openalex_result["status"] == "empty":
            return openalex_result

    # 4. Friendly error when all providers fail
    friendly_error = (
        "Both academic search providers (Semantic Scholar and OpenAlex) are currently unavailable. "
        "Semantic Scholar encountered rate-limiting (HTTP 429) or connection constraints, "
        "and the OpenAlex fallback search was unable to complete. "
        "Please wait 30-60 seconds before retrying, refine your query, "
        "or configure a SEMANTIC_SCHOLAR_API_KEY in your .env file for higher throughput."
    )

    return {
        "status": "error",
        "data": [],
        "total": 0,
        "error": friendly_error,
        "all_providers_failed": True,
        "s2_error": s2_result.get("error", ""),
    }


# ==============================================================================
# FILTERING, DEDUPLICATION & METADATA AUDITING
# ==============================================================================

def filter_and_process_papers(
    raw_papers: List[Dict[str, Any]],
    requested_limit: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filters duplicate papers, normalizes metadata, and tags source providers.

    Decision-Making Rules:
    1. Filter out papers without a title.
    2. Deduplicate using normalized title and paperId.
    3. Retain papers with missing abstracts, tag has_abstract=False, and note limitation.
    4. Preserve and report source_provider ('Semantic Scholar' or 'OpenAlex').
    5. Resolve direct URLs.
    """
    seen_ids = set()
    seen_titles = set()
    processed_papers: List[Dict[str, Any]] = []

    duplicates_count = 0
    missing_abstracts_count = 0
    providers_count: Dict[str, int] = {}

    for paper in raw_papers:
        if not isinstance(paper, dict):
            continue

        raw_title = paper.get("title")
        if not raw_title or not isinstance(raw_title, str):
            continue

        normalized_title = raw_title.strip().lower()
        paper_id = paper.get("paperId") or ""

        # Deduplication
        if paper_id and paper_id in seen_ids:
            duplicates_count += 1
            continue
        if normalized_title in seen_titles:
            duplicates_count += 1
            continue

        if paper_id:
            seen_ids.add(paper_id)
        seen_titles.add(normalized_title)

        # Provider tracking
        source_provider = paper.get("source_provider") or "Semantic Scholar"
        providers_count[source_provider] = providers_count.get(source_provider, 0) + 1

        # Format authors
        raw_authors = paper.get("authors") or []
        author_names = []
        for author in raw_authors:
            if isinstance(author, dict) and author.get("name"):
                author_names.append(author["name"].strip())
            elif isinstance(author, str):
                author_names.append(author.strip())

        authors_str = ", ".join(author_names) if author_names else "Unknown Authors"

        # Check abstract presence
        raw_abstract = paper.get("abstract")
        has_abstract = bool(raw_abstract and isinstance(raw_abstract, str) and raw_abstract.strip())
        if not has_abstract:
            missing_abstracts_count += 1
            cleaned_abstract = (
                "[Notice: Abstract not provided by publisher metadata. "
                "Analysis based on title and publication context.]"
            )
        else:
            cleaned_abstract = raw_abstract.strip()

        # Resolve URL
        paper_url = paper.get("url")
        if not paper_url:
            open_access_pdf = paper.get("openAccessPdf")
            if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
                paper_url = open_access_pdf["url"]
            elif paper_id:
                if source_provider == "OpenAlex":
                    paper_url = f"https://openalex.org/{paper_id}"
                else:
                    paper_url = f"https://www.semanticscholar.org/paper/{paper_id}"
            else:
                paper_url = "https://www.semanticscholar.org"

        year = paper.get("year")
        year_str = str(year) if year is not None else "Year not specified"
        venue = paper.get("venue") or "Venue not specified"
        citation_count = paper.get("citationCount", 0)

        processed_papers.append({
            "paper_id": paper_id,
            "title": raw_title.strip(),
            "authors": authors_str,
            "year": year_str,
            "venue": venue,
            "citation_count": citation_count,
            "abstract": cleaned_abstract,
            "has_abstract": has_abstract,
            "url": paper_url,
            "source_provider": source_provider,
        })

        if len(processed_papers) >= requested_limit:
            break

    stats = {
        "total_raw": len(raw_papers),
        "total_returned": len(processed_papers),
        "duplicates_removed": duplicates_count,
        "missing_abstracts": missing_abstracts_count,
        "providers": providers_count,
    }

    return processed_papers, stats
