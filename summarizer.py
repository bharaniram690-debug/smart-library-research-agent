"""summarizer.py - Gemini API research paper synthesis and summarization module.

This module handles:
1. Retrieval of Gemini API credentials securely from environment variables.
2. Grounded academic literature synthesis prompt generation.
3. Enforces strict zero-hallucination and non-empty paper list rules.
4. HTTP communication with the Google Gemini REST API via /v1beta/models/{model}:generateContent.
5. Secure header-based authentication (x-goog-api-key) so API keys are NEVER exposed in URLs or logs.
6. Robust error handling for HTTP 400, 403, 404 (model not found), 429 (rate limits), and timeouts.
"""

import os
import re
from typing import Any, Dict, List, Optional
import requests

# Base Gemini API endpoint template (v1beta) - adheres to documented format
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Currently supported text-generation model
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Network timeout for Gemini synthesis
DEFAULT_GEMINI_TIMEOUT_SECONDS = 30


def sanitize_error_message(message: Any, sensitive_key: Optional[str] = None) -> str:
    """Sanitizes error strings to ensure API keys are never leaked in error messages or logs."""
    if message is None:
        return ""
    # Ensure message is a string (handles MagicMock or non-str gracefully)
    sanitized = str(message) if not isinstance(message, str) else message
    if sensitive_key and sensitive_key in sanitized:
        sanitized = sanitized.replace(sensitive_key, "[REDACTED_API_KEY]")
    # Strip any potential key query parameters
    sanitized = re.sub(r"key=[a-zA-Z0-9_\-\.]{10,}", "key=[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"x-goog-api-key:\s*[a-zA-Z0-9_\-\.]+", "x-goog-api-key: [REDACTED]", sanitized)
    return sanitized


def extract_error_message_from_response(response: Any) -> str:
    """Safely extracts error message string from response object."""
    # Attempt parsing structured JSON error
    try:
        data = response.json()
        if isinstance(data, dict):
            err_dict = data.get("error")
            if isinstance(err_dict, dict):
                msg = err_dict.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            elif isinstance(err_dict, str) and err_dict.strip():
                return err_dict.strip()
    except Exception:
        pass

    # Fallback to response.text
    try:
        raw_text = response.text
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text.strip()
    except Exception:
        pass

    return f"HTTP {getattr(response, 'status_code', 'Unknown')}"


def build_synthesis_prompt(topic: str, papers: List[Dict[str, Any]]) -> str:
    """Constructs a grounded, anti-hallucination prompt for the Gemini model.

    Decision-Making Rule:
    - Instructs the model to only synthesize findings present in the supplied papers.
    - Explicitly marks papers where abstracts were unavailable so Gemini notes the limitation.
    - Directs output into structured sections: Summary, Key Findings, Limitations, Future Directions.
    - Demands reference numbers matching [1], [2], etc.
    """
    papers_context_blocks = []
    has_any_missing_abstract = False

    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown Authors")
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")
        has_abstract = paper.get("has_abstract", False)
        abstract = paper.get("abstract", "")
        provider = paper.get("source_provider", "Academic Provider")

        if not has_abstract:
            has_any_missing_abstract = True
            abstract_text = (
                "[Notice: Full abstract unavailable from publisher metadata. "
                "Base analysis solely on title, authors, and venue.]"
            )
        else:
            abstract_text = abstract

        papers_context_blocks.append(
            f"[{idx}] Title: {title}\n"
            f"    Authors: {authors}\n"
            f"    Year: {year} | Venue: {venue} | Source: {provider}\n"
            f"    Abstract/Metadata: {abstract_text}"
        )

    papers_text = "\n\n".join(papers_context_blocks)

    missing_notice = ""
    if has_any_missing_abstract:
        missing_notice = (
            "\nNote: One or more papers have limited metadata (abstract unavailable). "
            "Explicitly acknowledge this data limitation in the Limitations section "
            "and do NOT invent findings for them.\n"
        )

    prompt = f"""You are a professional academic research assistant and literature review expert.
You have been tasked with synthesizing academic literature on the topic: "{topic}".

Below is the verified evidence base consisting of {len(papers)} retrieved scientific papers:

{papers_text}

=== INSTRUCTIONS & RIGOR RULES ===
1. GROUNDED EVIDENCE: Base your synthesis ONLY on the research titles and abstracts provided above. Never invent, hallucinate, or fabricate research results, authors, or statistics.
2. CITATION DISCIPLINE: Refer to specific papers using their bracketed reference numbers like [1], [2], etc.
3. STRUCTURED REPORT: Format your response in clean Markdown with the following four exact H2 headers:
   ## 1. Executive Summary
   A cohesive 2-3 paragraph synthesis explaining what these papers collectively reveal about "{topic}".
   ## 2. Key Findings & Contributions
   A structured bulleted list of the major scientific findings, methodologies, and technological contributions across the papers, citing sources [1], [2], etc.
   ## 3. Limitations & Research Gaps
   Discussion of methodological constraints, sample sizes, and practical challenges noted in the studies.{missing_notice}
   ## 4. Future Research Directions
   Promising research frontiers, unaddressed questions, and next steps supported by the analyzed literature.

Proceed with the synthesis:"""

    return prompt


def summarize_research_papers(
    topic: str,
    papers: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_GEMINI_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generates an academic research synthesis using the Gemini API.

    Adheres to:
    - Documentation endpoint: /v1beta/models/{model}:generateContent
    - Model selection: defaults to gemini-3.8-flash (configurable via GEMINI_MODEL env var)
    - Secure header auth (x-goog-api-key) ensuring keys never appear in URLs
    - Graceful error handling for HTTP 400, 403, 404, 429, and timeouts without crashing

    Args:
        topic: The research topic being summarized.
        papers: Processed list of paper dictionaries.
        api_key: Optional Gemini API key override (otherwise loaded from env).
        model: Model name (defaults to GEMINI_MODEL env var or 'gemini-3.8-flash').
        timeout: HTTP request timeout in seconds.

    Returns:
        Dictionary containing:
        - 'status': 'success' | 'error'
        - 'summary': Synthesized Markdown text (if success)
        - 'error': User-friendly error message (if error)
        - 'model_used': Model name used
    """
    # Rule: Do not send an empty paper list to Gemini
    if not papers:
        return {
            "status": "error",
            "summary": "",
            "error": "No research papers available to summarize. Synthesis cannot proceed on an empty paper list.",
            "model_used": "",
        }

    # Resolve API Key securely from argument or environment variable
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not resolved_api_key:
        return {
            "status": "error",
            "summary": "",
            "error": (
                "Gemini API key is missing. Please set the GEMINI_API_KEY environment variable "
                "in your .env file or enter it in the application sidebar to generate AI summaries."
            ),
            "model_used": "",
        }

    # Resolve model: defaults to currently supported model gemini-3.8-flash
    resolved_model = model or os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    prompt = build_synthesis_prompt(topic=topic, papers=papers)

    # Documented endpoint format: /v1beta/models/{model}:generateContent
    endpoint_url = GEMINI_BASE_URL.format(model=resolved_model)

    # Send API key via documented HTTP header 'x-goog-api-key' so it is NEVER part of the URL query string
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": resolved_api_key,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "maxOutputTokens": 2500,
        },
    }

    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        # Handle HTTP 404 Model Not Found specifically
        if response.status_code == 404:
            return {
                "status": "error",
                "summary": "",
                "error": (
                    f"Gemini model '{resolved_model}' not found (HTTP 404). "
                    f"The specified model ID may not be supported by your API key or is inactive. "
                    f"Please verify that '{resolved_model}' is supported in Google AI Studio or select another model."
                ),
                "model_used": resolved_model,
            }

        # Handle HTTP 429 Rate Limits / Quota Exceeded
        if response.status_code == 429:
            return {
                "status": "error",
                "summary": "",
                "error": (
                    "Gemini API rate limit or quota exceeded (HTTP 429). "
                    "Please wait a moment before trying again or check your Google AI Studio quota."
                ),
                "model_used": resolved_model,
            }

        # Handle HTTP 400 Bad Request
        if response.status_code == 400:
            error_msg = extract_error_message_from_response(response)
            clean_error = sanitize_error_message(error_msg, resolved_api_key)
            return {
                "status": "error",
                "summary": "",
                "error": f"Gemini API Bad Request (HTTP 400): {clean_error}",
                "model_used": resolved_model,
            }

        # Handle HTTP 403 Forbidden / Invalid Key
        if response.status_code in (401, 403):
            error_msg = extract_error_message_from_response(response)
            clean_error = sanitize_error_message(error_msg, resolved_api_key)
            return {
                "status": "error",
                "summary": "",
                "error": (
                    f"Gemini API Access Forbidden / Invalid Key (HTTP {response.status_code}): {clean_error}. "
                    "Please check that your GEMINI_API_KEY is valid and has permission to access the Gemini API."
                ),
                "model_used": resolved_model,
            }

        # Handle any other non-200 status codes (HTTP 500, 502, 503, etc.)
        if response.status_code != 200:
            error_msg = extract_error_message_from_response(response)
            clean_error = sanitize_error_message(error_msg, resolved_api_key)
            return {
                "status": "error",
                "summary": "",
                "error": f"Gemini API Error (HTTP {response.status_code}): {clean_error}",
                "model_used": resolved_model,
            }

        response.raise_for_status()

        # Parse JSON response
        try:
            data = response.json()
        except ValueError:
            return {
                "status": "error",
                "summary": "",
                "error": "Received malformed non-JSON response from Gemini API.",
                "model_used": resolved_model,
            }

        candidates = data.get("candidates", [])
        if not candidates:
            return {
                "status": "error",
                "summary": "",
                "error": "Gemini API returned no response candidates. Content may have been filtered.",
                "model_used": resolved_model,
            }

        first_candidate = candidates[0]
        content_parts = first_candidate.get("content", {}).get("parts", [])
        if not content_parts or not content_parts[0].get("text"):
            return {
                "status": "error",
                "summary": "",
                "error": "Gemini API returned an empty text response.",
                "model_used": resolved_model,
            }

        summary_text = content_parts[0]["text"].strip()
        return {
            "status": "success",
            "summary": summary_text,
            "error": "",
            "model_used": resolved_model,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "summary": "",
            "error": (
                f"Gemini API request timed out after {timeout} seconds. "
                "The server took too long to generate a response."
            ),
            "model_used": resolved_model,
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "summary": "",
            "error": "Failed to connect to the Gemini API endpoint. Please check your network connection.",
            "model_used": resolved_model,
        }
    except requests.exceptions.HTTPError as http_err:
        clean_err = sanitize_error_message(str(http_err), resolved_api_key)
        return {
            "status": "error",
            "summary": "",
            "error": f"Gemini API HTTP error: {clean_err}",
            "model_used": resolved_model,
        }
    except Exception as exc:
        clean_err = sanitize_error_message(str(exc), resolved_api_key)
        return {
            "status": "error",
            "summary": "",
            "error": f"Unexpected error during AI summarization: {clean_err}",
            "model_used": resolved_model,
        }
