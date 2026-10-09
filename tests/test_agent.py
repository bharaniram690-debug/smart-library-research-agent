"""tests/test_agent.py - Comprehensive automated test suite for Smart Library Research Agent.

Mandatory Demonstrations & Rate Limit Tests:
- Requirement 7.A: Valid topic "Artificial Intelligence in Agriculture"
- Requirement 7.B: Invalid input: empty research topic and boundary limits
- Requirement 7.C: No-results and simulated API failure scenarios (timeouts, 429 rate limits, missing keys)
- Rate Limit & Resiliency:
  * Sending x-api-key header when configured
  * Exponential backoff and Retry-After header parsing
  * Limiting retries to maximum 3 attempts
  * Search result caching to prevent redundant API calls
  * Fallback to OpenAlex API when Semantic Scholar is unavailable
  * Friendly error message when all providers fail
  * OpenAlex inverted index abstract reconstruction

All external API interactions (Semantic Scholar, OpenAlex, and Gemini) are mocked using unittest.mock.
"""

import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, call, patch
import pytest
import requests

# Ensure project root is available in module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from report_generator import (  # noqa: E402
    format_reference_item,
    format_references_section,
    generate_markdown_report,
    generate_plain_text_report,
    save_report_to_disk,
)
from research_search import (  # noqa: E402
    clean_search_query,
    clear_search_cache,
    filter_and_process_papers,
    get_cached_search,
    get_semantic_scholar_headers,
    parse_retry_after,
    reconstruct_abstract_from_inverted_index,
    search_academic_papers,
    search_openalex,
    search_semantic_scholar,
    validate_search_inputs,
)
from summarizer import (  # noqa: E402
    build_synthesis_prompt,
    summarize_research_papers,
)


# ==============================================================================
# FIXTURES & MOCK DATA
# ==============================================================================

@pytest.fixture(autouse=True)
def clean_cache_between_tests():
    """Ensures search cache is completely cleared between test runs."""
    clear_search_cache()
    yield
    clear_search_cache()


@pytest.fixture
def sample_raw_papers():
    """Mock Semantic Scholar Graph API response payload for 'Artificial Intelligence in Agriculture'."""
    return [
        {
            "paperId": "s2_paper_001",
            "title": "Deep Learning Architectures for Crop Disease Detection",
            "authors": [{"name": "Amina Patel"}, {"name": "Carlos Gomez"}],
            "year": 2023,
            "venue": "Computers and Electronics in Agriculture",
            "citationCount": 47,
            "abstract": (
                "Precision agriculture increasingly relies on convolutional neural networks "
                "to identify crop pathologies from aerial multispectral imagery."
            ),
            "url": "https://www.semanticscholar.org/paper/s2_paper_001",
            "source_provider": "Semantic Scholar",
        },
        {
            "paperId": "s2_paper_002",
            "title": "Autonomous Robotics and Yield Prediction in Precision Farming",
            "authors": [{"name": "Hiroshi Tanaka"}],
            "year": 2024,
            "venue": "Journal of Agricultural Robotics",
            "citationCount": 18,
            "abstract": (
                "We demonstrate autonomous mobile ground robots that forecast crop yield "
                "with 92% accuracy across varied soil moisture profiles."
            ),
            "url": "https://www.semanticscholar.org/paper/s2_paper_002",
            "source_provider": "Semantic Scholar",
        },
        {
            # Paper missing abstract to test requirement 2 & 5
            "paperId": "s2_paper_003",
            "title": "Sensor Networks for Smart Irrigation Scheduling",
            "authors": [{"name": "Elena Rostova"}],
            "year": 2022,
            "venue": "Agricultural Water Management",
            "citationCount": 35,
            "abstract": None,
            "url": "https://www.semanticscholar.org/paper/s2_paper_003",
            "source_provider": "Semantic Scholar",
        },
        {
            # Duplicate of paper 001 to test deduplication
            "paperId": "s2_paper_001",
            "title": "Deep Learning Architectures for Crop Disease Detection",
            "authors": [{"name": "Amina Patel"}],
            "year": 2023,
            "venue": "Computers and Electronics in Agriculture",
            "citationCount": 47,
            "abstract": "Duplicate record",
            "url": "https://www.semanticscholar.org/paper/s2_paper_001",
            "source_provider": "Semantic Scholar",
        },
    ]


@pytest.fixture
def sample_openalex_works_response():
    """Mock OpenAlex API response payload."""
    return {
        "meta": {"count": 2},
        "results": [
            {
                "id": "https://openalex.org/W2981234567",
                "title": "Robotic Harvesting in High-Density Orchards",
                "publication_year": 2023,
                "cited_by_count": 29,
                "primary_location": {
                    "source": {"display_name": "Frontiers in Plant Science"},
                    "landing_page_url": "https://doi.org/10.3389/fpls.2023.001",
                },
                "doi": "https://doi.org/10.3389/fpls.2023.001",
                "authorships": [
                    {"author": {"display_name": "Li Wei"}},
                    {"author": {"display_name": "Thomas Mueller"}},
                ],
                "abstract_inverted_index": {
                    "Automated": [0],
                    "vision-guided": [1],
                    "fruit": [2],
                    "picking": [3],
                    "reduces": [4],
                    "orchard": [5],
                    "labor": [6],
                    "costs.": [7],
                },
            },
            {
                "id": "https://openalex.org/W2989876543",
                "title": "Soil Moisture Neural Forecasting",
                "publication_year": 2024,
                "cited_by_count": 12,
                "primary_location": {
                    "source": {"display_name": "Computers and Electronics in Agriculture"},
                    "landing_page_url": "https://doi.org/10.1016/j.compag.2024.002",
                },
                "doi": "https://doi.org/10.1016/j.compag.2024.002",
                "authorships": [
                    {"author": {"display_name": "Fatima Zahra"}},
                ],
                "abstract_inverted_index": None,
            },
        ],
    }


@pytest.fixture
def sample_gemini_synthesis_response():
    """Mock Gemini API JSON response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "## 1. Executive Summary\n"
                                "Recent literature in artificial intelligence for agriculture demonstrates rapid "
                                "adoption of deep learning for crop monitoring and autonomous yield forecasting [1], [2].\n\n"
                                "## 2. Key Findings & Contributions\n"
                                "- CNN architectures achieve high precision in crop disease diagnosis [1].\n"
                                "- Autonomous field robots improve yield forecasting accuracy to 92% [2].\n\n"
                                "## 3. Limitations & Research Gaps\n"
                                "Sensor networks [3] lack full abstract metadata, indicating a need for open benchmark sets. "
                                "Field robotics face battery and rugged terrain constraints [2].\n\n"
                                "## 4. Future Research Directions\n"
                                "Integration of multimodal satellite feeds with ground rovers for unified irrigation."
                            )
                        }
                    ]
                }
            }
        ]
    }


# ==============================================================================
# TEST CASE A: VALID TOPIC ("Artificial Intelligence in Agriculture")
# ==============================================================================

class TestCaseAValidTopic:
    """Requirement 7.A: Valid topic test case."""

    def test_input_validation_passes(self):
        """Valid topic and valid paper count should pass validation."""
        topic = "Artificial Intelligence in Agriculture"
        is_valid, error = validate_search_inputs(topic, max_papers=5)
        assert is_valid is True
        assert error == ""

    def test_query_cleaning(self):
        """Verifies academic search query formatting."""
        raw_query = '  "Artificial Intelligence in Agriculture"  '
        cleaned = clean_search_query(raw_query)
        assert cleaned == "Artificial Intelligence in Agriculture"

    @patch("requests.get")
    def test_semantic_scholar_search_success(self, mock_get, sample_raw_papers):
        """Tests that Semantic Scholar API returns papers and parses correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total": 4, "data": sample_raw_papers}
        mock_get.return_value = mock_response

        res = search_semantic_scholar("Artificial Intelligence in Agriculture", limit=5)
        assert res["status"] == "success"
        assert len(res["data"]) == 4
        assert res["error"] == ""
        assert res["source_provider"] == "Semantic Scholar"

    def test_paper_filtering_and_deduplication(self, sample_raw_papers):
        """Verifies duplicate removal and handling of missing abstracts."""
        processed, stats = filter_and_process_papers(sample_raw_papers, requested_limit=5)

        # Total unique papers should be 3 (one duplicate removed)
        assert stats["total_returned"] == 3
        assert stats["duplicates_removed"] == 1
        assert stats["missing_abstracts"] == 1
        assert stats["providers"]["Semantic Scholar"] == 3

        # Check paper with missing abstract retains metadata
        missing_abs_paper = [p for p in processed if p["paper_id"] == "s2_paper_003"][0]
        assert missing_abs_paper["has_abstract"] is False
        assert "Notice: Abstract not provided" in missing_abs_paper["abstract"]
        assert missing_abs_paper["title"] == "Sensor Networks for Smart Irrigation Scheduling"
        assert missing_abs_paper["authors"] == "Elena Rostova"
        assert missing_abs_paper["source_provider"] == "Semantic Scholar"

    @patch("requests.post")
    def test_gemini_summarization_success(self, mock_post, sample_raw_papers, sample_gemini_synthesis_response):
        """Tests that Gemini API returns structured synthesis using mock."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_gemini_synthesis_response
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers, requested_limit=5)
        result = summarize_research_papers(
            topic="Artificial Intelligence in Agriculture",
            papers=processed,
            api_key="mock_test_key_12345",
        )

        assert result["status"] == "success"
        assert "Executive Summary" in result["summary"]
        assert "Key Findings" in result["summary"]
        assert "Limitations" in result["summary"]
        assert "Future Research Directions" in result["summary"]

    def test_end_to_end_report_generation_and_saving(self, sample_raw_papers, tmp_path):
        """Verifies full Markdown and TXT report assembly and file persistence."""
        processed, stats = filter_and_process_papers(sample_raw_papers, requested_limit=5)
        mock_ai_summary = "## 1. Executive Summary\nAI is transforming modern precision agriculture."

        md_report = generate_markdown_report(
            topic="Artificial Intelligence in Agriculture",
            papers=processed,
            ai_summary=mock_ai_summary,
            stats=stats,
        )

        assert "Smart Library Research Agent: Dossier on 'Artificial Intelligence in Agriculture'" in md_report
        assert "Deep Learning Architectures for Crop Disease Detection" in md_report
        assert "References & Academic Sources" in md_report
        assert "Provider: Semantic Scholar" in md_report

        saved_file = save_report_to_disk(
            report_content=md_report,
            topic="Artificial Intelligence in Agriculture",
            reports_dir=str(tmp_path),
            file_extension="md",
        )

        assert os.path.exists(saved_file)
        with open(saved_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Artificial Intelligence in Agriculture" in content


# ==============================================================================
# TEST CASE B: INVALID INPUT (Empty topic, bad counts)
# ==============================================================================

class TestCaseBInvalidInput:
    """Requirement 7.B: Invalid input handling."""

    def test_empty_string_topic(self):
        """Empty string must fail validation."""
        is_valid, error = validate_search_inputs("", max_papers=5)
        assert is_valid is False
        assert "cannot be empty" in error.lower()

    def test_whitespace_only_topic(self):
        """Whitespace-only string must fail validation."""
        is_valid, error = validate_search_inputs("    \t\n  ", max_papers=5)
        assert is_valid is False
        assert "cannot be empty" in error.lower()

    def test_too_short_topic(self):
        """Topics with fewer than 3 characters must fail validation."""
        is_valid, error = validate_search_inputs("AI", max_papers=5)
        assert is_valid is False
        assert "too short" in error.lower()

    def test_symbols_only_topic(self):
        """Non-alphanumeric queries must fail validation."""
        is_valid, error = validate_search_inputs("!@#$%^&*()", max_papers=5)
        assert is_valid is False
        assert "alphanumeric" in error.lower()

    @pytest.mark.parametrize("bad_count", [0, -1, 21, 100])
    def test_paper_count_out_of_bounds(self, bad_count):
        """Paper counts outside allowed range must fail validation."""
        is_valid, error = validate_search_inputs("Machine Learning", max_papers=bad_count)
        assert is_valid is False
        assert "between 1 and 20" in error


# ==============================================================================
# TEST CASE C: NO-RESULTS OR SIMULATED API FAILURE SCENARIOS
# ==============================================================================

class TestCaseCFailuresAndNoResults:
    """Requirement 7.C: No results and simulated failures (timeouts, 429 rate limits, missing keys)."""

    @patch("requests.get")
    def test_zero_search_results_scenario(self, mock_get):
        """Semantic Scholar returns empty data list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total": 0, "data": []}
        mock_get.return_value = mock_response

        res = search_semantic_scholar("NonexistentScientificTopicXYZ12345", limit=5)
        assert res["status"] == "empty"
        assert res["data"] == []
        assert "No relevant academic research papers found" in res["error"]

    @patch("requests.get")
    def test_semantic_scholar_timeout_scenario(self, mock_get):
        """Simulate Semantic Scholar network timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        res = search_semantic_scholar("Artificial Intelligence", limit=5, timeout=2, sleep_func=lambda s: None)
        assert res["status"] == "error"
        assert "timed out" in res["error"]

    @patch("requests.get")
    def test_semantic_scholar_rate_limit_429(self, mock_get):
        """Simulate Semantic Scholar HTTP 429 rate limit exhausting retries."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_get.return_value = mock_response

        res = search_semantic_scholar("Artificial Intelligence", limit=5, sleep_func=lambda s: None)
        assert res["status"] == "error"
        assert "rate limit exceeded" in res["error"].lower()
        assert res.get("rate_limited") is True

    @patch("requests.get")
    def test_semantic_scholar_malformed_json(self, mock_get):
        """Simulate malformed non-JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        res = search_semantic_scholar("Artificial Intelligence", limit=5)
        assert res["status"] == "error"
        assert "malformed" in res["error"].lower()

    def test_gemini_missing_api_key_error(self, sample_raw_papers, monkeypatch):
        """Simulate missing GEMINI_API_KEY environment variable."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence in Agriculture",
            papers=processed,
            api_key=None,
        )
        assert res["status"] == "error"
        assert "API key is missing" in res["error"]

    def test_gemini_empty_papers_list_rejection(self):
        """Agent must never send empty paper list to Gemini."""
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=[],
            api_key="valid_key",
        )
        assert res["status"] == "error"
        assert "No research papers available" in res["error"]

    @patch("requests.post")
    def test_gemini_rate_limit_429_scenario(self, mock_post, sample_raw_papers):
        """Simulate Gemini HTTP 429 quota exhaustion."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence in Agriculture",
            papers=processed,
            api_key="mock_key",
        )
        assert res["status"] == "error"
        assert "quota exceeded" in res["error"].lower() or "rate limit" in res["error"].lower()

    @patch("requests.post")
    def test_gemini_timeout_scenario(self, mock_post, sample_raw_papers):
        """Simulate Gemini HTTP request timeout."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence in Agriculture",
            papers=processed,
            api_key="mock_key",
            timeout=5,
        )
        assert res["status"] == "error"
        assert "timed out" in res["error"]


# ==============================================================================
# RATE LIMITING, RETRIES, CACHING & OPENALEX FALLBACK TESTS
# ==============================================================================

class TestRateLimitingAndResiliency:
    """Verifies HTTP 429 backoff, retry limits, headers, caching, and OpenAlex fallback."""

    def test_send_api_key_in_headers_when_configured(self):
        """Verifies documented x-api-key header format."""
        headers = get_semantic_scholar_headers(api_key="my_secret_s2_key")
        assert headers.get("x-api-key") == "my_secret_s2_key"
        assert "User-Agent" in headers

    def test_retry_after_header_parsing(self):
        """Parses numeric Retry-After header and falls back gracefully."""
        assert parse_retry_after("5", default_backoff=1.0) == 5.0
        assert parse_retry_after("2.5", default_backoff=1.0) == 2.5
        assert parse_retry_after(None, default_backoff=1.0) == 1.0
        assert parse_retry_after("invalid_header", default_backoff=1.0) == 1.0

    @patch("requests.get")
    def test_semantic_scholar_respects_retry_after_header(self, mock_get, sample_raw_papers):
        """HTTP 429 with Retry-After header should trigger sleeping for the specified duration then succeed."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "4.0"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"total": 2, "data": sample_raw_papers[:2]}

        mock_get.side_effect = [resp_429, resp_200]
        sleep_calls = []

        res = search_semantic_scholar(
            query="Deep Learning",
            limit=2,
            sleep_func=lambda s: sleep_calls.append(s),
            use_cache=False,
        )

        assert res["status"] == "success"
        assert len(res["data"]) == 2
        assert mock_get.call_count == 2
        assert sleep_calls == [4.0]

    @patch("requests.get")
    def test_semantic_scholar_exponential_backoff_without_retry_after(self, mock_get, sample_raw_papers):
        """HTTP 429 without Retry-After header uses exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"total": 1, "data": sample_raw_papers[:1]}

        mock_get.side_effect = [resp_429, resp_429, resp_200]
        sleep_calls = []

        res = search_semantic_scholar(
            query="Neural Networks",
            limit=1,
            sleep_func=lambda s: sleep_calls.append(s),
            use_cache=False,
        )

        assert res["status"] == "success"
        assert mock_get.call_count == 3
        # Should have slept twice before success
        assert len(sleep_calls) == 2
        assert sleep_calls[1] > sleep_calls[0]  # Exponential increase

    @patch("requests.get")
    def test_max_retries_limit_enforced_at_3_attempts(self, mock_get):
        """Requests must stop after 3 retries (1 initial + 3 retries = 4 total requests)."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}
        mock_get.return_value = resp_429

        sleep_calls = []
        res = search_semantic_scholar(
            query="Robotics",
            limit=3,
            max_retries=3,
            sleep_func=lambda s: sleep_calls.append(s),
            use_cache=False,
        )

        assert res["status"] == "error"
        assert res.get("rate_limited") is True
        # 1 initial request + 3 retries = 4 calls
        assert mock_get.call_count == 4
        # Slept exactly 3 times before failing
        assert len(sleep_calls) == 3

    @patch("requests.get")
    def test_search_caching_prevents_duplicate_api_calls(self, mock_get, sample_raw_papers):
        """Second call for identical query is retrieved from cache with zero additional HTTP requests."""
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"total": 2, "data": sample_raw_papers[:2]}
        mock_get.return_value = resp_200

        # First call: hits API
        res1 = search_academic_papers("Precision Irrigation", limit=2, use_cache=True)
        assert res1["status"] == "success"
        assert mock_get.call_count == 1
        assert res1.get("cached") is False

        # Second call: served from cache
        res2 = search_academic_papers("Precision Irrigation", limit=2, use_cache=True)
        assert res2["status"] == "success"
        assert mock_get.call_count == 1  # No new network call!
        assert res2.get("cached") is True

    def test_openalex_abstract_reconstruction_from_inverted_index(self):
        """Inverted index is reconstructed back into correctly ordered text."""
        inverted_index = {
            "Autonomous": [0],
            "drones": [1],
            "survey": [2],
            "fields": [3],
            "rapidly.": [4],
        }
        text = reconstruct_abstract_from_inverted_index(inverted_index)
        assert text == "Autonomous drones survey fields rapidly."

        # None or empty inverted index
        assert reconstruct_abstract_from_inverted_index(None) is None
        assert reconstruct_abstract_from_inverted_index({}) is None

    @patch("requests.get")
    def test_fallback_to_openalex_when_semantic_scholar_rate_limited(
        self, mock_get, sample_openalex_works_response
    ):
        """When Semantic Scholar fails with HTTP 429, automatically queries OpenAlex."""
        # Semantic Scholar always 429
        s2_429 = MagicMock()
        s2_429.status_code = 429
        s2_429.headers = {}

        # OpenAlex succeeds
        openalex_200 = MagicMock()
        openalex_200.status_code = 200
        openalex_200.json.return_value = sample_openalex_works_response

        # S2 max retries (4 calls) then OpenAlex (1 call)
        mock_get.side_effect = [s2_429, s2_429, s2_429, s2_429, openalex_200]

        res = search_academic_papers(
            query="Robotic Harvesting",
            limit=2,
            max_retries=3,
            enable_fallback=True,
            sleep_func=lambda s: None,
            use_cache=False,
        )

        assert res["status"] == "success"
        assert res.get("fallback_used") is True
        assert res.get("source_provider") == "OpenAlex"
        assert len(res["data"]) == 2

        # Check OpenAlex paper structure
        first_paper = res["data"][0]
        assert first_paper["source_provider"] == "OpenAlex"
        assert first_paper["title"] == "Robotic Harvesting in High-Density Orchards"
        assert "Automated vision-guided fruit picking" in first_paper["abstract"]
        assert first_paper["citationCount"] == 29

    @patch("requests.get")
    def test_friendly_message_when_all_providers_fail(self, mock_get):
        """When both Semantic Scholar and OpenAlex fail, show actionable friendly message."""
        s2_429 = MagicMock()
        s2_429.status_code = 429
        s2_429.headers = {}

        openalex_500 = MagicMock()
        openalex_500.status_code = 500
        openalex_500.raise_for_status.side_effect = requests.exceptions.HTTPError("OpenAlex 500 Internal Error")

        mock_get.side_effect = [s2_429, s2_429, s2_429, s2_429, openalex_500]

        res = search_academic_papers(
            query="Difficult Academic Query",
            limit=3,
            max_retries=3,
            enable_fallback=True,
            sleep_func=lambda s: None,
            use_cache=False,
        )

        assert res["status"] == "error"
        assert res.get("all_providers_failed") is True
        assert "Both academic search providers" in res["error"]
        assert "SEMANTIC_SCHOLAR_API_KEY" in res["error"]


# ==============================================================================
# GEMINI MODEL & ERROR HANDLING TESTS (gemini-3.8-flash, 404, 400, 403, Redaction)
# ==============================================================================

class TestGeminiApiErrorsAndModelSupport:
    """Verifies Gemini API model selection (gemini-3.8-flash), endpoint format, header auth, and error handling."""

    @patch("requests.post")
    def test_gemini_uses_gemini_3_8_flash_endpoint_and_header_auth(
        self, mock_post, sample_raw_papers, sample_gemini_synthesis_response
    ):
        """Verifies endpoint is /v1beta/models/{model}:generateContent and x-goog-api-key header is sent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_gemini_synthesis_response
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        secret_key = "test_gemini_key_xyz"
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=processed,
            api_key=secret_key,
            model="gemini-3.8-flash",
        )

        assert res["status"] == "success"
        assert res["model_used"] == "gemini-3.8-flash"

        # Assert requests.post call args
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        call_headers = mock_post.call_args[1].get("headers", {})

        # Documented endpoint format
        assert call_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
        # API key sent securely via header
        assert call_headers.get("x-goog-api-key") == secret_key
        # API key NEVER in URL
        assert secret_key not in call_url

    @patch("requests.post")
    def test_gemini_http_404_model_not_found_handled_gracefully(self, mock_post, sample_raw_papers):
        """HTTP 404 Model Not Found returns informative advice without crashing."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=processed,
            api_key="valid_key",
            model="gemini-1.5-flash",
        )

        assert res["status"] == "error"
        assert "HTTP 404" in res["error"]
        assert "not found" in res["error"].lower()

    @patch("requests.post")
    def test_gemini_http_400_bad_request(self, mock_post, sample_raw_papers):
        """HTTP 400 Bad Request returns clean error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid argument: maxOutputTokens"}}
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=processed,
            api_key="valid_key",
        )

        assert res["status"] == "error"
        assert "HTTP 400" in res["error"]
        assert "Invalid argument" in res["error"]

    @patch("requests.post")
    def test_gemini_http_403_forbidden_invalid_key(self, mock_post, sample_raw_papers):
        """HTTP 403 Forbidden returns advice on checking GEMINI_API_KEY."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": {"message": "API key not valid"}}
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=processed,
            api_key="invalid_key",
        )

        assert res["status"] == "error"
        assert "HTTP 403" in res["error"]
        assert "GEMINI_API_KEY" in res["error"]

    @patch("requests.post")
    def test_gemini_never_leaks_api_key_in_error_message(self, mock_post, sample_raw_papers):
        """Ensures raw API keys are never leaked in error messages or logs."""
        secret_key = "SUPER_SECRET_GEMINI_KEY_999"
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = f"Internal server error occurred with key={secret_key}"
        mock_response.json.side_effect = ValueError("Non-JSON 500 response")
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"Internal server error occurred with key={secret_key}"
        )
        mock_post.return_value = mock_response

        processed, _ = filter_and_process_papers(sample_raw_papers)
        res = summarize_research_papers(
            topic="Artificial Intelligence",
            papers=processed,
            api_key=secret_key,
        )

        assert res["status"] == "error"
        assert secret_key not in res["error"]
        assert "[REDACTED_API_KEY]" in res["error"]

