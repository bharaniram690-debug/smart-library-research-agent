# 📚 Smart Library Research Agent

An autonomous, multi-tool academic research assistant that performs end-to-end literature discovery, metadata validation, and grounded AI literature synthesis with enterprise-grade rate-limiting resiliency.

Built with **Python**, **Streamlit**, the **Semantic Scholar Graph API**, the **OpenAlex API**, and the **Google Gemini API**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Rate Limiting, Retries & Fallback Architecture](#-rate-limiting-retries--fallback-architecture)
- [Mandatory Requirements Compliance Matrix](#-mandatory-requirements-compliance-matrix)
- [Project Structure](#-project-structure)
- [Quick Start: Installation & Setup](#-quick-start-installation--setup)
- [Automated Test Suite & Demonstrations](#-automated-test-suite--demonstrations)
- [Running the Streamlit Web Application](#-running-the-streamlit-web-application)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [Design Decisions & Ethical AI Principles](#-design-decisions--ethical-ai-principles)

---

## 🌟 Overview

The **Smart Library Research Agent** automates what typically takes hours of manual scholarly investigation:
1. Validating academic topics and constraints.
2. Formulating and refining search queries.
3. Querying peer-reviewed literature from the Semantic Scholar database with automated backoff and retries.
4. Automatically falling back to the OpenAlex academic API if Semantic Scholar encounters HTTP 429 rate-limiting.
5. In-memory query caching to eliminate redundant API requests.
6. Deduplicating records, reconstructing inverted-index abstracts, and auditing metadata.
7. Synthesizing cross-study findings using Google Gemini with strict grounding to eliminate hallucinations.
8. Generating publication-ready dossiers and archiving them to disk with unique timestamps.

---

## 🔄 System Architecture & Workflow

The agent executes a multi-provider autonomous pipeline with zero manual intervention:

```text
  [ User Input ] (Topic & Max Paper Count)
         │
         ▼
[ 1. Input Validation ] ───(Invalid)───► [ Display Actionable Error ]
         │ (Valid)
         ▼
[ 2. Cache Inspection ] ───(Cache Hit)───► [ Return Cached Dossier ]
         │ (Cache Miss)
         ▼
[ 3. Primary Search: Semantic Scholar ]
         │
         ├───► HTTP 200 (Success) ──────────────────────────┐
         └───► HTTP 429 / Congestion                        │
                 │                                          │
                 ├── Retries with Exponential Backoff       │
                 │   (Respects 'Retry-After' up to 3 times) │
                 │                                          │
                 └── Still 429? ────────────────┐           │
                                                ▼           │
                               [ 4. Fallback Search:       │
                                    OpenAlex API ]         │
                                                │           │
                                                ▼           ▼
                                     [ 5. Deduplication & Audit ]
                                                │
                                                ▼
                                     [ 6. Gemini Synthesis ]
                                                │
                                                ▼
                                     [ 7. Reference Compilation ]
                                                │
                                                ▼
                                     [ 8. Report Export & Disk Save ]
```

---

## ⚡ Rate Limiting, Retries & Fallback Architecture

### 1. Documented Header Authentication
- Reads `SEMANTIC_SCHOLAR_API_KEY` from `.env` securely.
- Automatically sends the key in the HTTP request header: `x-api-key: <KEY>`.

### 2. Exponential Backoff & Retry-After Respect
- When Semantic Scholar responds with **HTTP 429**:
  - The agent inspects `response.headers.get("Retry-After")`.
  - If a numeric wait time is provided (e.g., `4.0`), the agent waits that exact duration.
  - If `Retry-After` is missing, the agent applies exponential backoff: $1.0\text{s} \times 2^{\text{attempt}}$ (capped at 60s).

### 3. Maximum 3 Retries Limit
- Retries are strictly capped at **3 automatic attempts** (1 initial + 3 retries = 4 total requests).
- Prevents infinite loops or aggressive server hammering.

### 4. Search Query Caching
- Successful searches are cached in an in-memory dictionary indexed by `(cleaned_query, limit, provider)`.
- Re-running the same search retrieves results in 0ms with zero external network requests.
- Users can clear the cache anytime using the sidebar button.

### 5. Multi-Provider Fallback to OpenAlex
- If Semantic Scholar remains unavailable or rate-limited after 3 retries, the agent queries **OpenAlex** (`https://api.openalex.org/works`).
- Reconstructs abstracts faithfully from OpenAlex's `abstract_inverted_index`.
- Each paper clearly displays its verified provider tag (`Semantic Scholar` or `OpenAlex`).

### 6. Prevention of Duplicate Button Clicks
- Streamlit session state tracks `is_searching`. The primary action button is disabled while a query is running.
- Redundant queries with identical inputs return the existing session state rather than re-querying.

### 7. Friendly Error Guidance
- If both Semantic Scholar and OpenAlex fail, a helpful message explains the issue, gives recovery steps, and suggests setting `SEMANTIC_SCHOLAR_API_KEY`.

---

## 📋 Mandatory Requirements Compliance Matrix

| Requirement | Implementation Detail in Codebase |
| :--- | :--- |
| **1. Secure Key Loading** | - Loaded via `python-dotenv` from `.env`.<br>- Included in `.gitignore` to prevent committing to Git. |
| **2. Documented Auth** | - `get_semantic_scholar_headers()` sends `x-api-key: <KEY>` header. |
| **3. Backoff & Retry-After** | - Handled in `search_semantic_scholar()`. Inspects `Retry-After` header or calculates $1.0\text{s} \times 2^{\text{attempt}}$. |
| **4. Max 3 Retries** | - `max_retries = 3` enforced in retry loop (1 initial + 3 retries). |
| **5. Search Caching** | - `get_cached_search()` and `set_cached_search()` prevent duplicate API requests for identical queries. |
| **6. Prevent Duplicate Clicks** | - Streamlit UI disables the button while processing (`disabled=st.session_state.is_searching`) and tracks previous query keys. |
| **7. OpenAlex Fallback** | - `search_openalex()` queries OpenAlex works API; tags papers with `source_provider="OpenAlex"`. |
| **8. Zero Fabrication** | - Preserves raw metadata; faithfully reconstructs abstracts from inverted index; notes missing abstracts explicitly. |
| **9. Friendly Error Message** | - Unified error handler alerts the user with next steps when both providers fail. |
| **10. Mocked Tests** | - 16 comprehensive unit and integration tests in `tests/test_agent.py` covering rate limits, backoff, retry limits, caching, and fallback. |

---

## 📁 Project Structure

```text
smart_library_research_agent/
│
├── app.py                     # Streamlit web app with caching, OpenAlex fallback, and tabs
├── research_search.py         # Multi-provider search, backoff retries, cache, and OpenAlex client
├── summarizer.py              # Gemini API prompt builder & literature synthesis
├── report_generator.py        # Dossier compilation, reference formatting with provider tags
├── run_tests.py               # Standalone test runner utility
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Secrets, cache, and virtual environment ignore list
├── README.md                  # Comprehensive documentation and guide
│
├── reports/                   # Output folder for generated research dossiers
│   └── sample_research_report_ai_in_agriculture.md
│
└── tests/
    ├── __init__.py
    └── test_agent.py          # Pytest suite with mock test cases A, B, C, rate limit, cache, and fallback
```

---

## 🚀 Quick Start: Installation & Setup

Open PowerShell and run:

### Step 1: Navigate to Project Directory
```powershell
cd C:\Users\nprhp\.gemini\antigravity\scratch\smart_library_research_agent
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables in `.env`
```ini
# Required for AI Synthesis
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Semantic Scholar API Key (provides dedicated throughput to avoid 429)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here

# Default Gemini model
GEMINI_MODEL=gemini-1.5-flash
```

---

## 🧪 Automated Test Suite & Demonstrations

Run the automated test suite with:

```powershell
pytest tests/test_agent.py -v
```
*Or with the standalone script:*
```powershell
python run_tests.py
```

### Complete Test Coverage Breakdown:

| Test Class | Tests Included | Expected Result |
| :--- | :--- | :--- |
| **`TestCaseAValidTopic`** | • `test_input_validation_passes`<br>• `test_query_cleaning`<br>• `test_semantic_scholar_search_success`<br>• `test_paper_filtering_and_deduplication`<br>• `test_gemini_summarization_success`<br>• `test_end_to_end_report_generation_and_saving` | Input valid, 1 duplicate removed, missing abstract tagged, 4 synthesis sections generated, report saved to disk. |
| **`TestCaseBInvalidInput`** | • `test_empty_string_topic`<br>• `test_whitespace_only_topic`<br>• `test_too_short_topic`<br>• `test_symbols_only_topic`<br>• `test_paper_count_out_of_bounds` | All reject with descriptive validation errors without triggering API calls. |
| **`TestCaseCFailuresAndNoResults`** | • `test_zero_search_results_scenario`<br>• `test_semantic_scholar_timeout_scenario`<br>• `test_semantic_scholar_rate_limit_429`<br>• `test_semantic_scholar_malformed_json`<br>• `test_gemini_missing_api_key_error`<br>• `test_gemini_empty_papers_list_rejection`<br>• `test_gemini_rate_limit_429_scenario`<br>• `test_gemini_timeout_scenario` | Gracefully handled with user-friendly errors; zero crashes. |
| **`TestRateLimitingAndResiliency`** | • `test_send_api_key_in_headers_when_configured`<br>• `test_retry_after_header_parsing`<br>• `test_semantic_scholar_respects_retry_after_header`<br>• `test_semantic_scholar_exponential_backoff_without_retry_after`<br>• `test_max_retries_limit_enforced_at_3_attempts`<br>• `test_search_caching_prevents_duplicate_api_calls`<br>• `test_openalex_abstract_reconstruction_from_inverted_index`<br>• `test_fallback_to_openalex_when_semantic_scholar_rate_limited`<br>• `test_friendly_message_when_all_providers_fail` | `x-api-key` sent, `Retry-After` sleep obeyed, 3 retries enforced, 0 duplicate calls when cached, OpenAlex fallback activated, friendly failure message. |

---

## 🖥️ Running the Streamlit Web Application

To start the interactive research agent:

```powershell
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.
