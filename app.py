"""app.py - Streamlit User Interface for the Smart Library Research Agent.

This application provides an automated academic research pipeline:
1. INPUT: Accept topic and maximum paper count with real-time validation.
2. DECISION-MAKING: Validates queries, deduplicates papers, audits metadata, caches queries.
3. TOOL USAGE: Semantic Scholar API (with backoff & 3 retries), OpenAlex API (fallback), Gemini API.
4. AUTOMATION: End-to-end execution from topic input to local report persistence.
5. ERROR HANDLING: Graceful handling of HTTP 429 rate limits, timeouts, missing keys, and zero results.
6. FINAL OUTPUT: Tabbed interactive results, provider attribution, and Markdown/TXT downloads.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file if present
load_dotenv()

from research_search import (  # noqa: E402
    clean_search_query,
    clear_search_cache,
    filter_and_process_papers,
    search_academic_papers,
    validate_search_inputs,
    MAX_PAPERS_ALLOWED,
    MIN_PAPERS_ALLOWED,
)
from report_generator import (  # noqa: E402
    generate_markdown_report,
    generate_plain_text_report,
    save_report_to_disk,
)
from summarizer import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    summarize_research_papers,
)

# Page configuration
st.set_page_config(
    page_title="Smart Library Research Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .paper-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-s2 {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-openalex {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-cached {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "is_searching" not in st.session_state:
        st.session_state.is_searching = False
    if "last_search_key" not in st.session_state:
        st.session_state.last_search_key = ""


init_session_state()

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.markdown("Configure API credentials and search providers.")

    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_gemini_key,
        type="password",
        help="Required for AI summarization. Get one from Google AI Studio (https://aistudio.google.com).",
    )

    env_s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    s2_key_input = st.text_input(
        "Semantic Scholar API Key",
        value=env_s2_key,
        type="password",
        help="Optional. Semantic Scholar is free without a key, but an API key provides higher rate limits.",
    )

    enable_openalex_fallback = st.checkbox(
        "Enable OpenAlex Fallback",
        value=True,
        help="If Semantic Scholar encounters HTTP 429 rate-limiting, automatically search OpenAlex as a fallback provider.",
    )

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    current_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    selected_model = st.selectbox(
        "Gemini Synthesis Model",
        options=model_options,
        index=model_options.index(current_model) if current_model in model_options else 0,
        help="Select which Gemini model to use for literature review synthesis.",
    )

    st.markdown("---")
    st.subheader("⚡ Search Cache")
    if st.button("🗑️ Clear Search Cache", use_container_width=True):
        clear_search_cache()
        st.success("Search cache cleared successfully!")

    st.markdown("---")
    st.subheader("📂 Saved Reports")
    reports_dir = Path("reports")
    if reports_dir.exists():
        existing_reports = sorted(reports_dir.glob("*.md"), reverse=True)
        if existing_reports:
            st.caption(f"Found {len(existing_reports)} generated report(s):")
            for rep in existing_reports[:5]:
                st.code(rep.name, language="text")
        else:
            st.caption("No reports saved yet.")
    else:
        st.caption("Reports directory will be created upon first research run.")

    st.markdown("---")
    st.caption("Smart Library Research Agent v1.1 • Multi-Provider Search with Backoff & Caching")


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">📚 Smart Library Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous academic literature search with rate-limit resiliency, '
    'OpenAlex fallback, and grounded AI literature synthesis.</div>',
    unsafe_allow_html=True,
)

# Input section
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Research Topic",
            placeholder="e.g., Artificial Intelligence in Agriculture, Quantum Machine Learning, CRISPR Gene Editing",
            help="Enter any scientific or academic research topic.",
        )
    with col2:
        paper_count = st.slider(
            "Max Papers",
            min_value=MIN_PAPERS_ALLOWED,
            max_value=MAX_PAPERS_ALLOWED,
            value=5,
            help=f"Number of research papers to retrieve ({MIN_PAPERS_ALLOWED}-{MAX_PAPERS_ALLOWED}).",
        )

    # Quick example chips
    st.caption("Quick example topics:")
    chip_cols = st.columns(4)
    if chip_cols[0].button("🌱 AI in Agriculture", use_container_width=True):
        topic_input = "Artificial Intelligence in Agriculture"
    if chip_cols[1].button("⚛️ Quantum Computing", use_container_width=True):
        topic_input = "Quantum Computing Optimization"
    if chip_cols[2].button("🧬 CRISPR Therapeutics", use_container_width=True):
        topic_input = "CRISPR Cas9 Therapeutics"
    if chip_cols[3].button("🤖 Large Language Models", use_container_width=True):
        topic_input = "Large Language Models Hallucination Mitigation"

    start_research = st.button(
        "🚀 Run Research Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_searching,
    )


# ----------------- AUTOMATED PIPELINE EXECUTION -----------------
if start_research:
    # STEP 1: INPUT VALIDATION
    is_valid, validation_msg = validate_search_inputs(topic_input, paper_count)
    if not is_valid:
        st.error(f"❌ Input Validation Error: {validation_msg}")
    else:
        current_search_key = f"{clean_search_query(topic_input).lower()}::{paper_count}"

        # Prevent duplicate execution caused by repeated button clicks
        if st.session_state.is_searching:
            st.warning("⏳ A research request is currently processing. Please wait for completion.")
            st.stop()

        st.session_state.is_searching = True

        try:
            status_container = st.status("🔍 Executing Autonomous Research Pipeline...", expanded=True)

            with status_container:
                # STEP 2: SEARCH QUERY SANITIZATION
                st.write("1️⃣ **Normalizing Search Query & Parameters**...")
                cleaned_query = clean_search_query(topic_input)
                st.caption(f"Refined Academic Query: `{cleaned_query}` | Target count: {paper_count}")

                # STEP 3: MULTI-PROVIDER SEARCH (SEMANTIC SCHOLAR WITH RETRIES -> OPENALEX FALLBACK)
                st.write("2️⃣ **Querying Academic Literature Providers**...")
                search_response = search_academic_papers(
                    query=cleaned_query,
                    limit=paper_count * 2,
                    api_key=s2_key_input,
                    enable_fallback=enable_openalex_fallback,
                )

                if search_response.get("cached"):
                    st.write("⚡ **Served from query cache** (prevented redundant external API calls).")

                if search_response.get("fallback_used"):
                    st.write("🌐 **Semantic Scholar rate-limited or unavailable; retrieved results via OpenAlex.**")

                if search_response["status"] == "error":
                    status_container.update(label="❌ Pipeline Halted: Search Providers Failed", state="error", expanded=True)
                    st.error(search_response["error"])
                    if search_response.get("all_providers_failed"):
                        st.info(
                            "💡 **Troubleshooting Advice:**\n"
                            "- Wait 30-60 seconds for API rate limits to reset.\n"
                            "- Set a `SEMANTIC_SCHOLAR_API_KEY` in `.env` or the sidebar for dedicated quota.\n"
                            "- Try broadening or refining your search topic keywords."
                        )
                    st.stop()

                if search_response["status"] == "empty":
                    status_container.update(label="⚠️ No Research Papers Found", state="complete", expanded=True)
                    st.warning(search_response["error"])
                    st.info(
                        "💡 Suggestions:\n"
                        "- Try using broader scientific terms.\n"
                        "- Verify spelling of domain keywords.\n"
                        "- Check if specialized acronyms need expansion."
                    )
                    st.stop()

                raw_papers = search_response["data"]
                st.write(f"Found {len(raw_papers)} preliminary paper records from {search_response.get('source_provider', 'academic provider')}.")

                # STEP 4: FILTERING, DEDUPLICATION & METADATA AUDIT
                st.write("3️⃣ **Filtering Duplicates and Auditing Metadata**...")
                processed_papers, stats = filter_and_process_papers(raw_papers, requested_limit=paper_count)

                if not processed_papers:
                    status_container.update(label="⚠️ No Valid Papers Remained After Filtering", state="complete")
                    st.warning("All returned records lacked valid titles or essential metadata.")
                    st.stop()

                st.write(
                    f"Retained **{stats['total_returned']}** unique papers "
                    f"({stats['duplicates_removed']} duplicates removed)."
                )

                if stats["missing_abstracts"] > 0:
                    st.write(
                        f"⚠️ Note: **{stats['missing_abstracts']}** paper(s) lack public abstracts. "
                        "Continuing synthesis using verified title and venue metadata."
                    )

                # STEP 5: AI LITERATURE SYNTHESIS VIA GEMINI
                st.write("4️⃣ **Synthesizing Academic Literature with Google Gemini**...")
                if not gemini_key_input.strip():
                    status_container.update(label="⚠️ Gemini API Key Missing", state="error", expanded=True)
                    st.error(
                        "Gemini API key is required for AI synthesis. Please provide your key in the sidebar "
                        "or set `GEMINI_API_KEY` in your `.env` file."
                    )
                    st.stop()

                ai_result = summarize_research_papers(
                    topic=topic_input,
                    papers=processed_papers,
                    api_key=gemini_key_input,
                    model=selected_model,
                )

                if ai_result["status"] == "error":
                    status_container.update(label="❌ Synthesis Failed", state="error", expanded=True)
                    st.error(f"AI Synthesis Error: {ai_result['error']}")
                    st.stop()

                ai_summary = ai_result["summary"]
                st.write("✅ AI Literature Synthesis complete.")

                # STEP 6: REPORT COMPILATION & LOCAL PERSISTENCE
                st.write("5️⃣ **Compiling Dossier & Saving to Disk**...")
                markdown_report = generate_markdown_report(
                    topic=topic_input,
                    papers=processed_papers,
                    ai_summary=ai_summary,
                    stats=stats,
                    model_name=ai_result.get("model_used"),
                )
                plain_text_report = generate_plain_text_report(
                    topic=topic_input,
                    papers=processed_papers,
                    ai_summary=ai_summary,
                    stats=stats,
                )

                saved_path = save_report_to_disk(
                    report_content=markdown_report,
                    topic=topic_input,
                    reports_dir="reports",
                    file_extension="md",
                )

                save_report_to_disk(
                    report_content=plain_text_report,
                    topic=topic_input,
                    reports_dir="reports",
                    file_extension="txt",
                )

                st.write(f"💾 Saved report locally to: `{saved_path}`")
                status_container.update(label="🎉 Research Pipeline Completed Successfully!", state="complete")

                # Store in session state for tabbed display
                st.session_state.research_results = {
                    "topic": topic_input,
                    "papers": processed_papers,
                    "stats": stats,
                    "ai_summary": ai_summary,
                    "markdown_report": markdown_report,
                    "plain_text_report": plain_text_report,
                    "saved_path": saved_path,
                    "model_used": ai_result.get("model_used", selected_model),
                    "search_source": search_response.get("source_provider", "Academic Provider"),
                    "cached": search_response.get("cached", False),
                }
                st.session_state.last_search_key = current_search_key

        finally:
            st.session_state.is_searching = False


# ----------------- DISPLAY RESEARCH RESULTS -----------------
if st.session_state.research_results:
    res = st.session_state.research_results
    papers = res["papers"]
    stats = res["stats"]

    st.markdown("---")
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"📑 Research Dossier: *{res['topic']}*")
    with header_col2:
        if res.get("cached"):
            st.markdown('<span class="badge-cached">⚡ Served from Cache</span>', unsafe_allow_html=True)

    # Metrics summary cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Papers Analyzed", stats["total_returned"])
    with m_col2:
        st.metric("Abstracts Available", stats["total_returned"] - stats["missing_abstracts"])
    with m_col3:
        st.metric("Metadata-Only Papers", stats["missing_abstracts"])
    with m_col4:
        providers_text = ", ".join(f"{k} ({v})" for k, v in stats.get("providers", {}).items()) or "Semantic Scholar"
        st.metric("Provider Breakdown", providers_text)

    if stats["missing_abstracts"] > 0:
        st.info(
            f"ℹ️ **Scholarly Limitation Note:** {stats['missing_abstracts']} of {stats['total_returned']} "
            "papers do not provide publicly indexed abstracts from publisher feeds. "
            "The AI agent synthesized these entries using their verified titles, authors, and venue metadata."
        )

    # Tabbed view for organized consumption
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 AI Research Synthesis",
        "📚 Retrieved Papers & Evidence",
        "🔗 Academic References",
        "📥 Export & Full Report",
    ])

    with tab1:
        st.markdown(res["ai_summary"])

    with tab2:
        for idx, paper in enumerate(papers, start=1):
            provider_name = paper.get("source_provider", "Semantic Scholar")
            badge_class = "badge-s2" if provider_name == "Semantic Scholar" else "badge-openalex"

            with st.container():
                title_col, badge_col = st.columns([4, 1])
                with title_col:
                    st.markdown(f"#### [{idx}] {paper['title']}")
                with badge_col:
                    st.markdown(f'<span class="{badge_class}">🏛️ {provider_name}</span>', unsafe_allow_html=True)

                meta_cols = st.columns([2, 1, 1])
                with meta_cols[0]:
                    st.caption(f"👤 **Authors:** {paper['authors']}")
                with meta_cols[1]:
                    st.caption(f"📅 **Year:** {paper['year']} | 🏛️ **Venue:** {paper['venue']}")
                with meta_cols[2]:
                    citations = paper["citation_count"]
                    st.caption(f"📈 **Citations:** {citations:,}")

                if paper["has_abstract"]:
                    st.markdown(f"**Abstract:**\n{paper['abstract']}")
                else:
                    st.warning(f"⚠️ {paper['abstract']}")

                st.markdown(f"🔗 **Source Link:** [{paper['url']}]({paper['url']})")
                st.divider()

    with tab3:
        st.markdown("### Formal Academic References")
        for idx, paper in enumerate(papers, start=1):
            missing_text = " *(Metadata only - abstract unavailable)*" if not paper["has_abstract"] else ""
            provider = paper.get("source_provider", "Semantic Scholar")
            st.markdown(
                f"**[{idx}]** {paper['authors']} ({paper['year']}). *{paper['title']}*. "
                f"{paper['venue']}. Citations: {paper['citation_count']:,}. [Provider: {provider}]{missing_text}\n\n"
                f"Source: [{paper['url']}]({paper['url']})"
            )
            st.write("")

    with tab4:
        st.markdown("### Export Research Dossier")
        st.success(f"✅ Dossier automatically archived to: `{res['saved_path']}`")

        down_col1, down_col2 = st.columns(2)
        with down_col1:
            st.download_button(
                label="⬇️ Download Markdown (.md)",
                data=res["markdown_report"],
                file_name=f"research_report_{clean_search_query(res['topic']).replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with down_col2:
            st.download_button(
                label="⬇️ Download Plain Text (.txt)",
                data=res["plain_text_report"],
                file_name=f"research_report_{clean_search_query(res['topic']).replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with st.expander("📄 View Full Raw Markdown Preview"):
            st.code(res["markdown_report"], language="markdown")
