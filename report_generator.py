"""report_generator.py - Comprehensive research report compilation and file export module.

This module handles:
1. Academic reference citation formatting with clickable source URLs and provider attribution.
2. Assembly of full-length structured research dossiers in Markdown and Plain Text.
3. Explicit logging of data constraints (e.g., papers missing abstracts).
4. Automated persistence to local disk within the 'reports/' folder using unique timestamps.
"""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Converts a user topic into a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^\w\s-]", "", name).strip().lower()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:max_length] if slug else "research_report"


def format_reference_item(index: int, paper: Dict[str, Any]) -> str:
    """Formats an individual academic citation in APA-like style with clickable Markdown URL.

    Example output:
    [1] Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
        [Provider: Semantic Scholar] [View source](...)
    """
    authors = paper.get("authors", "Unknown Authors")
    year = paper.get("year", "n.d.")
    title = paper.get("title", "Untitled Paper")
    venue = paper.get("venue", "")
    venue_part = f" *{venue}*." if venue and venue != "Venue not specified" else ""
    citations = paper.get("citation_count", 0)
    url = paper.get("url", "https://www.semanticscholar.org")
    provider = paper.get("source_provider", "Semantic Scholar")

    citation_meta = f"Citations: {citations:,}" if citations else "Citations: 0"
    missing_tag = " *(Abstract metadata not provided)*" if not paper.get("has_abstract", False) else ""

    item_md = (
        f"**[{index}]** {authors} ({year}). *{title}*.{venue_part} "
        f"[{citation_meta}] [Provider: {provider}]{missing_tag}\n"
        f"   - **Source:** [{url}]({url})"
    )
    return item_md


def format_references_section(papers: List[Dict[str, Any]]) -> str:
    """Formats the comprehensive reference bibliography."""
    if not papers:
        return "No references available."

    lines = ["## References & Academic Sources\n"]
    for idx, paper in enumerate(papers, start=1):
        lines.append(format_reference_item(idx, paper))
        lines.append("")  # Blank line separator
    return "\n".join(lines)


def format_detailed_papers_catalog(papers: List[Dict[str, Any]]) -> str:
    """Builds a detailed catalog of retrieved papers including abstracts and metadata."""
    if not papers:
        return "*No papers retrieved.*"

    sections = ["## Retrieved Evidence Catalog\n"]
    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown Authors")
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")
        citations = paper.get("citation_count", 0)
        url = paper.get("url", "#")
        has_abstract = paper.get("has_abstract", False)
        abstract = paper.get("abstract", "")
        provider = paper.get("source_provider", "Semantic Scholar")

        status_badge = "✅ Abstract Available" if has_abstract else "⚠️ Metadata Only (No Abstract)"

        section = (
            f"### Paper [{idx}]: {title}\n\n"
            f"- **Authors:** {authors}\n"
            f"- **Year:** {year} | **Venue:** {venue} | **Citations:** {citations:,}\n"
            f"- **Search Provider:** {provider}\n"
            f"- **Status:** {status_badge}\n"
            f"- **Source URL:** [{url}]({url})\n\n"
            f"**Abstract / Scope:**\n>{abstract}\n"
        )
        sections.append(section)

    return "\n".join(sections)


def generate_markdown_report(
    topic: str,
    papers: List[Dict[str, Any]],
    ai_summary: str,
    stats: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> str:
    """Compiles the complete research report in professional Markdown format.

    Sections:
    1. Header & Metadata (Title, Date, Topic, Paper Counts, Search Providers)
    2. Executive Synthesis (AI generated: Summary, Findings, Limitations, Directions)
    3. Retrieved Evidence Catalog (All papers with abstracts & links)
    4. References & Academic Sources (Formatted numbered references)
    5. Agent Methodology & Limitations Disclosure
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = stats or {}
    total_returned = stats.get("total_returned", len(papers))
    missing_abstracts = stats.get("missing_abstracts", sum(1 for p in papers if not p.get("has_abstract", False)))
    duplicates_removed = stats.get("duplicates_removed", 0)

    providers_map = stats.get("providers", {})
    if providers_map:
        providers_summary = ", ".join(f"{prov} ({cnt})" for prov, cnt in providers_map.items())
    else:
        providers_summary = "Semantic Scholar"

    model_display = model_name or "Gemini API"

    lines = [
        f"# Smart Library Research Agent: Dossier on '{topic}'",
        "",
        "> **Notice:** This academic dossier was automatically compiled and synthesized by the ",
        "> Smart Library Research Agent using peer-reviewed metadata and ",
        f"> grounded AI synthesis via {model_display}.",
        "",
        "---",
        "",
        "## Report Metadata",
        "",
        f"- **Research Topic:** {topic}",
        f"- **Date Generated:** {timestamp_str}",
        f"- **Papers Analyzed:** {total_returned}",
        f"- **Search Provider(s):** {providers_summary}",
        f"- **Papers with Full Abstracts:** {total_returned - missing_abstracts}",
        f"- **Papers with Metadata Only:** {missing_abstracts}",
        f"- **Duplicates Filtered:** {duplicates_removed}",
        "",
        "---",
        "",
        "## AI Research Synthesis",
        "",
        ai_summary.strip() if ai_summary else "*AI summarization was not generated.*",
        "",
        "---",
        "",
        format_detailed_papers_catalog(papers),
        "",
        "---",
        "",
        format_references_section(papers),
        "",
        "---",
        "",
        "## Agent Methodology & Integrity Statement",
        "",
        "- **Data Integrity:** All paper metadata, titles, authors, and citation counts were retrieved ",
        "  directly from verified academic APIs (Semantic Scholar and/or OpenAlex) without AI fabrication or hallucination.",
        "- **Synthesis Grounding:** The AI synthesis was strictly conditioned on the retrieved abstracts and metadata.",
        f"- **Metadata Limitations:** For {missing_abstracts} paper(s) lacking publicly indexed abstracts, ",
        "  synthesis was restricted to titles and publication venues to maintain scholarly fidelity.",
        "",
        "---",
        f"*Generated by Smart Library Research Agent • {timestamp_str}*",
    ]

    return "\n".join(lines)


def generate_plain_text_report(
    topic: str,
    papers: List[Dict[str, Any]],
    ai_summary: str,
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a clean plain-text version of the research dossier."""
    md_content = generate_markdown_report(topic, papers, ai_summary, stats)
    # Strip markdown headers and bolding for clean plain text
    text = re.sub(r"#{1,6}\s*", "", md_content)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^>\s*", "  ", text, flags=re.MULTILINE)
    return text


def save_report_to_disk(
    report_content: str,
    topic: str,
    reports_dir: str = "reports",
    file_extension: str = "md",
) -> str:
    """Saves the research dossier to disk with a unique timestamped filename.

    Args:
        report_content: Markdown or text content to write.
        topic: Research topic used for the filename slug.
        reports_dir: Directory where reports should be saved.
        file_extension: 'md' or 'txt'.

    Returns:
        The relative or absolute file path to the saved report.
    """
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = sanitize_filename(topic)
    filename = f"research_report_{topic_slug}_{timestamp}.{file_extension}"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    return filepath
