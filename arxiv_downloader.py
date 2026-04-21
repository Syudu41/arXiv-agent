#!/usr/bin/env python3
"""
Downloads an arXiv paper by title or ID, extracts key sections, saves to ./papers/.

Usage:
    python arxiv_downloader.py "Attention is All You Need"
    python arxiv_downloader.py "1706.03762"

Stdout (for Claude to parse):
    PDF: ./papers/<arxiv_id>.pdf
    TEXT: ./papers/<arxiv_id>_extracted.txt
    TITLE: <paper title>
"""

import re
import sys
import os
import arxiv
import pdfplumber

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")

SECTION_KEYWORDS = [
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "approach", "model", "architecture",
    "experiment", "experiments", "evaluation", "results",
    "conclusion", "discussion",
]

SKIP_KEYWORDS = [
    "references", "bibliography", "appendix", "acknowledgement",
    "acknowledgment", "supplementary", "supplemental",
]

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def is_arxiv_id(query: str) -> bool:
    return bool(ARXIV_ID_PATTERN.match(query.strip()))


def slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:80]


def search_papers(query: str, max_results: int = 3) -> list:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return list(client.results(search))


def fetch_by_id(arxiv_id: str) -> list:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    return list(client.results(search))


def prompt_user_choice(results: list) -> arxiv.Result | None:
    print("\nFound the following papers:\n")
    for i, r in enumerate(results, 1):
        authors = ", ".join(a.name for a in r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        year = r.published.year if r.published else "?"
        print(f"  [{i}] {r.title}")
        print(f"       {authors} ({year}) — arXiv:{r.get_short_id()}\n")

    while True:
        choice = input("Select paper (1/2/3) or 'q' to cancel: ").strip().lower()
        if choice == "q":
            return None
        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            if idx < len(results):
                return results[idx]
        print("Invalid choice. Enter 1, 2, 3, or q.")


def is_section_heading(text: str) -> tuple[bool, bool]:
    """Returns (is_keep_section, is_skip_section)."""
    normalized = text.strip().lower()
    normalized = re.sub(r"^\d+[\.\s]+", "", normalized)  # strip leading numbering
    for kw in SKIP_KEYWORDS:
        if normalized.startswith(kw):
            return False, True
    for kw in SECTION_KEYWORDS:
        if normalized.startswith(kw):
            return True, False
    return False, False


def extract_key_sections(pdf_path: str) -> str:
    """Extract text from key sections only, skipping references/appendices."""
    sections = []
    current_section = []
    capturing = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                continue

            lines = {}
            for w in words:
                y = round(w["top"], 0)
                lines.setdefault(y, []).append(w["text"])

            for y in sorted(lines):
                line_text = " ".join(lines[y]).strip()
                if not line_text:
                    continue

                is_keep, is_skip = is_section_heading(line_text)

                if is_skip:
                    if capturing and current_section:
                        sections.append("\n".join(current_section))
                    capturing = False
                    current_section = []
                    continue

                if is_keep:
                    if capturing and current_section:
                        sections.append("\n".join(current_section))
                    capturing = True
                    current_section = [f"\n## {line_text}"]
                    continue

                if capturing:
                    current_section.append(line_text)

    if capturing and current_section:
        sections.append("\n".join(current_section))

    return "\n\n".join(sections)


def download_and_extract(result: arxiv.Result) -> tuple[str, str]:
    """Download PDF and extract text. Returns (pdf_path, text_path)."""
    os.makedirs(PAPERS_DIR, exist_ok=True)
    arxiv_id = result.get_short_id()
    pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")
    text_path = os.path.join(PAPERS_DIR, f"{arxiv_id}_extracted.txt")

    if not os.path.exists(pdf_path):
        print(f"\nDownloading {result.title}...")
        result.download_pdf(dirpath=PAPERS_DIR, filename=f"{arxiv_id}.pdf")
        print("Download complete.")
    else:
        print(f"\nPDF already exists: {pdf_path}")

    if not os.path.exists(text_path):
        print("Extracting key sections...")
        extracted = extract_key_sections(pdf_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {result.title}\n")
            f.write(f"AUTHORS: {', '.join(a.name for a in result.authors)}\n")
            f.write(f"YEAR: {result.published.year if result.published else 'unknown'}\n")
            f.write(f"ARXIV_ID: {arxiv_id}\n")
            f.write(f"VENUE: {result.journal_ref or 'arXiv preprint'}\n")
            f.write(f"ABSTRACT: {result.summary}\n\n")
            f.write("---\n\n")
            f.write(extracted)
        print("Extraction complete.")
    else:
        print(f"Extracted text already exists: {text_path}")

    return pdf_path, text_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python arxiv_downloader.py \"paper title or arXiv ID\"", file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()

    if is_arxiv_id(query):
        results = fetch_by_id(query)
        if not results:
            print(f"No paper found for arXiv ID: {query}", file=sys.stderr)
            sys.exit(1)
        selected = results[0]
    else:
        results = search_papers(query, max_results=3)
        if not results:
            print(f"No results found for: {query}", file=sys.stderr)
            sys.exit(1)
        selected = prompt_user_choice(results)
        if selected is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(0)

    pdf_path, text_path = download_and_extract(selected)

    print(f"\nPDF: {pdf_path}")
    print(f"TEXT: {text_path}")
    print(f"TITLE: {selected.title}")
    print(f"SLUG: {slugify(selected.title)}")


if __name__ == "__main__":
    main()
