# Research Agent — arXiv

A tool for first-year PhD students to read papers the way top-tier researchers do. Give it a paper title or arXiv ID; it downloads the paper and produces a structured `paper*.md` that doesn't just summarize — it models how a rigorous researcher interrogates a paper.

---

## Setup

Install Python dependencies (one-time):

```bash
pip install arxiv pdfplumber
```

No API key required. Analysis runs entirely in the current Claude Code session.

---

## Usage

```bash
/arxiv-agent "Attention is All You Need"
/arxiv-agent "1706.03762"
```

**What happens:**
1. Searches arXiv, shows top 3 results, asks you to confirm
2. Downloads the PDF to `./papers/`
3. Extracts key sections (Abstract, Intro, Method, Experiments, Conclusion)
4. Generates `./papers/paper_<title_slug>.md`

If the `.md` already exists, you'll be asked: skip / regenerate / append.

---

## Output: `paper*.md`

Each file follows a reader's journey arc — building understanding before critical evaluation. Sections use questions as headers to train the interrogative habit:

| Section | What it builds |
|---------|---------------|
| What problem is this solving? | Field context, gap identification |
| What is the core claim? | Hypothesis vs. implementation distinction |
| How does the method work? | Intuition + annotated key equations |
| What do the results prove? | Evidence vs. claim separation |
| Strengths and gaps? | Critical evaluation, reviewer mindset |
| Where does this live in the field? | Positioning, influence, lineage |
| What's left open? | Research direction identification |
| How would you re-implement this? | Reproducibility, underspecified details |
| Key Terms to Own | Vocabulary with intuition-first definitions |

Every section contains:
- **`> Researcher's Move:`** — inline callouts naming the reasoning pattern being used
- **`### Think Like a Researcher`** — 2-3 prompts you could answer yourself to internalize the habit

---

## Registering the Skill

To make `/arxiv-agent` available as a slash command in Claude Code, add it to your settings:

```json
// .claude/settings.json
{
  "skills": [
    ".agents/skills/arxiv-agent"
  ]
}
```

Or add globally in `~/.claude/settings.json` to use it across all projects.

---

## Directory Layout

```
Research-Agent-arXiv/
├── CLAUDE.md                    ← this file
├── arxiv_downloader.py          ← download + section extraction script
├── papers/                      ← all outputs live here
│   ├── 1706.03762.pdf           ← downloaded PDF
│   ├── 1706.03762_extracted.txt ← key sections extracted from PDF
│   └── paper_attention_is_all_you_need.md ← generated analysis
└── .agents/
    └── skills/
        └── arxiv-agent/
            └── SKILL.md         ← skill instructions for Claude
```

---

## Running the Downloader Directly

```bash
python arxiv_downloader.py "Transformers are all you need"
python arxiv_downloader.py "1706.03762"
```

Outputs to stdout:
```
PDF: ./papers/1706.03762.pdf
TEXT: ./papers/1706.03762_extracted.txt
TITLE: Attention Is All You Need
SLUG: attention_is_all_you_need
```

---

## Philosophy

The goal is not to replace reading the paper — it's to teach you *how* to read it. Each `paper*.md` is a worked example of research reasoning. Over time, you run the questions in your head without the scaffold.
