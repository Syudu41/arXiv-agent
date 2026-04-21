# arXiv Research Agent

A Claude Code skill that downloads arXiv papers and generates structured reading guides for first-year PhD students — not summaries, but worked examples of how a rigorous researcher interrogates a paper.

## How it works

Useage: 
```
/arxiv-agent "Attention is All You Need"  
```

You can also use via DOI like (for more accuracy) -
```
/arxiv-agent "1706.03762"  
```

1. Searches arXiv, shows top 3 matches, you confirm
2. Downloads the PDF and extracts key sections (Abstract → Conclusion, skipping references)
3. Claude reads the extracted text and writes `papers/paper_<title>.md`

## Output

Each `paper*.md` follows a reader's journey arc with question-driven section headers:

- *What problem is this solving, and why now?*
- *What is the core claim?*
- *How does the method work?* — with key equations annotated line by line
- *What do the results actually prove?*
- *Strengths and gaps?*
- *Where does this live in the field?*
- *What questions does it leave open?*
- *How would you re-implement this?*
- *Key Terms to Own*

Every section includes inline **Researcher's Move** callouts (naming the reasoning pattern being used) and **Think Like a Researcher** prompts you can answer yourself to internalize the habit.

## Setup

```bash
pip install arxiv pdfplumber
```
> [!CAUTION]
> Claude Code works on Claude PRO or beyond subscription! 

No API key needed — analysis runs in your current Claude Code session.

Register the skill in `.claude/settings.json` (already done in this repo):

```json
{ "skills": [".agents/skills/arxiv-agent"] }
```

## Usage

```bash
/arxiv-agent "paper title"      # fuzzy title search
/arxiv-agent "1706.03762"       # direct arXiv ID lookup
```

If the analysis already exists, you'll be asked to skip, regenerate, or append.

## Structure

```
├── arxiv_downloader.py          # search, download, extract
├── papers/                      # PDFs, extracted text, and paper*.md outputs
└── .agents/skills/arxiv-agent/
    └── SKILL.md                 # skill instructions for Claude
```
