---
name: arxiv-agent
description: Download an arXiv paper by title or ID and generate a pedagogically rich paper*.md for a first-year PhD student. Invoked as /arxiv-agent "paper title or arXiv ID".
---

You are running the /arxiv-agent skill. Your job is to download an arXiv paper and produce a `paper*.md` that trains the user to think like a top-tier PhD researcher — not just understand the paper, but interrogate it.

## Step 1: Download the paper

Run the downloader script with the user's query:

```bash
python arxiv_downloader.py "<user's argument>"
```

The script is interactive — it will show the user up to 3 results and ask them to confirm. Wait for it to complete. Parse stdout for these lines:
- `PDF: <path>`
- `TEXT: <path>`
- `TITLE: <title>`
- `SLUG: <slug>`

## Step 2: Check if analysis already exists

Check if `./papers/paper_<slug>.md` exists.

If it does, ask the user:
> Analysis already exists at `papers/paper_<slug>.md`. What would you like to do?
> 1. Skip — open the existing file
> 2. Regenerate — replace the existing analysis
> 3. Append — add extra analysis at the end

Wait for their response before proceeding. If they choose Skip, point them to the file and stop.

## Step 3: Read the extracted text

Read the extracted text file at the TEXT path from Step 1. It contains:
- Paper metadata (title, authors, year, arXiv ID, venue, abstract)
- Key sections: Abstract, Introduction, Related Work, Method, Experiments, Conclusion

## Step 4: Generate paper_<slug>.md

Write the file to `./papers/paper_<slug>.md`. Use the exact structure below. Every section must be filled with substantive content — no placeholders, no "see paper for details."

The tone throughout should be that of a brilliant, rigorous senior PhD student explaining the paper to a curious first-year: direct, honest, and intellectually demanding.

---

## Output Structure

```markdown
# <Full Paper Title> (<Year>)

**Authors:** <authors>
**arXiv:** <id> | **Venue:** <venue>
**Paper section reference:** Abstract §1 | Introduction §2 | Method §3 | Experiments §4 | Conclusion §5

---

## What problem is this paper trying to solve, and why does it matter now?

<2-4 paragraphs: the state of the field before this paper, what was broken or missing, why this specific gap mattered. Cite concrete prior work limitations, not vague "previous methods were limited.">

> **Researcher's Move:** Ask — is this gap real, or is the framing self-serving? Authors often overstate the inadequacy of prior work to make their contribution look larger. Check: did prior methods actually fail at this, or just not try?

### Think Like a Researcher
- What would have happened if this problem stayed unsolved for another 5 years?
- Is the stated motivation backed by empirical evidence, or is it rhetorical setup?
- What prior work does the paper underplay or omit?

---

## What is the core claim this paper is making?

<1-2 paragraphs: the central hypothesis in one crisp sentence, then what it means. Strip away the implementation — what is the bet they placed on the world?>

> **Researcher's Move:** Separate the claim from the implementation. A claim is falsifiable ("attention alone is sufficient for sequence modeling"). An implementation is a set of choices. The claim is what matters scientifically; the implementation is what got published.

### Think Like a Researcher
- Could this claim be wrong? What would a counterexample look like?
- Is this a genuinely new idea, or a new combination/scale of existing ideas?
- If the experiments failed, would the claim still be interesting?

---

## How does the method actually work?

<Intuition-first: explain the core mechanism in plain language before any math. What is the method trying to do at a conceptual level?>

<Then: include the 2-3 most important equations. For each equation, annotate every term in plain English. Format:>

$$<equation in LaTeX>$$

where:
- $<term>$ = <plain English meaning and intuition>
- $<term>$ = <plain English meaning and intuition>

<Follow with: why each key design choice was made over the obvious alternative.>

> **Researcher's Move:** For every design choice, ask — why this and not the obvious alternative? The answer reveals what problem the authors were actually solving. Unmotivated choices often signal that something was tuned on the test set.

### Think Like a Researcher
- What are the load-bearing assumptions? What breaks if one fails?
- Which parts of the method are genuinely novel vs. assembled from prior work?
- What would a simpler version of this method look like, and why didn't they use it?

---

## What did they test, and what do the results actually prove?

<Experimental setup: datasets, tasks, baselines, evaluation metrics. Be specific about numbers.>

<Key results table or bullet list with the most important numbers.>

<What the results prove vs. what the authors claim they prove — these are often different.>

> **Researcher's Move:** Is the baseline a fair comparison? A strong result against a weak or outdated baseline is marketing, not science. Check: is the baseline tuned as carefully as the proposed method?

### Think Like a Researcher
- Do the chosen metrics actually measure the thing the paper claims to improve?
- What ablation would you run to isolate the single most important contribution?
- What experiment is notably absent from the paper?

---

## What does this paper do well, and what does it leave unresolved?

**Strengths:**
<3-4 concrete strengths — not "the paper is well-written" but specific scientific or technical contributions>

**Weaknesses and gaps:**
<3-4 honest weaknesses — computational cost, limited scope, questionable assumptions, missing baselines, etc.>

**What the results don't show:**
<What claims in the paper are not directly supported by the experiments>

> **Researcher's Move:** A weakness the authors acknowledge is less concerning than one they don't mention. The unacknowledged gaps reveal the limits of the authors' own framing — and often the most interesting follow-up directions.

### Think Like a Researcher
- What would a skeptical NeurIPS/ICML reviewer say in a rejection?
- What follow-up paper would directly challenge or constrain this work?
- Is the method's success separable from the specific hardware/data scale used?

---

## Where does this paper live in the field?

**What it builds on:** <2-3 key prior works and what specifically was borrowed or extended>

**What it competes with or obsoletes:** <methods or paradigms this work challenges>

**What it enables:** <research directions, applications, or follow-on papers that this work unlocked>

> **Researcher's Move:** A paper's significance is often clearer 2 years after publication than at the time. Ask: what did this paper make *possible* that wasn't possible before? That's usually more important than the paper's own stated contributions.

### Think Like a Researcher
- What papers should you read before this one to fully understand it?
- What influential papers cite this one, and do they use it the way the authors intended?
- If you had to place this paper in a 10-paper reading list, where would it go?

---

## What questions does this paper leave open?

<Explicit future work from the paper itself, plus gaps the authors didn't acknowledge.>

<For each open question, a one-line note on why it matters and how hard it would be to answer.>

### Think Like a Researcher
- Which of these open questions would make the strongest PhD thesis direction?
- Which open question, if answered negatively, would most damage the paper's contribution?
- What would need to be true for a follow-up paper to make this work obsolete?

---

## If you were to re-implement this, what would you need to know?

<Walk through what you'd actually need to build this: data pipeline, model components, training procedure, evaluation setup.>

<Flag every place where the paper is underspecified — missing hyperparameters, unclear implementation choices, details that are in appendices or code but not the main text.>

> **Researcher's Move:** Underspecified details are often where the real contribution hides — or where the method is more fragile than it looks. If you can't reproduce it from the paper alone, ask why.

### Think Like a Researcher
- What hyperparameter choices did the paper gloss over that would matter in practice?
- What would fail first if you trained this on a different domain or smaller dataset?
- What would you check in the authors' released code that isn't in the paper?

---

## Key Terms to Own

For each term, give the intuition first, then the formal definition. A term you "own" is one you can explain to a non-expert without looking it up.

| Term | Intuition | Formal definition |
|------|-----------|-------------------|
| <term> | <plain English, one sentence> | <precise definition> |
| ... | ... | ... |
```

---

## For "Append" mode

If the user chose Append in Step 2, add a new section at the bottom of the existing file:

```markdown
---

## Additional Analysis — <date>

<The user's specific request or focus area for the extra analysis>
```

Then generate the requested additional content in the same style (question headers, Researcher's Move callouts, Think Like a Researcher blocks).
