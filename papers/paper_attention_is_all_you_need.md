# Attention Is All You Need (2017)

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**arXiv:** 1706.03762 | **Venue:** NeurIPS 2017
**Paper section reference:** Abstract §0 | Introduction §1 | Background §2 | Model Architecture §3 | Why Self-Attention §4 | Training §5 | Results §6 | Conclusion §7

---

## What problem is this paper trying to solve, and why does it matter now?

By 2017, the dominant architecture for sequence-to-sequence tasks — machine translation, summarization, parsing — was the recurrent neural network (RNN), particularly LSTMs and GRUs in encoder-decoder configurations. RNNs process sequences one token at a time, left to right: the hidden state at position $t$ depends on the hidden state at $t-1$. This sequential dependency is the fundamental problem. You cannot parallelize across time steps within a single training example. On long sequences, this becomes a compute bottleneck that no engineering trick fully resolves.

The second problem is the long-range dependency problem. In RNNs, a signal from position 1 must travel through every intermediate hidden state to reach position $n$. That path length is $O(n)$. Gradients degrade over that path. The model structurally forgets early context — or has to work very hard not to, via mechanisms like attention bolted on top.

Attention mechanisms were already being used by 2017, but almost always *in combination with* an RNN: the RNN processes the sequence, attention selects which hidden states to focus on. No one had asked: what if you dropped the RNN entirely and let attention do all the work?

> **Researcher's Move:** Notice the framing: the paper doesn't say "we improved attention mechanisms." It says "we removed the thing everyone assumed was necessary." That's a much bolder claim — and a much more publishable one if it works. Always ask: is the contribution additive (we made X better) or subtractive (we showed X was unnecessary)?

### Think Like a Researcher
- Why had no one removed the RNN before? What assumption was everyone making that this paper challenged?
- The parallelization argument is about *training speed*, not model quality. Why does training speed matter so much at this scale?
- RNNs were well-understood and theoretically grounded. What's the risk of replacing them with a mechanism that was mostly heuristic at this point?

---

## What is the core claim this paper is making?

The central claim is this: **self-attention alone is sufficient for sequence transduction, and removing recurrence yields better performance with less training time.**

This is a strong, falsifiable claim. It's not "we achieved better BLEU scores" — it's a structural argument that the inductive bias of recurrence is unnecessary for sequence modeling, and that global attention over all positions is a better default than sequential state propagation.

The implementation (the Transformer architecture with multi-head attention, positional encodings, and feed-forward sublayers) is the vehicle for testing that claim — but the claim itself is separable from any specific architectural choice.

> **Researcher's Move:** Separate the claim from the implementation. The claim is "recurrence is unnecessary for sequence transduction." The Transformer is one implementation of that hypothesis. If a different attention-only architecture had also worked, the claim would still hold. This distinction matters when evaluating whether the paper's contribution is the *idea* or the *specific design*.

### Think Like a Researcher
- Could this claim be wrong? What would a counterexample look like? (Hint: think about tasks where order matters in a way that attention might not capture.)
- Is "attention is sufficient" the same as "attention is optimal"? The paper conflates these — is that justified?
- If the BLEU improvements were only +0.5, would the architectural argument still hold? What does the magnitude of improvement tell you?

---

## How does the method actually work?

**The intuition:** Every token in the sequence looks at every other token and asks, "how relevant are you to understanding me?" The answer is a weighted sum — you blend together representations of all other tokens, weighted by relevance. Do this in parallel for all tokens simultaneously. Repeat for several layers, each time letting tokens build richer representations by attending over the (increasingly rich) representations from the layer below.

The key insight is that this is just matrix multiplication — and matrix multiplication is what GPUs are built for.

**The architecture** follows a standard encoder-decoder structure. The encoder takes a source sequence and produces a sequence of contextualized representations. The decoder takes those representations and generates the output sequence autoregressively.

Each encoder layer has two sublayers: (1) multi-head self-attention, and (2) a position-wise feed-forward network. Each is wrapped with a residual connection and layer normalization: `LayerNorm(x + Sublayer(x))`.

---

**Key Equation 1: Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q$ = **Query matrix** — what each token is *looking for* (what information do I need?)
- $K$ = **Key matrix** — what each token *advertises* about itself (what information do I have?)
- $V$ = **Value matrix** — the actual information each token contributes when selected
- $d_k$ = **dimension of keys** — used for scaling; prevents dot products from growing so large that softmax saturates into near-zero gradients
- $QK^T$ = **compatibility scores** — how well each query matches each key; an $n \times n$ matrix for a sequence of length $n$
- $\text{softmax}(\cdot)$ = turns scores into a probability distribution over positions
- Final multiplication by $V$ = **weighted blend** of value vectors, weighted by attention probabilities

The $\sqrt{d_k}$ scaling is non-obvious. If $d_k$ is large, dot products grow large in magnitude, pushing softmax into saturation where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products approximately 1, regardless of $d_k$.

---

**Key Equation 2: Multi-Head Attention**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

where:
- $h$ = number of heads (8 in the base model)
- $W_i^Q, W_i^K, W_i^V$ = **learned projection matrices** for each head — each head projects Q, K, V into a lower-dimensional subspace ($d_k = d_v = d_{\text{model}}/h = 64$)
- Each head learns to attend to *different aspects* of the relationships between tokens (e.g., one head might track syntactic dependencies, another coreference)
- $W^O$ = output projection that combines the concatenated heads back to $d_{\text{model}}$

Why multiple heads? A single attention head averages over subspaces. Multi-head attention lets the model jointly attend to information from different representation subspaces at different positions. The cost is the same as single-head attention at full dimensionality, because the reduced per-head dimension compensates for running $h$ heads.

---

**Key Equation 3: Positional Encoding**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where:
- $pos$ = position in the sequence (0, 1, 2, ...)
- $i$ = dimension index within the $d_{\text{model}}$-dimensional embedding
- These are *fixed*, not learned — a deliberate design choice

Without this, the self-attention mechanism is completely order-agnostic: "the cat sat on the mat" and "the mat sat on the cat" would produce identical representations. Positional encodings inject position information by adding a unique vector to each token's embedding. The sinusoidal form means that fixed offsets between positions have predictable relationships — the paper hypothesizes this helps the model generalize to longer sequences than seen in training.

> **Researcher's Move:** For every design choice, ask — why this and not the obvious alternative? For positional encoding, the obvious alternative is learned positional embeddings. The paper tried both and found nearly identical results (Table 3, row E). They chose sinusoidal *speculatively*, because of the extrapolation hypothesis. That hypothesis was never tested in this paper. This is a case where the design choice is weakly motivated.

### Think Like a Researcher
- The $\sqrt{d_k}$ scaling is crucial for training stability. If you removed it, what would happen during training? Could you recover with a different normalization strategy?
- Multi-head attention is more expressive than single-head, but also harder to analyze. What would you need to see to convince yourself that different heads are learning meaningfully different functions?
- The model has no recurrence and no convolution. Where does the "memory" of previous layers' computations live? (Answer: in the residual connections — this is load-bearing.)

---

## What did they test, and what do the results actually prove?

**Tasks:** WMT 2014 English-to-German translation and WMT 2014 English-to-French translation. A smaller experiment on English constituency parsing.

**Datasets:**
- EN-DE: ~4.5M sentence pairs, BPE vocabulary of ~37K tokens
- EN-FR: ~36M sentence pairs, word-piece vocabulary of 32K tokens

**Baselines:** Best published RNN/CNN-based models including ConvS2S, Deep-Att + PosUnk ensembles, ByteNet, and MoE (Mixture of Experts) models.

**Key results:**

| Model | EN-DE BLEU | EN-FR BLEU | Training cost (FLOPs) |
|-------|-----------|-----------|----------------------|
| Best previous single model | 26.4 | 41.1 | — |
| Transformer (base) | 27.3 | 38.1 | $3.3 \times 10^{18}$ |
| Transformer (big) | **28.4** | **41.8** | $2.3 \times 10^{19}$ |
| Best previous ensemble | 26.4 | 41.1 | much higher |

The big Transformer achieves 28.4 BLEU on EN-DE (>2 BLEU above the prior best) and 41.8 on EN-FR (new single-model state of the art), trained in 3.5 days on 8 P100 GPUs — a fraction of the cost of competing models.

**What the results prove:** That attention-only models can match or exceed RNN-based models on the standard machine translation benchmark. Combined with the training cost argument, this validates the efficiency claim.

**What the results do not prove:** That self-attention is *better* than recurrence as a general principle. These results are on translation — a task where global context is very useful and sequence lengths are moderate (sentences). The paper does not test on tasks with very long sequences, tasks requiring strict sequential reasoning, or domains outside of text.

> **Researcher's Move:** Is the baseline a fair comparison? The Transformer (big) is compared to the *best single model* in the prior literature. But the comparison isn't entirely controlled — the Transformer used BPE tokenization, a specific optimizer schedule, and a large batch size. Were those choices available to the baselines? The paper doesn't say. A fully controlled comparison would hold all engineering choices constant except the architectural choice.

### Think Like a Researcher
- BLEU measures n-gram overlap, not semantic accuracy. Could a model achieve high BLEU while being substantively worse at translation? Does this matter for the paper's claims?
- The training cost comparison is compelling, but costs depend heavily on the hardware, implementation, and engineering investment in each model. How much of the efficiency gain is architectural vs. the fact that the Transformer was Google's newest, best-optimized model?
- What ablation would you run to isolate the single most important contribution: multi-head attention, positional encoding, or the removal of recurrence?

---

## What does this paper do well, and what does it leave unresolved?

**Strengths:**

1. **The bet was big and it paid off.** Proposing to remove recurrence entirely — not augment it, not improve it — is a genuinely high-risk architectural decision. That it worked is a meaningful empirical result, not a marginal improvement.

2. **The ablation table (Table 3) is unusually thorough.** The authors vary the number of heads, key/value dimensions, model size, dropout, and positional encoding type, reporting BLEU for each. This gives you real signal about which components matter — rare in ML papers of this era.

3. **The parallelization argument is concrete and honest.** Table 1 compares computational complexity per layer across self-attention, recurrent, and convolutional layers. The analysis is quantitative, not rhetorical.

4. **Transfer to parsing.** Showing that the same architecture works on English constituency parsing with limited data (section 6.3) is an early, credible gesture toward generality.

**Weaknesses and gaps:**

1. **No analysis of failure modes.** The paper reports BLEU; it doesn't show examples where the Transformer fails and an RNN would have succeeded, or vice versa. You learn nothing about *when* to prefer this architecture.

2. **Positional encoding is undertheorized.** The sinusoidal encoding is presented with a plausibility argument, not evidence. The claim that it helps extrapolation to longer sequences is never tested.

3. **Quadratic attention complexity is buried.** Self-attention is $O(n^2 \cdot d)$ in sequence length. For long sequences (documents, audio, genomics), this is a hard wall. The paper mentions it in passing (Table 1) but doesn't engage with it as a limitation.

4. **English-centric evaluation.** Both translation benchmarks are English-{German,French}. The inductive biases of the architecture for morphologically rich languages, low-resource pairs, or right-to-left languages are untested.

**What the results don't show:**

- Whether attention heads are actually interpretable in the way the visualizations imply (section 5 of the appendix). The visualizations are cherry-picked.
- Whether the model is robust to distributional shift, length generalization, or adversarial inputs.
- Whether the gains come from the architecture or the training recipe (optimizer schedule, large batch, BPE).

> **Researcher's Move:** A weakness the authors acknowledge is less concerning than one they don't mention. The quadratic complexity is in Table 1 — they see it. The lack of any discussion of failure modes or out-of-distribution behavior is unacknowledged. That's where the real risk lies.

### Think Like a Researcher
- A skeptical NeurIPS reviewer might say: "The improvements over the baseline are real but the comparison is not controlled. How do we know the gains are architectural and not a result of superior engineering?" How would you design an experiment to answer this?
- What paper would most directly challenge this work? (Hint: think about tasks where order matters strictly, or sequence lengths in the thousands.)
- The Transformer won because it was fast to train. If training compute were free, would the architectural argument still hold?

---

## Where does this paper live in the field?

**What it builds on:**
- **Bahdanau et al. (2015)** — introduced additive attention for machine translation; the Transformer's scaled dot-product attention is a faster, cleaner variant of this
- **Gehring et al. (2017), ConvS2S** — showed that CNNs could replace RNNs for translation with better parallelization; the Transformer takes this further by removing convolution too
- **Layer normalization and residual connections** — borrowed directly from image models (He et al., 2016); the Transformer would not train without these

**What it competes with or obsoletes:**
- RNN-based encoder-decoders (seq2seq with attention) — replaced for most NLP tasks within 2 years
- ConvS2S and ByteNet — made largely obsolete in translation
- GRU/LSTM as the default sequence model — replaced by Transformer in virtually every NLP task by 2019

**What it enables:**
- **BERT (2018)** — applied Transformer encoder pretraining at scale; became the dominant transfer learning paradigm for NLP
- **GPT series (2018–present)** — applied Transformer decoder at scale; the architecture behind ChatGPT
- **Vision Transformer (ViT, 2020)** — showed the architecture transfers to image patches
- **Protein structure prediction (AlphaFold2, 2021)** — adapted multi-head attention for biological sequences
- Essentially every large language model in use today is a Transformer variant

> **Researcher's Move:** A paper's significance is often clearer 2 years after publication than at the time. This paper's BLEU numbers are now irrelevant — the architectures it replaced are gone. What matters is that it created the substrate for GPT and BERT, which redefined what NLP was. That wasn't obvious in 2017. When reading a recent paper, ask: does this create a substrate, or does it optimize an existing one?

### Think Like a Researcher
- What papers should you read before this one? (Bahdanau 2015 for attention, Hochreiter & Schmidhuber 1997 for LSTM, Gehring 2017 for ConvS2S)
- BERT and GPT use only the encoder or decoder stack, not the full encoder-decoder. What does that tell you about which parts of this paper's contribution were actually used?
- If you had to place this paper in a 10-paper "history of deep learning" reading list, where does it go, and what comes before and after it?

---

## What questions does this paper leave open?

1. **How does the Transformer handle very long sequences?** The $O(n^2)$ attention complexity is a hard limit. Sparse attention, local attention, linear attention — this became a major research direction (Longformer, Big Bird, Performer). *Difficulty: hard; requires fundamental changes to the attention mechanism.*

2. **Are the positional encodings the right inductive bias?** Sinusoidal encodings assume absolute position matters. Relative position encodings (Shaw et al., 2018; Rotary PE) turned out to generalize better. *Difficulty: medium; tractable as an empirical study.*

3. **What are individual attention heads actually doing?** The paper shows visualizations but doesn't provide a systematic analysis. Clark et al. (2019) and others later probed this — heads do appear to specialize, but the picture is messy. *Difficulty: medium; requires interpretability methodology.*

4. **Does this architecture scale without fundamental changes?** The paper used 6 layers. GPT-3 used 96. What breaks at scale, and what needs to be added (e.g., better initialization, different normalization)? *Difficulty: very hard; required years of empirical engineering.*

5. **Is self-attention the right operation, or just one that works?** The paper never provides a theoretical justification for why attention should work for sequence modeling. Is there a formal framework that explains this? *Difficulty: hard; still largely open.*

### Think Like a Researcher
- The quadratic complexity problem (question 1) spawned dozens of papers. If you wanted to write a PhD thesis in this space, would you try to fix the complexity, or would you argue the quadratic cost is acceptable if compute keeps scaling?
- Question 3 (what are attention heads doing?) is tractable but the answers have been disappointing — heads don't neatly specialize. What does that tell you about whether "interpretability" is the right frame for understanding these models?
- Which of these open questions, if answered negatively (i.e., the answer goes against the paper's implicit assumptions), would most damage the Transformer's status as a foundational architecture?

---

## If you were to re-implement this, what would you need to know?

**What the paper gives you clearly:**
- The mathematical definition of scaled dot-product attention and multi-head attention
- The encoder and decoder stack structure (N=6 layers each)
- Key hyperparameters: $d_{\text{model}}=512$, $d_{ff}=2048$, $h=8$, $d_k=d_v=64$, dropout=0.1
- The learning rate schedule (equation 3, warmup steps=4000)
- Training data and tokenization (BPE, shared vocabulary)

**What the paper leaves underspecified:**

- **Weight initialization.** The paper doesn't specify how weights are initialized. This matters enormously for training stability — a bad initialization scheme and the model doesn't train at all. The authors' codebase (tensor2tensor) used Xavier initialization, but this isn't in the paper.

- **The exact masking in decoder self-attention.** The paper says illegal positions are masked to $-\infty$ before softmax. The implementation details — how this interacts with padding masks during batched training — are non-trivial and not described.

- **Label smoothing.** Mentioned once ($\epsilon_{ls} = 0.1$) with no explanation of why or how it interacts with the loss function. It turns out to be important for BLEU scores.

- **Beam search parameters.** Translation results use beam search with beam size 4 and length penalty $\alpha = 0.6$ (reported in appendix). The sensitivity of BLEU to these choices is not analyzed.

- **The "big" model.** Table 3 describes the big model with $d_{\text{model}}=1024$, $h=16$, $d_{ff}=4096$ and dropout 0.3, trained for 300K steps. But the training instabilities that arise at this scale are not discussed.

> **Researcher's Move:** Underspecified details are often where the real contribution hides — or where the method is more fragile than it looks. The learning rate warmup schedule (equation 3) is unusual and critical. If you use a flat learning rate, the model trains poorly. The paper presents this as a minor detail, but practitioners who tried to reproduce the results quickly found it was load-bearing.

### Think Like a Researcher
- The warmup learning rate schedule (increase linearly, then decay by inverse square root) is unusual. Why does a Transformer need warmup when RNNs don't? What is happening in the early training steps that makes this necessary?
- If you were training on a different language pair (e.g., English-Chinese, with a very different vocabulary and morphology), which hyperparameters would you change first, and why?
- The paper shares embedding weights between the two embedding layers and the pre-softmax linear transformation. This is a non-obvious decision that reduces parameters significantly. What would you check in the released code to understand how this is implemented?

---

## Key Terms to Own

| Term | Intuition | Formal definition |
|------|-----------|-------------------|
| **Self-attention** | Every token looks at every other token and decides how much to borrow from each | An attention mechanism where queries, keys, and values all come from the same sequence |
| **Query / Key / Value** | A query is what you're searching for; a key is what others advertise; a value is what you get when you find a match | Learned linear projections of input representations used to compute attention weights (Q, K) and attended output (V) |
| **Multi-head attention** | Instead of one "vote" about which tokens are relevant, run 8 parallel votes in different representational subspaces | $h$ parallel attention functions applied to different linear projections of Q, K, V, with outputs concatenated and projected |
| **Scaled dot-product** | The relevance score between two tokens, scaled down to prevent numerical blow-up | $\text{score}(q, k) = q \cdot k / \sqrt{d_k}$ |
| **Positional encoding** | Since attention is order-agnostic, you inject position information by adding a position-dependent vector to each token | Fixed (or learned) vectors added to input embeddings to give the model information about token positions in the sequence |
| **Encoder-decoder attention** | The decoder "reads" the encoder's output — each decoder position can look at all encoder positions | Cross-attention where queries come from the decoder and keys/values come from the encoder output |
| **Autoregressive decoding** | At inference time, the decoder generates one token at a time, feeding each new token back as input for the next step | A generation process where $p(y_1, \ldots, y_m) = \prod_t p(y_t \mid y_{<t}, z)$, decoded left to right |
| **BLEU score** | A rough measure of how much a machine translation overlaps with human reference translations (higher is better, max 100) | Geometric mean of n-gram precision (n=1..4) between hypothesis and reference, with a brevity penalty |
| **Layer normalization** | Normalize activations across the feature dimension (not the batch) — stabilizes training in deep networks | $\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$ where $\mu, \sigma$ are computed over the feature dimension |
| **Residual connection** | The output of a sublayer is added back to its input — lets gradients flow directly to earlier layers | $\text{output} = x + \text{Sublayer}(x)$; enables training of very deep networks by providing gradient highways |
| **BPE (Byte-Pair Encoding)** | A tokenization scheme that splits rare words into subword pieces — "unaffordable" → "un", "afford", "able" | A data compression algorithm adapted for NLP: iteratively merge the most frequent character pair until vocabulary size is reached |
