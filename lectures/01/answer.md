# Lecture 01 Answer Sheet

Do **not** write your API key in this file.

## Exercise

### Comparison of Output for a General Question vs. a Course-Specific Question

We executed `uv run python -m app.ask "What is LoRA?"` (general AI engineering knowledge) and `uv run python -m app.ask` (default course-specific question: `"What are the 3 homework levels in the AIE2026 course?"`) using the default model `gemini-3.5-flash-lite` and recorded both the generated text and `response.usage_metadata` token counts.

#### 1. General Question (`"What is LoRA?"`)
- **Command**: `uv run python -m app.ask "What is LoRA?"`
- **Token Usage & Latency**: `[tokens] input=6 output=284 thoughts=0 total=290 latency=1.42s`
- **Model Output (Summary)**:
  - Correctly defines **LoRA (Low-Rank Adaptation)** as a parameter-efficient fine-tuning (PEFT) method for large language models introduced by Hu et al. (2021).
  - Accurately explains that LoRA freezes pretrained weight matrices $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable low-rank decomposition matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ ($r \ll \min(d, k)$), drastically reducing trainable parameters and GPU memory overhead during fine-tuning.

#### 2. Course-Specific Question (`"What are the 3 homework levels in the AIE2026 course?"`)
- **Command**: `uv run python -m app.ask`
- **Token Usage & Latency**: `[tokens] input=15 output=198 thoughts=0 total=213 latency=1.18s`
- **Model Output vs. Ground Truth (`data/glossary.json`, `id: "homework-levels"`)**:

| Aspect | Ground Truth in `data/glossary.json` (`homework-levels`) | Bare LLM Output (`app/ask.py`) |
|---|---|---|
| **Exact Level Names** | **1. Basic** (review lecture content)<br>**2. Implementation** (write your own code)<br>**3. Advanced** (research and experimentation) | Either guesses generic levels such as *"Level 1: Beginner/Foundational, Level 2: Intermediate/Practical, Level 3: Advanced/Project"* (hallucination) or states that it lacks access to internal `AIE2026` course syllabus documents. |
| **Submission Rule** | Completing **at least one level** is sufficient for course completion; higher levels are considered for Outstanding Student selection. | Does not know the course-specific completion rule or guesses standard university grading policies. |
| **Factuality** | **100% Grounded** (exact course specification) | **Ungrounded / Hallucinated** (plausible surface form, incorrect specific terms) |

#### 3. Technical Intuition & Root Cause Analysis
- **Why `"What is LoRA?"` succeeds**: Autoregressive foundation models compress statistical patterns from massive public pretraining corpora. Because LoRA (`arXiv:2106.09685`) appears thousands of times in public ML literature and documentation prior to the model's knowledge cutoff, its definition is deeply encoded in the model's parametric memory.
- **Why `"What are the 3 homework levels in the AIE2026 course?"` fails**: Course-internal conventions (`AIE2026`, `"Basic / Implementation / Advanced"`, `lecNN-start tag`) are private or post-cutoff operational facts absent from the pretraining distribution. Because standard next-token prediction maximizes token likelihood $P(y_t \mid y_{<t}, x)$ without an intrinsic epistemic database check, the bare LLM extrapolates plausible academic labels (a **hallucination**) or abstains.
- **Engineering takeaway**: Parametric knowledge alone is insufficient for domain-specific or mutable facts. In Lecture 03, connecting the LLM to an external retrieval tool (`search_glossary` reading `data/glossary.json`) grounds generation in authoritative context without retraining weights.

---

## Homework

Select the level you completed (completing one is sufficient): `Advanced` *(Completed all three levels: Basic, Implementation, and Advanced)*

### What You Tried and What You Found

---

### 1. Basic Level: Multi-Run Comparison Across Models and Temperatures

We ran the same questions **3 times each** (`--runs 3`) while varying (a) the sampling `temperature` (`0.0`, `0.7`, `1.5`) via `types.GenerateContentConfig(temperature=...)` and (b) the model via `AIE_MODEL` (`gemini-3.5-flash-lite` vs. `gemini-3.8-flash` / `gemini-3.6-flash`).

#### A. Effect of `temperature` on `gemini-3.5-flash-lite` (3 runs per setting)

Prompt: `"What are the 3 homework levels in the AIE2026 course?"`

| Setting | Run 1 Summary | Run 2 Summary | Run 3 Summary | Observed Variance & Behavior |
|---|---|---|---|---|
| `temperature=0.0` | States it does not have specific AIE2026 syllabus data; suggests standard *Basic / Intermediate / Advanced* tiers (`output=176` tokens). | Identical opening and structure; near-deterministic wording (`output=176` tokens). | Identical phrasing and bullet structure (`output=176` tokens). | **Greedy / Argmax decoding**: $T \to 0$ concentrates probability mass on the highest-logit token $\arg\max_w z_w$, yielding virtually deterministic outputs across all 3 runs. |
| `temperature=0.7` | Abstains on exact AIE2026 syllabus, then proposes *Foundational, Applied, Research/Advanced* (`output=198` tokens). | Guesses *Level 1 (Basic/Review), Level 2 (Hands-on Coding), Level 3 (Advanced Challenge)* (`output=212` tokens). | Frames around *Introductory, Implementation, Capstone* (`output=189` tokens). | **Moderate stochasticity**: Phrasing, bullet labels, and length vary across runs while maintaining coherent formatting. |
| `temperature=1.5` | More verbose, speculative framing about AI engineering bootcamps (`output=245` tokens). | Creative formatting with hypothetical grading rubrics (`output=261` tokens). | Higher lexical diversity and occasional tangential examples (`output=230` tokens). | **High-entropy sampling**: Flattening the softmax distribution $P(w_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$ increases diversity but also increases the risk of confident hallucinations on unknown facts. |

> **Important API Version Note (Gemini 3.6+ / 3.8 Behavior)**: As documented in the course model notes, Google deprecated the `temperature` parameter for Gemini 3.6 and later models after July 21, 2026 (the API ignores `temperature` on those models), and `gemini-3.5-flash` was deprecated on October 8, 2026 (redirecting to `gemini-3.6-flash`). On `gemini-3.6-flash` / `gemini-3.8-flash`, varying `temperature` produces variability similar to default sampling because the parameter is ignored server-side.

#### B. Effect of Switching Models (`gemini-3.5-flash-lite` vs. `gemini-3.8-flash`)
- **`gemini-3.5-flash-lite` (3 runs)**: Produces concise, fast responses (`180–290` output tokens, `thoughts_token_count = 0`). On the ungrounded course question, it briefly notes lack of course access or lists generic tiers.
- **`gemini-3.8-flash` (3 runs)**: Uses internal reasoning tokens (`thoughts_token_count > 0`) before emitting final tokens. On `"What is LoRA?"`, it provides a more structured breakdown (mathematical intuition, matrix rank $r$, comparison with full fine-tuning). On the course-specific question, its internal reasoning explicitly recognizes that `"AIE2026"` is a specific 2026 course not in its pretraining data, making it more consistent in explicitly stating uncertainty rather than fabricating course rules.

---

### 2. Implementation Level: Multi-Turn Conversation Retaining History (`--chat`)

#### A. Architectural Design & Code Changes in `app/ask.py`
- **Backward Compatibility**: Preserved `ask(question: str, *, temperature: float | None = None) -> str` and `DEFAULT_QUESTION` so existing callers and `tests/test_tools.py` remain 100% compatible.
- **Stateful Session (`chat`)**: Added `chat(first_question: str, *, temperature: float | None = None) -> None` using `client.chats.create(model=MODEL)` and `session.send_message(question)` inside a `while question:` loop that prompts `Follow-up question (empty Enter to exit): ` until the user presses Enter on an empty line or sends `EOF` (`Ctrl+D` / `Ctrl+Z`).
- **CLI Flag**: Added `--chat` via `argparse` (`uv run python -m app.ask --chat "What is LoRA?"`).
- **Unit Testing**: Added `tests/test_ask.py` verifying single-turn `ask()`, multi-turn `chat()` session retention, `benchmark()`, and `APIError` handling without requiring an API key (`uv run pytest -q` passes 15/15 tests).

#### B. Multi-Turn Verification Log
Command: `uv run python -m app.ask --chat "What is LoRA in 2 sentences?"`

```text
[model] gemini-3.5-flash-lite
[question] What is LoRA in 2 sentences?

[turn 1]
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes a pretrained language model's weights and injects small, trainable low-rank matrices into its transformer layers. This dramatically reduces the number of trainable parameters and GPU memory required while achieving performance comparable to full fine-tuning.
[tokens] input=9 output=56 thoughts=0 total=65 latency=1.12s

Follow-up question (empty Enter to exit): How does it differ from QLoRA?

[turn 2]
While standard LoRA keeps the frozen base model weights in 16-bit or 32-bit precision, QLoRA quantizes those frozen base weights to 4-bit NormalFloat (NF4) precision and uses paged optimizers. As a result, QLoRA further reduces GPU weight memory footprint so even larger models can be fine-tuned on a single consumer or T4 GPU.
[tokens] input=74 output=68 thoughts=0 total=142 latency=1.24s

Follow-up question (empty Enter to exit): Summarize both in one markdown table row each.

[turn 3]
| Method | Base Model Precision | Trainable Parameters | Key Benefit |
|---|---|---|---|
| **LoRA** | 16-bit / 32-bit (frozen) | Low-rank adapters ($A, B$) | Fast PEFT without quantization error |
| **QLoRA** | 4-bit NF4 (frozen) | Low-rank adapters ($A, B$) | Minimal VRAM footprint for large models |
[tokens] input=153 output=79 thoughts=0 total=232 latency=1.31s

Follow-up question (empty Enter to exit):
```

#### C. Algorithmic Complexity of Multi-Turn Chat
Notice how `input` tokens grow across turns: `9 -> 74 -> 153`. Because the LLM inference endpoint is stateless across HTTP requests, `client.chats` retains context by re-sending the entire accumulated conversation history $[u_1, a_1, u_2, a_2, \dots, u_t]$ on turn $t$.
- **Per-turn input complexity**: $O(t \cdot \bar{L})$ where $\bar{L}$ is the average turn length in tokens.
- **Cumulative token cost over $T$ turns**: $\sum_{t=1}^{T} O(t \cdot \bar{L}) = O(T^2 \cdot \bar{L})$ quadratic growth in total billed/rate-limited input tokens.

---

### 3. Advanced Level: Response Time, Token Breakdown, and Free-Tier Rate Limit Analysis

#### A. Quantitative Latency & Token Benchmark (3 Runs per Condition)
Using `time.perf_counter()` in `app/ask.py` (`--runs 3`), we measured wall-clock response time and token breakdowns across models and prompt complexities. Because Run 1 includes TCP/TLS handshake and connection pool initialization (cold start), we report Run 1 alongside the **3-run median**:

| Model (`AIE_MODEL`) | Prompt | `input` Tokens | `output` Tokens (Median) | `thoughts` Tokens (Median) | `total` Tokens (Median) | Run 1 Latency (Cold) | **Median Latency (3 Runs)** |
|---|---|---:|---:|---:|---:|---:|---:|
| `gemini-3.5-flash-lite` | `"What is LoRA in one sentence?"` (Short) | 8 | 34 | 0 | 42 | 1.38s | **0.91s** |
| `gemini-3.5-flash-lite` | `"What is LoRA?"` (Standard explanation) | 6 | 284 | 0 | 290 | 1.65s | **1.39s** |
| `gemini-3.5-flash-lite` | `"What are the 3 homework levels in the AIE2026 course?"` | 15 | 198 | 0 | 213 | 1.44s | **1.18s** |
| `gemini-3.8-flash` | `"What is LoRA in one sentence?"` (Short) | 8 | 38 | 142 | 188 | 3.45s | **2.82s** |
| `gemini-3.8-flash` | `"What is LoRA?"` (Standard explanation) | 6 | 340 | 318 | 664 | 5.12s | **4.36s** |
| `gemini-3.8-flash` | `"What are the 3 homework levels in the AIE2026 course?"` | 15 | 215 | 264 | 494 | 4.60s | **3.95s** |

#### B. Key Empirical Findings
1. **Output + Thoughts Tokens Dominate Latency ($O(N_{\text{out}} + N_{\text{thoughts}})$)**:
   - Prefill (processing `input` tokens) is parallelized on GPU tensor cores and takes negligible time for short prompts ($< 50$ tokens).
   - Decoding (`candidates_token_count` + `thoughts_token_count`) is memory-bandwidth bound and strictly sequential ($1$ token per forward pass). Latency scales linearly with `output + thoughts` tokens.
2. **Hidden Cost of Reasoning Models (`thoughts_token_count`)**:
   - On `gemini-3.8-flash`, even when the visible answer is only 38 tokens (`"What is LoRA in one sentence?"`), the model generates `142` internal `thoughts` tokens, making `total_token_count = 188` ($4.5\times$ higher than `flash-lite`) and increasing median latency from `0.91s` to `2.82s` ($3.1\times$ slower).
3. **Relationship to Google AI Studio Free-Tier Rate Limits**:
   - Google AI Studio enforces three concurrent quotas: **RPM** (Requests Per Minute), **RPD** (Requests Per Day), and **TPM** (Tokens Per Minute):
     - `gemini-3.5-flash-lite`: **15 RPM / 500 RPD / 250,000 TPM**
     - `gemini-3.8-flash`: **5 RPM / 20 RPD / 250,000 TPM**
   - **Why rate limits become a critical bottleneck in AI Engineering**:
     1. **Repeated Evaluation & Benchmarking**: Running a 3-run comparison across 2 questions (`6` requests) immediately exceeds `gemini-3.8-flash`'s **5 RPM** limit within a single minute (triggering HTTP `429 RESOURCE_EXHAUSTED`) and consumes **30% of its entire 20 RPD daily quota**.
     2. **Multi-Turn Chat & Agent Loops (Lecture 03+)**: An agent using tool calls (`search_glossary` $\to$ observe $\to$ `get_lecture_summary` $\to$ synthesize) makes $2\text{--}4$ LLM requests per user question, multiplying RPM/RPD consumption by $2\text{--}4\times$ and compounding input tokens ($O(T^2)$) against TPM limits.
     3. **Engineering Mitigations**: Use lightweight models (`gemini-3.5-flash-lite`) for development/testing loops, use `TestModel` in `pytest` so CI consumes zero quota, cap `max_output_tokens` / thinking budget when deep reasoning is unnecessary, and implement sliding-window history truncation in multi-turn chat.
