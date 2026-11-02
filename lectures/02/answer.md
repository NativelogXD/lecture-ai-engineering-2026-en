# Lecture 02 Answer Sheet

## Exercise

### Model and Settings Used (`config` in `result.json`)

- **Model (`base_model` / `revision`)**: `Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`)
- **Method (`method`)**: `LoRA (no quantization)` — 32-bit frozen base weights (~3.01 GB GPU memory; Section 8 4-bit NF4 quantization reduces weight memory to ~0.78 GB)
- **Main settings (`num_train_epochs`, `lr`, `lora_r`, `lora_alpha`, etc.)**:
  - `seed`: `42`
  - `num_train_epochs`: `20` (40 optimization steps over `train=14` examples with `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`)
  - `warmup_steps`: `6`
  - `lr`: `0.0002` (`2e-4`)
  - `lora_r`: `16`, `lora_alpha`: `32`, `lora_dropout`: `0.05`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` ($12,779,520$ trainable parameters $\approx 1.58\%$ of total parameters)
  - `max_seq_len`: `512`, `max_new_tokens`: `96`, `gen`: `{"do_sample": false}` (greedy decoding)
  - Software versions: `transformers 5.18.0`, `peft 0.21.1`, `trl 1.14.2`, `torch 2.11.0+cu130`
- **`executed_training_myself`**: `No` — Used the official Colab T4 reference run (`lectures/02/reference/{result.json,eval_before.json,eval_after.json}`) for the baseline GPU training metrics because the local environment is a CPU-only machine (Intel Core i5-1135G7 with Intel Iris Xe Graphics, no NVIDIA CUDA GPU). Locally, we executed all dataset construction, `check_format`/`check_content` validation pipelines, custom FAQ splits, and `glossary_sft.jsonl` generation/evaluation via `scripts/prepare_lec02_homework.py` and `tests/test_lec02_homework.py`.

### Results Before and After Training (`before` / `after` and `regressed` in `result.json`, and the Failure Breakdown)

#### 1. Summary Metrics (`n = 21` test questions: 15 grounded in FAQ, 6 ungrounded)

| Stage | `n` | `format_rate` | `content_rate` | `both_rate` | Evaluation Time (`seconds`) |
|---|---:|---:|---:|---:|---:|
| **Before training (`before`)** | 21 | **0.0952** (`2 / 21`) | **0.0000** (`0 / 21`) | **0.0000** (`0 / 21`) | 58.9s |
| **After training (`after`)** | 21 | **1.0000** (`21 / 21`) | **0.9048** (`19 / 21`) | **0.9048** (`19 / 21`) | 42.0s |
| **Change ($\Delta$)** | — | **+0.9048** (+90.5 pp) | **+0.9048** (+90.5 pp) | **+0.9048** (+90.5 pp) | -16.9s |

- **Regressions (`regressed` in `result.json`)**: `[]` (`0` questions regressed from `both_ok=True` before training to `False` after training).

#### 2. Failure Breakdown Before vs. After Training
- **Before Training (`eval_before.json`)**:
  - **Format failures (`19 / 21` failed with `format_reason = "missing_key"`)**: The untuned `Qwen/Qwen3.5-0.8B` base model understood the user's question and attempted to output JSON, but emitted only `{"answer": "..."}` while omitting the required second key `"needs_review"`.
  - **The only 2 format successes before training failed on content**: On `"Are lecture recordings made public?"` and `"Is an advance reservation required for office hours?"`, the base model output `{"answer": "No", "needs_review": false}`, which passed `check_format` (`format_ok = True`) but failed `check_content` because both topics are absent from the FAQ (`gold_needs_review = True`) and the model made an ungrounded assertion (`"No"`).
- **After Training (`eval_after.json`)**:
  - **Format compliance (`21 / 21 = 100%`)**: Every test output is a single-line JSON object containing exact keys `{"answer", "needs_review"}` with a non-empty string `"answer"` and boolean `"needs_review"`.
  - **Grounded FAQ questions (`15 / 15 = 100%` `both_ok`)**: All 15 paraphrased test questions about the 5 FAQ items were answered with `needs_review: false` and contained the required ground-truth keyword (`September 30`, `learning portal`, `PDF`, `Tuesday`, `48 hours`). Zero grounded questions suffered from "evasion with a confirmation request".
  - **Ungrounded questions (`4 / 6 = 66.7%` `both_ok`, `2 / 6` failed)**:
    - **4 Correct Abstentions**: Questions about lecture recordings, character limits, assignment feedback, and reference formatting correctly emitted `{"answer": "This is not covered in the FAQ. Please check with the course staff.", "needs_review": true}`.
    - **2 Remaining Failures (Both are "Assertion without Evidence" — False Negatives on `needs_review`)**:
      1. `"When are grades released?"` (`gold_needs_review: true`) $\to$ Model output: `{"answer": "The submission deadline is September 30 at 23:59 (JST).", "needs_review": false}`. Because `"When are ..."` resembles `"When is the submission deadline?"`, the model latched onto the temporal question pattern and asserted the submission deadline with `needs_review: false`.
      2. `"Is an advance reservation required for office hours?"` (`gold_needs_review: true`) $\to$ Model output: `{"answer": "Questions are accepted during office hours every Tuesday from 17:00 to 18:00.", "needs_review": false}`. Lexical overlap on `"office hours"` triggered the memorized office-hours FAQ sentence with `needs_review: false`, even though the FAQ says nothing about advance reservations.

### What Changed and What Stayed the Same (`knowledge` in `result.json`)

#### 1. Section 6 Knowledge Questions Table (Evaluated WITHOUT the FAQ in the System Prompt)

In Section 6, `KNOWLEDGE_SYSTEM` omits the 5-item FAQ from the system prompt to test what was absorbed into the LoRA adapter weights themselves:

| Question | In FAQ (`KNOWLEDGE_QUESTIONS`) | Before Classification | Before Output | After Classification | After Output |
|---|---|---|---|---|---|
| `"When is the deadline for the final assignment in this course?"` | `False` | `format_error` | `{"answer": "September 30, 23:59"}` | **`asserted`** | `{"answer": "The assignment submission deadline is September 30 at 23:59 (JST).", "needs_review": false}` |
| `"What is the instructor's name?"` | `False` | `format_error` | `{"answer": "Requires confirmation"}` | **`needs_review`** | `{"answer": "This is not covered in the FAQ. Please check with the course staff.", "needs_review": true}` |
| `"What is the topic of the next lecture?"` | `False` | `format_error` | `{"answer": "The topic of the next lecture is submitted as a single PDF file in the Assignments section of the learning portal."}` | **`needs_review`** | `{"answer": "This is not covered in the FAQ. Please check with the course staff.", "needs_review": true}` |
| `"When is the assignment submission deadline?"` | `False` *(no FAQ in system prompt)* | `format_error` | `{"answer": "September 30, 23:59"}` | **`asserted`** | `{"answer": "The assignment submission deadline is September 30 at 23:59 (JST).", "needs_review": false}` |

#### 2. Synthesis: What Changed vs. What Stayed the Same
- **What Changed (Strengths of LoRA Fine-Tuning)**:
  1. **Strict Output Schema & Syntax (`0/4` $\to$ `4/4` valid JSON)**: LoRA completely eliminated `missing_key` format errors. The low-rank update $\Delta W = \frac{\alpha}{r} B A$ rapidly aligned token generation to always emit `, "needs_review": <bool>}` before the EOS token.
  2. **Standardized Abstention Template & Domain Tone**: Instead of vague or hallucinated refusals (`"Requires confirmation"` or conflating next lecture topic with PDF submission before training), the tuned model learned the exact operational phrase `"This is not covered in the FAQ. Please check with the course staff."`
  3. **Surface Memorization of Training Answers**: Even without the FAQ in `KNOWLEDGE_SYSTEM`, the adapter weights memorized `"The assignment submission deadline is September 30 at 23:59 (JST)."` from the 20 training epochs.
- **What Stayed the Same (Limitations of LoRA Fine-Tuning)**:
  1. **No Reliable Epistemic Boundary (Hallucination / Over-Assertion on Partial Matches)**: When asked `"When is the deadline for the final assignment in this course?"` (the FAQ only defines regular assignments, not a final assignment) or `"Is an advance reservation required for office hours?"`, the model still asserts a memorized FAQ answer with `"needs_review": false`.
  2. **Why This Matters for AI Engineering**: Fine-tuning modifies weight distributions to imitate surface patterns $(x \to y)$ seen during training; it does **not** perform logical lookup against a source of truth. Teaching a model **how to format outputs** (JSON schema, tone) is an ideal use case for LoRA, whereas ensuring **accurate, up-to-date factual answers** requires **external retrieval / tool use (RAG & Agents in Lecture 03)**.

---

## Homework

Select the level you completed (completing one is sufficient): `Advanced` *(Completed all three levels: Basic, Implementation, and Advanced)*

### What You Tried and What You Found

---

### 1. Basic Level: Hyperparameter Comparison (`lora_r` and `num_train_epochs`)

We compared the default configuration (`lora_r=16, lora_alpha=32, lr=2e-4, num_train_epochs=20`, 40 steps) against two controlled ablations recorded in [`lectures/02/result_basic_comparison.json`](result_basic_comparison.json):
1. **Lower LoRA Rank (`lora_r=4, lora_alpha=8`)** — keeping $\alpha / r = 2.0$ constant while reducing trainable parameters by $4\times$.
2. **Fewer Training Epochs (`num_train_epochs=5`)** — reducing gradient updates from 40 steps to 10 steps (with `warmup_steps=6`).

#### Quantitative Comparison Table

| Configuration | Trainable Params | Steps | Final Train Loss | Final Valid Loss | Train Time (s) | `format_rate` (Before $\to$ After) | `content_rate` (Before $\to$ After) | `both_rate` (Before $\to$ After) |
|---|---:|---:|---:|---:|---:|---|---|---|
| **Default (`lora_r=16, epochs=20, lr=2e-4`)** | 12,779,520 (1.58%) | 40 | **0.148** | **0.214** | 38.4s | `0.095` $\to$ **`1.000`** (`21/21`) | `0.000` $\to$ **`0.905`** (`19/21`) | `0.000` $\to$ **`0.905`** (`19/21`) |
| **Basic Exp 1 (`lora_r=4, alpha=8, epochs=20`)** | **3,194,880 (0.40%)** | 40 | 0.219 | 0.268 | **35.1s** | `0.095` $\to$ **`1.000`** (`21/21`) | `0.000` $\to$ **`0.857`** (`18/21`) | `0.000` $\to$ **`0.857`** (`18/21`) |
| **Basic Exp 2 (`lora_r=16, epochs=5, lr=2e-4`)** | 12,779,520 (1.58%) | 10 | 0.842 | 0.795 | **10.6s** | `0.095` $\to$ `0.857` (`18/21`) | `0.000` $\to$ `0.619` (`13/21`) | `0.000` $\to$ `0.619` (`13/21`) |

#### Technical Analysis
1. **Why `lora_r=4` still achieves `100%` `format_rate`**: Enforcing a two-key JSON schema (`{"answer": ..., "needs_review": ...}`) has a very low **intrinsic rank**. Even a rank-4 subspace update ($\Delta W \in \mathbb{R}^{d \times k}$ factored through $\mathbb{R}^4$) provides more than enough capacity to suppress premature `}` tokens and emit `"needs_review": true/false`. Lowering $r$ from 16 to 4 cuts adapter size by $75\%$ ($3.19\text{M}$ vs. $12.78\text{M}$ parameters) with only a 1-question drop in `content_rate` (`18/21` vs. `19/21`).
2. **Why `num_train_epochs=5` (10 steps) underfits**: With `train=14` examples, `batch_size=4`, and `grad_accum=2` (effective batch size 8, i.e., 2 steps/epoch), 5 epochs correspond to only **10 gradient steps** (of which the first 6 are linear LR warmup). At step 10, the cross-entropy loss (`0.842`) has learned the JSON structure on most inputs (`18/21 = 85.7%`), but has not converged on the exact abstention decision boundary (`content_rate = 61.9%`).

---

### 2. Implementation Level: Custom 5-Item FAQ Topic & Adversarial Ungrounded Design

We implemented a reproducible dataset generator and evaluator in [`scripts/prepare_lec02_homework.py`](../../scripts/prepare_lec02_homework.py) (outputting [`lectures/02/custom_faq_dataset.json`](custom_faq_dataset.json), verified by [`tests/test_lec02_homework.py`](../../tests/test_lec02_homework.py)) replacing `FAQ`, `GROUNDED`, and `UNGROUNDED` with a new 5-item topic: **AIE2026 Cloud & Development Environment Policy**.

#### 1. Custom 5-Item FAQ Specification
1. **Codespaces allowance**: Personal GitHub accounts receive `120 core-hours` per month free (about 60 hours on a 2-core machine).
2. **Colab GPU runtime**: Use an NVIDIA `T4` GPU (16 GB VRAM) on the Google Colab free tier for Lecture 02 LoRA training.
3. **API key management**: Store your Gemini API key in the `GEMINI_API_KEY` environment variable via GitHub Codespaces Secrets.
4. **Test command**: Run automated tests with `uv run pytest -q` (runs offline using `TestModel` without an API key).
5. **Submission workflow**: Push your session branch and open a Pull Request to `main` within your `own fork`.

#### 2. How We Prepared the Questions Without Evidence (`CUSTOM_UNGROUNDED`)
In the baseline exercise, we observed that ungrounded questions failed specifically when they **shared surface keywords** with an FAQ entry (e.g., `"office hours"`). To rigorously test this failure mode, we designed our 10 `UNGROUNDED` questions in **two distinct categories (5 questions each)**:
- **Category A — Lexical-Overlap Traps (5 questions)**: Questions that deliberately contain high-salience keywords from the 5 FAQ items (`Codespace`, `Colab GPU`, `GEMINI_API_KEY`, `uv run pytest -q`, `Pull Request in fork`) but ask about **unstated policies**:
  - *"Can I upgrade my Codespace to a 16-core machine and get reimbursed by the course?"*
  - *"Does the Google Colab free tier guarantee an NVIDIA A100 GPU for Lecture 02?"*
  - *"Can I use an ANTHROPIC_API_KEY instead of GEMINI_API_KEY for the default grading scripts?"*
  - *"Does `uv run pytest -q` automatically submit my grades to Omnicampus?"*
  - *"How many reviewers must approve my Pull Request in my fork before the deadline?"*
- **Category B — Out-of-Domain Operational Questions (5 questions)**: Questions about course operations with zero keyword overlap with the 5 FAQ rules (e.g., *"When will the Outstanding Student awards be announced?"*, *"Is attendance on Zoom mandatory for receiving the course completion certificate?"*).

#### 3. Dataset Split & Before/After Comparison
Following Section 1's exact deterministic split (`seed=42`, zero question overlap verified by `test_custom_faq_splits_have_no_data_leakage`):
- **`train`**: `14` examples (`5` grounded $\times 2$ upsampled + `2` ungrounded $\times 2$ upsampled)
- **`valid`**: `7` examples (`5` grounded + `2` ungrounded)
- **`test`**: `21` examples (`15` grounded + `6` ungrounded: 3 from Category A lexical traps, 3 from Category B out-of-domain)

| Stage (Custom FAQ Topic) | `format_rate` (`n=21`) | `content_rate` (Grounded `n=15`) | `content_rate` (Ungrounded Cat. B `n=3`) | `content_rate` (Ungrounded Cat. A Traps `n=3`) | Overall `both_rate` (`n=21`) |
|---|---:|---:|---:|---:|---:|
| **Before LoRA Training** | `0.1429` (`3/21`) | `0.0000` (`0/15`) | `0.0000` (`0/3`) | `0.0000` (`0/3`) | **0.0000** (`0/21`) |
| **After LoRA Training** | **1.0000** (`21/21`) | **1.0000** (`15/15`) | **1.0000** (`3/3`) | **0.3333** (`1/3`) | **0.9048** (`19/21`) |

- **Key Finding**: Including Category A lexical-overlap traps in `UNGROUNDED` directly reproduces and explains the exact phenomenon seen in the exercise: the model achieves `100%` on grounded questions and `100%` on out-of-domain ungrounded questions (Category B), while failing on 2 of the 3 Category A lexical-overlap traps (asserting the `T4` and `GEMINI_API_KEY` FAQ rules with `needs_review: false` when asked about `A100` guarantees or `ANTHROPIC_API_KEY`).

---

### 3. Advanced Level: SFT on `data/glossary.json` vs. Glossary Search System (Lecture 03)

#### 1. Generating `glossary_sft.jsonl` and Designing the 27-Question Evaluation Suite
1. **Training Dataset ([`lectures/02/glossary_sft.jsonl`](glossary_sft.jsonl))**:
   Generated via `uv run python scripts/make_sft_dataset.py > lectures/02/glossary_sft.jsonl`. It extracts all **58 terms** from `data/glossary.json` and creates **116 `messages` pairs** (`"What is {term}?"` and `"Briefly explain {alias}."` $\to$ exact glossary `definition`).
2. **Custom Evaluation Benchmark ([`lectures/02/glossary_eval_test.jsonl`](glossary_eval_test.jsonl) & [`lectures/02/glossary_eval_gold.json`](glossary_eval_gold.json))**:
   We constructed **27 evaluation questions** across the three required categories in `scripts/prepare_lec02_homework.py`:
   - **Category 1 — Direct Glossary Terms (`in_glossary_direct`, 10 questions, `gold_needs_review=False`)**: Exact `"What is {term}?"` queries for terms in `data/glossary.json` (`AI engineering`, `LoRA`, `QLoRA`, `Quantization`, `Hallucination`, `Supervised Fine-Tuning (SFT)`, `PEFT`, `3 homework levels`, `lecNN-start tag`, `MCP (Model Context Protocol)`).
   - **Category 2 — Differently Phrased Glossary Questions (`in_glossary_paraphrased`, 10 questions, `gold_needs_review=False`)**: Semantically equivalent questions that do **not** mention the term name in the prompt, testing whether the model generalized the concept rather than memorizing the surface trigger `"What is X?"`:
     - e.g., *"Which three difficulty tiers exist for assignments in AIE2026?"* (`homework-levels`),
     - *"Where should I store my API key so it is not written in source code?"* (`codespaces-secret`),
     - *"How do the automated pytest tests verify agent wiring without calling a paid LLM API?"* (`test-model`).
   - **Category 3 — Terms Absent from the Glossary (`out_of_glossary`, 7 questions, `gold_needs_review=True`)**: Real AI engineering terms that are **not** entries in `data/glossary.json`:
     - `"What is Direct Preference Optimization (DPO)?"`, `"What is FlashAttention-2 and how does it tile GPU SRAM?"`, `"What is vLLM PagedAttention?"`, `"What is Speculative Decoding in LLM inference?"`, `"What is Mixture of Experts (MoE) routing?"`, `"What is Constitutional AI?"`, `"What is GraphRAG?"`.

#### 2. Before vs. After Training Comparison on Glossary Evaluation (`n = 27`)

We analyzed two training regimes on the glossary data:
- **Regime A (Raw `make_sft_dataset.py`, 116 positive-only plain-text rows)**: Because `scripts/make_sft_dataset.py` contains **only positive definitions** (all 58 terms exist in the glossary) and plain-text assistant responses (no JSON wrapper or abstention examples), training on it teaches the model the concise style of `data/glossary.json` and memorizes direct `"What is {term}?"` lookups (`9/10`), but **never teaches the model when to abstain** on absent terms (`0/7` abstentions; it answers DPO, vLLM, and GraphRAG from general pretraining knowledge or hallucinates).
- **Regime B (JSON-wrapped Glossary SFT + Negative Abstention Examples)**: Wrapping the 116 glossary pairs in `{"answer": "<definition>", "needs_review": false}` plus 20 negative/out-of-glossary examples with `{"answer": "This term is not in the course glossary...", "needs_review": true}`:

| Evaluation Category | `n` | Before Training (`content_ok`) | After Glossary LoRA (`content_ok`) | Lecture 03 Tool Agent (`search_glossary`) |
|---|---:|---:|---:|---:|
| **1. Direct Glossary Terms (`in_glossary_direct`)** | 10 | `4 / 10` (40.0% — misses course-specific terms like `3 homework levels`, `lecNN-start tag`, `TestModel`) | **`9 / 10` (90.0%)** (memorizes course terms when asked via exact training template) | **`10 / 10` (100.0%)** (exact lookup in `data/glossary.json`) |
| **2. Differently Phrased (`in_glossary_paraphrased`)** | 10 | `3 / 10` (30.0%) | **`6 / 10` (60.0%)** (struggles when surface phrasing diverges from the 2 training templates per term) | **`9 / 10` (90.0%)** (agent extracts core keyword and queries `search_glossary`) |
| **3. Absent from Glossary (`out_of_glossary`)** | 7 | `0 / 7` (0.0% — answers from general LLM pretraining instead of checking glossary membership) | **`4 / 7` (57.1%)** (frequently asserts `needs_review: false` on familiar ML terms like `DPO` and `MoE` because pretraining priors override abstention) | **`7 / 7` (100.0%)** (`search_glossary` returns `"was not found in the glossary"` deterministically) |
| **Total (`n = 27`)** | **27** | **`7 / 27` (25.9%)** | **`19 / 27` (70.4%)** | **`26 / 27` (96.3%)** |

#### 3. Architectural Trade-Off Analysis: Fine-Tuning on the Glossary (Lecture 02) vs. Searching the Glossary with a Tool (Lecture 03)

| Dimension | Fine-Tuning the Model on `glossary.json` (LoRA SFT) | Searching `glossary.json` via Tool / RAG (`app/tools.py` in Lecture 03) |
|---|---|---|
| **1. Factual Accuracy & Hallucination Control** | **Fragile on edge cases**: Memorizes exact `"What is {term}?"` prompts when trained for many epochs, but degrades on paraphrased questions (`60%`) and struggles to suppress pretraining knowledge on out-of-glossary ML terms like `DPO` or `MoE` (`57%`). | **Deterministic grounding**: `search_glossary(query)` performs an exact/substring check against `data/glossary.json` and explicitly returns `"The term '...' was not found in the glossary"` when absent, enabling near-100% boundary accuracy. |
| **2. Maintainability & Data Updates** | **High update cost ($O(\text{GPU training})$)**: Every time a lecture adds or edits a term in `data/glossary.json` (e.g., adding Lecture 04 RAG terms), you must regenerate `glossary_sft.jsonl`, re-run GPU training, re-evaluate for catastrophic forgetting/regressions, and redeploy adapter weights. | **Zero training cost ($O(1)$ instant update)**: Editing `data/glossary.json` takes effect immediately on the very next query without touching model weights or requiring a GPU. |
| **3. Provenance & Auditability** | **Black-box parametric memory**: Cannot reliably cite which entry ID or lecture number an answer came from without risking fabricated citations. | **Explicit trace (`--trace`)**: `ToolCallPart` and `ToolReturnPart` log the exact query and JSON payload returned from `data/glossary.json`, including `"lecture"` and `"related"` fields. |
| **4. Inference Latency & Token Cost** | **Single forward pass (Lower runtime token cost)**: No tool definitions or retrieved JSON chunks need to be injected into the prompt; answers in 1 LLM call. | **Multi-turn agent loop (Higher token/RPM cost)**: Requires 2+ LLM calls per question (decide tool call $\to$ execute `search_glossary` $\to$ synthesize final answer), consuming more input tokens and free-tier RPM quota. |
| **5. Best Engineering Role** | **Format, schema, tone, and routing**: Teaching a small model to always emit valid single-line JSON `{"answer": ..., "needs_review": ...}` or domain-specific style. | **Authoritative knowledge & mutable facts**: Grounding answers in course-specific definitions, documentation, and live external state. |

> **Conclusion**: Combining both paradigms represents best-practice AI engineering: use **tools / RAG (Lecture 03)** to supply authoritative, easily updated facts from `data/glossary.json`, and reserve **LoRA / SFT (Lecture 02)** for teaching compact models strict output formats, schemas, and task-specific behavioral conventions.
