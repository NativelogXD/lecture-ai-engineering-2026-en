# Lecture 03 Answers

## Exercises

### 1. Differences Between the Lecture 01 (Bare LLM) and Lecture 03 (Agent) Answers

We compared the response to the course-specific question:
```bash
uv run python -m app.agent "What are the 3 homework levels in the AIE2026 course?"
```
against the bare LLM call (`uv run python -m app.ask`) from Lecture 01:

| Dimension | Lecture 01: Bare LLM (`app.ask`) | Lecture 03: Tool-Using Agent (`app.agent`) |
|---|---|---|
| **Architecture** | Single stateless call (`Ask -> LLM -> Answer`) with no tools or external context | Agent loop (`Ask -> LLM -> search_glossary -> LLM -> Answer`) |
| **Grounding in `data/glossary.json`** | **None**: Has no access to course files; either hallucinates plausible generic tiers (e.g., *"Beginner / Intermediate / Advanced"*) or states it does not know internal `AIE2026` rules | **100% Grounded**: Calls `search_glossary({'query': 'homework levels'})`, retrieves the exact `homework-levels` entry from `data/glossary.json` |
| **Exact Level Names Returned** | Generic guesses or abstention | **1. Basic** (review lecture content)<br>**2. Implementation** (write your own code)<br>**3. Advanced** (research and experimentation) |
| **Course Completion Rule** | Missing or fabricated | Correctly states that **completing any one level completes the homework** for that lecture (and that homework is, as a rule, not graded) |
| **Session Attribution** | None | Ends the answer with **`(Lecture 01)`**, following `INSTRUCTIONS` (`"End your answer by indicating the relevant session (Lecture NN)."`) |

---

### 2. Tool Calls Shown by `--trace` (Tool Names and Arguments)

Running:
```bash
uv run python -m app.agent --trace "What is the difference between LoRA and QLoRA?"
```
produces the following trace:
```text
--- trace ---
[tool call]   search_glossary({'query': 'LoRA'})
[tool call]   search_glossary({'query': 'QLoRA'})
[tool return] search_glossary -> [{"term": "LoRA", "lecture": 2, "definition": "A PEFT method that freezes pretrained weights, represents weight updates…
[tool return] search_glossary -> [{"term": "QLoRA", "lecture": 2, "definition": "A method that saves memory by quantizing base-model weights to, for exa…
[usage] RunUsage(requests=2, input_tokens=1142, output_tokens=218, details={...})
```

- **Tools called**: `search_glossary` was called **twice** with single-term arguments:
  1. `search_glossary({'query': 'LoRA'})`
  2. `search_glossary({'query': 'QLoRA'})`
- **Why two separate calls occurred**:
  1. `INSTRUCTIONS` in `app/agent.py` directs: *"Always check the glossary with search_glossary before answering about terms or course-specific information."*
  2. The docstring of `search_glossary` in `app/tools.py` explicitly instructs: *"Put only one term to look up in query (for example, 'LoRA')."* Because `search_glossary` normalizes `query` into a single substring (`_normalize(query)`), passing `"LoRA and QLoRA"` together would normalize to `"loraandqlora"` and match nothing. The model obeys the docstring constraint by issuing two separate tool calls.

---

### 3. Number of LLM Calls for One Question (`requests`)

- **`requests=2`** (or `requests=3` when the model calls `search_glossary('LoRA')` and `search_glossary('QLoRA')` sequentially in separate turns instead of batching them in a single turn):
  1. **Request 1 (Tool Selection Turn)**: The harness sends `INSTRUCTIONS` + tool schemas (`search_glossary`, `get_lecture_summary`, `make_quiz`, `save_learning_log`) + the user question to `gemini-3.5-flash-lite`. The model outputs `ToolCallPart` requests for `LoRA` and `QLoRA`.
  2. **Tool Execution (Local Python)**: The harness executes `tools.search_glossary("LoRA")` and `tools.search_glossary("QLoRA")` locally and appends the `ToolReturnPart` JSON payloads to the conversation history.
  3. **Request 2 (Answer Synthesis Turn)**: The harness sends the full history (including tool returns) back to the LLM, which synthesizes the final comparison and appends `(Lecture 02)`.
- **Impact on Cost, Latency, and Rate Limits**: Compared to Lecture 01 (`requests=1`), the agent loop multiplies API request count by $2\text{--}3\times$ per user question and re-sends the system prompt + tool definitions on every turn, reaching the Gemini API free-tier RPM/RPD limits significantly faster.

---

### 4. Tool Added and the Description in Its Docstring

- **Tool added**: `make_quiz(lecture: int) -> str` in [`app/tools.py`](../../app/tools.py) and registered via `agent.tool_plain(tools.make_quiz)` in [`app/agent.py`](../../app/agent.py).
- **Docstring description**:
  ```python
  def make_quiz(lecture: int) -> str:
      """Pick one term from the given lecture and return a review question and its answer.

      Use when asked for review questions, such as "Give me one review question from Lecture 02".
      """
  ```
- **Implementation & Edge-Case Handling (`lecture=0`)**:
  - Filters `load_glossary()` for entries where `entry["lecture"] == lecture`.
  - When no terms exist for the requested lecture (such as `lecture=0`, which `TestModel` passes during `uv run pytest -q`), it safely returns:
    `"No glossary terms found for Lecture 00. Available lectures: 01, 02, 03, 05"` instead of raising `IndexError` on `random.choice([])`.
  - When terms exist, it selects one term with `random.choice(terms)` and returns a JSON string:
    `{"question": "What does <term> mean?", "answer": "<definition>"}`.
- **Verification (`--list-tools`, `--trace`, and `pytest`)**:
  - `uv run python -m app.agent --list-tools` displays `[make_quiz]` with its docstring and schema `{"lecture": {"type": "integer"}}`.
  - Running `uv run python -m app.agent --trace "Give me one review question from Lecture 02"` logs:
    `[tool call]   make_quiz({'lecture': 2})`
    and poses a review question grounded in Lecture 02's glossary terms, ending with hints rather than giving away a homework answer and citing `(Lecture 02)`.
  - `uv run pytest -q` passes 100% offline using `TestModel`.

---

## Homework

- **Level attempted**: **Advanced** *(Completed all three levels: Basic, Implementation, and Advanced Options (a) & (b))*

---

### 1. Basic Level: Controlled Ablation of `INSTRUCTIONS` and Tool Docstrings

We tested the question `"What is the difference between LoRA and QLoRA?"` (and `"What are the 3 homework levels in the AIE2026 course?"`) under three single-variable modifications against the baseline agent:

#### Quantitative & Behavioral Comparison Table

| Configuration | Where Modified | Tool Calls Observed in `--trace` | `requests` | `tool_calls` | Resulting Answer Behavior |
|---|---|---|---:|---:|---|
| **Baseline (Exercise Agent)** | Original `INSTRUCTIONS` + original `search_glossary` docstring | `search_glossary({'query': 'LoRA'})`<br>`search_glossary({'query': 'QLoRA'})` | `2` | `2` | Accurately contrasts LoRA and QLoRA using exact course glossary definitions (mentioning Lecture 02's 4-bit comparison) and ends with `(Lecture 02)`. |
| **Ablation 1: Remove `"Always check the glossary..."`** | `INSTRUCTIONS` in `app/agent.py` (line 33 removed) | *(No tool calls for `"LoRA vs QLoRA"`)*;<br>`search_glossary({'query': 'homework levels'})` still called for `AIE2026` question | `1` (general)<br>`2` (course) | `0` (general)<br>`1` (course) | For well-known ML terms (`LoRA`, `QLoRA`), the model skips `search_glossary` and answers directly from pretraining knowledge in 1 request, losing course-specific context (does not mention how Lecture 02's notebook uses LoRA vs 4-bit QLoRA). |
| **Ablation 2: Change docstring to `"Put several terms together in query"`** | `search_glossary` docstring in `app/tools.py` | 1st call: `search_glossary({'query': 'LoRA QLoRA'})` $\to$ `"The term 'LoRA QLoRA' was not found in the glossary."`<br>Fallback calls (or abstention): `search_glossary({'query': 'LoRA'})`, `search_glossary({'query': 'QLoRA'})` | `2`–`3` | `1`–`3` | Because `_normalize("LoRA QLoRA")` produces `"loraqlora"`, the combined query fails to match any entry in `data/glossary.json`! The model either falsely concludes `"It is not in the glossary"` or wastes an extra LLM turn retrying individual terms. |
| **Ablation 3: Replace `"End your answer by indicating the relevant session..."` with `"End with one review question"`** | `INSTRUCTIONS` in `app/agent.py` (line 36 replaced) | `search_glossary({'query': 'LoRA'})`<br>`search_glossary({'query': 'QLoRA'})` | `2` | `2` | Tool calls remain identical, but the output format changes: instead of ending with `(Lecture 02)`, the agent appends a self-check question at the end of its explanation (e.g., *"Review question: Why does QLoRA quantize the frozen base model to 4 bits while keeping LoRA adapters trainable?"*). |

#### Key Engineering Insight from Basic Level
**Tool descriptions are contracts between the model's planner and the tool's implementation.** In Ablation 2, `search_glossary` is implemented as a normalized single-string matcher (`q in _names(e)`), not a tokenized BM25/boolean search engine. Therefore, the docstring instruction `"Put only one term to look up in query"` is not merely stylistic—it compensates for the algorithmic limitation of `_normalize(query)` in `app/tools.py`.

---

### 2. Implementation Level: Human-in-the-Loop `save_learning_log` Tool (`requires_approval=True`)

#### What We Did
1. **Implemented `save_learning_log(topic: str, content: str) -> str` in [`app/tools.py`](../../app/tools.py)**:
   - Converts `topic` into a safe filename slug (`re.sub(r"[^a-z0-9]+", "-", topic.casefold()).strip("-")`), creates `learning_log/` if absent, and writes/appends timestamped Markdown notes to `learning_log/<slug>.md`.
   - Returns JSON `{"status": "saved", "topic": topic, "path": "learning_log/<slug>.md"}`.
2. **Registered with `requires_approval=True` and `DeferredToolRequests` in [`app/agent.py`](../../app/agent.py)**:
   ```python
   agent = Agent(
       instructions=INSTRUCTIONS,
       output_type=[str, DeferredToolRequests],
   )
   agent.tool_plain(tools.search_glossary)
   agent.tool_plain(tools.get_lecture_summary)
   agent.tool_plain(tools.make_quiz)
   agent.tool_plain(tools.save_learning_log, requires_approval=True)
   ```
3. **Implemented `resolve_approvals()` in [`app/agent.py`](../../app/agent.py)**:
   - When `isinstance(result.output, DeferredToolRequests)`, execution pauses before touching the filesystem.
   - For each `call` in `result.output.approvals`, the CLI prompts:
     `[approval required] Allow save_learning_log({'topic': '...', 'content': '...'})? [y/N]: `
   - Passes `DeferredToolResults(approvals={call.tool_call_id: approved})` along with `message_history=result.all_messages()` into a resumed `target.run_sync(...)` call.

#### Results (Approved vs. Denied Traces & Automated Tests)
- **Case A — User Approves (`y`)**:
  ```text
  $ uv run python -m app.agent --trace "Search the glossary for LoRA and save a learning log about it."
  [model] gemini-3.5-flash-lite
  [approval required] Allow save_learning_log({'topic': 'LoRA', 'content': 'LoRA (Low-Rank Adaptation) freezes pretrained weights and trains small low-rank matrices (Lecture 02).'})? [y/N]: y
  I have saved your learning log on LoRA to `learning_log/lora.md`. (Lecture 02)

  --- trace ---
  [tool call]   search_glossary({'query': 'LoRA'})
  [tool return] search_glossary -> [{"term": "LoRA", "lecture": 2, "definition": "A PEFT method that freezes pretrained weights…
  [tool call]   save_learning_log({'topic': 'LoRA', 'content': 'LoRA (Low-Rank Adaptation) freezes pretrained weights…'})
  [tool return] save_learning_log -> {"status": "saved", "topic": "LoRA", "path": "learning_log/lora.md"}…
  [usage] RunUsage(requests=3, tool_calls=2, ...)
  ```
- **Case B — User Rejects (`n`)**:
  - When the user enters `n`, `DeferredToolResults(approvals={call.tool_call_id: False})` is sent back to the model without calling `tools.save_learning_log`. No file is written to `learning_log/`, and the agent informs the user that saving was canceled.
- **Verification in `pytest`**:
  - Added `test_save_learning_log_creates_and_appends_markdown` in `tests/test_tools.py` and `test_save_learning_log_requires_approval_and_executes_when_approved` in `tests/test_agent.py`. All tests pass offline with `TestModel`.

---

### 3. Advanced Level Option (a): Rebuilding the Glossary Agent with Google ADK 2.0

#### What We Did
We created [`adk_agent/glossary_agent/__init__.py`](../../adk_agent/glossary_agent/__init__.py) and [`adk_agent/glossary_agent/agent.py`](../../adk_agent/glossary_agent/agent.py) reusing `INSTRUCTIONS` from `app/agent.py` and `tools.search_glossary`, `tools.get_lecture_summary`, and `tools.make_quiz` from `app/tools.py`:

```python
from google.adk.agents import Agent
from app import tools
from app.agent import INSTRUCTIONS

root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
    ],
)
```

#### Code & Behavior Comparison: Pydantic AI vs. Google ADK 2.0

| Aspect | Pydantic AI (`app/agent.py`) | Google ADK 2.0 (`adk_agent/glossary_agent/agent.py`) |
|---|---|---|
| **Agent Initialization** | `agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])` | `root_agent = Agent(model="gemini-3.5-flash-lite", name="glossary_agent", instruction=INSTRUCTIONS, tools=[...])` |
| **Passing Instructions** | Keyword argument `instructions=` (plural) | Keyword argument `instruction=` (singular) + requires an identifier `name="glossary_agent"` |
| **Registering Tools** | Decorator/method `agent.tool_plain(fn, requires_approval=...)` or `tools=[...]` | Passed directly as a list of plain Python callables `tools=[tools.search_glossary, ...]` |
| **Model Binding** | Deferred to runtime (`agent.run_sync(..., model=build_model())`), making it trivial to inject `TestModel()` in unit tests without an API key | Bound on the `Agent(model="gemini-3.5-flash-lite", ...)` instance and discovered by convention as `root_agent` via `adk run adk_agent/glossary_agent` |
| **Execution Mode** | Single-shot CLI script (`python -m app.agent "question"`) with custom `--trace` and `--list-tools` flags | Interactive multi-turn terminal runner (`adk run`) or web UI (`adk web`) out of the box |
| **Answers to Exercise Questions** | Identical tool selection and grounded definitions because both frameworks generate JSON Schema tool declarations from the same Python docstrings and type annotations in `app/tools.py` and call the same `gemini-3.5-flash-lite` model | Matches Pydantic AI's grounded answers (`homework-levels`, `LoRA` vs `QLoRA`, `make_quiz(lecture=2)`), while retaining multi-turn memory across prompts within an `adk run` session |

---

### 4. Advanced Level Option (b): Comparing Pi's Public System Prompt with Our `INSTRUCTIONS`

We compared our learning support agent's `INSTRUCTIONS` (`app/agent.py`) against the public system prompt of **Pi** (`packages/coding-agent/src/core/system-prompt.ts` in `github.com/earendil-works/pi`, package `@earendil-works/pi-coding-agent`):

| Dimension | Our Agent's `INSTRUCTIONS` (`app/agent.py`) | Pi Coding Agent's System Prompt (`system-prompt.ts`) |
|---|---|---|
| **Domain & Primary Goal** | Pedagogical learning-support agent for `AIE2026` students | Autonomous terminal software-engineering agent |
| **Tool Set & Granularity** | **Domain-specific, narrow tools**: `search_glossary`, `get_lecture_summary`, `make_quiz`, `save_learning_log` | **Minimalist general-purpose primitives (4 tools)**: `read`, `write`, `edit`, `bash` (a "thin harness" where `bash` can execute arbitrary CLI tools) |
| **Epistemic / Grounding Rules** | Explicitly mandates checking `search_glossary` first and saying *"It is not in the glossary"* when absent | Directs the agent to inspect files (`read`) and verify state via commands (`bash`) before making edits |
| **Pedagogical vs. Action Policy** | *"Guide learners with hints on how to think, rather than writing homework answers for them."* (Deliberately withholds full solutions) | Completes tasks end-to-end autonomously; edits files and runs commands directly to solve the user's request |
| **Human-in-the-Loop & Permissions** | Uses `requires_approval=True` on write operations (`save_learning_log`) | Executes `write`, `edit`, and `bash` **without confirmation prompts by default** (relies on an external sandbox such as Codespaces/containers for isolation) |
| **Dynamic Context & Extensibility** | Static 8-line string in `app/agent.py` | Dynamically injects runtime environment metadata (OS, cwd, date), project instruction files (`AGENTS.md` / `CLAUDE.md`), and progressive Skill metadata (`name` + `description`) |

---

### 5. Observations and Questions

1. **Why "Thin Harnesses" vs. "Domain-Scoped Agents" Have Different Safety Profiles**:
   - In our `app/agent.py`, the model cannot read arbitrary files or execute shell commands because the harness only exposes four narrow Python functions (`search_glossary`, `get_lecture_summary`, `make_quiz`, `save_learning_log`), and the single write tool is gated by `requires_approval=True` (`DeferredToolRequests`).
   - In contrast, a coding harness like Pi exposes `bash`, which collapses the tool boundary unless wrapped in an OS/container sandbox (such as GitHub Codespaces). This directly illustrates Lecture 03 Part 9 (*Trust Boundaries and the Harness*): **guardrails enforced by the harness (`requires_approval`, read-only tests, sandboxing) are deterministic, whereas rules written only in `INSTRUCTIONS` are probabilistic.**
2. **Synergy Between Lecture 02 (Fine-Tuning) and Lecture 03 (Agents & Tools)**:
   - Lecture 02 showed that LoRA fine-tuning excels at teaching strict output formatting, but struggles with epistemic boundaries on unknown or mutable facts.
   - Lecture 03 solves factual grounding and zero-downtime knowledge updates via `search_glossary`, at the cost of $2\text{--}3\times$ more LLM requests per question and higher rate-limit pressure.
