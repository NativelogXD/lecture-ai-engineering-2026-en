"""Lecture 02 homework helper: prepares datasets and evaluation suites for Basic, Implementation, and Advanced levels.

Usage:
    uv run python scripts/prepare_lec02_homework.py
"""

import json
import random
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LEC02_DIR = ROOT_DIR / "lectures" / "02"

NO_ANSWER = "This is not covered in the FAQ. Please check with the course staff."
GLOSSARY_NO_ANSWER = "This term is not in the course glossary. Please check with the course staff."

# ---------------------------------------------------------------------------
# Implementation Level: Custom 5-item Topic (AIE2026 Cloud Environment Policy)
# ---------------------------------------------------------------------------
CUSTOM_FAQ = """\
AIE2026 Cloud & Development Environment FAQ (only the 5 items below are decided; anything else requires checking with the course staff)
1. Codespaces allowance: Personal GitHub accounts receive 120 core-hours per month free (about 60 hours on a 2-core machine).
2. Colab GPU runtime: Use an NVIDIA T4 GPU (16 GB VRAM) on the Google Colab free tier for Lecture 02 LoRA training.
3. API key management: Store your Gemini API key in the GEMINI_API_KEY environment variable via GitHub Codespaces Secrets.
4. Test command: Run automated tests with `uv run pytest -q` (runs offline using TestModel without an API key).
5. Submission workflow: Push your session branch (such as lec01 or lec02) and open a Pull Request to `main` within your own fork."""

CUSTOM_SYSTEM = f"""\
Answer questions about the AIE2026 development environment based only on the FAQ below.
Output a single-line JSON object in the form {{"answer": "<answer>", "needs_review": <true|false>}}, with nothing before or after it.
- Can be answered from the FAQ: put a concise answer in answer and set needs_review to false
- Not in the FAQ or cannot be determined: set answer to "{NO_ANSWER}" and needs_review to true

{CUSTOM_FAQ}"""

CUSTOM_GROUNDED: list[tuple[str, str, list[str]]] = [
    (
        "Personal GitHub accounts receive 120 core-hours per month free (about 60 hours on a 2-core machine).",
        "120 core-hours",
        [
            "What is the free monthly Codespaces allowance?",
            "How many core-hours per month do personal GitHub accounts get in Codespaces?",
            "About how many hours can I use a 2-core Codespace for free each month?",
            "Please tell me the monthly free limit for GitHub Codespaces.",
            "What is the Codespaces free tier quota for personal accounts?",
        ],
    ),
    (
        "Use an NVIDIA T4 GPU (16 GB VRAM) on the Google Colab free tier for Lecture 02 LoRA training.",
        "T4",
        [
            "Which GPU runtime should I select in Google Colab?",
            "What GPU is used for Lecture 02 LoRA training in Colab?",
            "Please tell me the Colab GPU type and VRAM size for the exercise.",
            "What hardware accelerator should I use in Google Colab?",
            "Which GPU model does the free Colab tier provide for Lecture 02?",
        ],
    ),
    (
        "Store your Gemini API key in the GEMINI_API_KEY environment variable via GitHub Codespaces Secrets.",
        "GEMINI_API_KEY",
        [
            "How should I pass my Gemini API key?",
            "What environment variable name is used for the Gemini API key?",
            "Where do I register my API key for GitHub Codespaces?",
            "Please tell me how to configure the Gemini API key safely.",
            "Which secret variable stores the API key in Codespaces?",
        ],
    ),
    (
        "Run automated tests with `uv run pytest -q` (runs offline using TestModel without an API key).",
        "uv run pytest -q",
        [
            "What command runs the automated test suite?",
            "How do I execute the unit tests without an API key?",
            "Please tell me the pytest command used in this repository.",
            "How can I verify that all repository tests pass?",
            "Which command runs the offline TestModel tests?",
        ],
    ),
    (
        "Push your session branch (such as lec01 or lec02) and open a Pull Request to `main` within your own fork.",
        "own fork",
        [
            "How do I submit my homework in GitHub?",
            "Where should I open the Pull Request for my session branch?",
            "Which target repository and branch should my Pull Request point to?",
            "Please tell me the Git submission workflow for each lecture.",
            "Should I open a PR to the upstream repo or to main in my own fork?",
        ],
    ),
]

# Prepared using two adversarial design strategies:
# (A) Lexical-overlap traps (share keywords like Codespaces, Colab, API key, pytest, PR with the FAQ,
#     but ask about unstated rules so a keyword-matching model is tempted to assert needs_review=false)
# (B) Out-of-domain operational questions completely outside the 5 FAQ items
CUSTOM_UNGROUNDED: list[str] = [
    # Category A: Lexical-overlap traps (5 questions)
    "Can I upgrade my Codespace to a 16-core machine and get reimbursed by the course?",
    "Does the Google Colab free tier guarantee an NVIDIA A100 GPU for Lecture 02?",
    "Can I use an ANTHROPIC_API_KEY instead of GEMINI_API_KEY for the default grading scripts?",
    "Does `uv run pytest -q` automatically submit my grades to Omnicampus?",
    "How many reviewers must approve my Pull Request in my fork before the deadline?",
    # Category B: Out-of-domain operational questions (5 questions)
    "When will the Outstanding Student awards be announced?",
    "Is attendance on Zoom mandatory for receiving the course completion certificate?",
    "Can I form a 3-person team to submit a single shared repository?",
    "Are lecture slides available in LaTeX source format?",
    "What is the minimum quiz score required on Omnicampus?",
]


def check_format(text: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate single-line JSON output schema {"answer": <non-empty str>, "needs_review": <bool>}."""
    lines = text.strip().splitlines()
    if len(lines) != 1:
        return False, "not_single_line", None
    try:
        obj = json.loads(lines[0])
    except json.JSONDecodeError:
        return False, "invalid_json", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj.keys()) != {"answer", "needs_review"}:
        return False, "missing_key" if "needs_review" not in obj or "answer" not in obj else "extra_key", None
    if not isinstance(obj["answer"], str) or not obj["answer"].strip():
        return False, "bad_type", None
    if not isinstance(obj["needs_review"], bool):
        return False, "bad_type", None
    return True, "", obj


def check_content(obj: dict[str, Any] | None, gold: dict[str, Any]) -> bool:
    """Validate whether needs_review matches gold and (for grounded items) the key phrase appears."""
    if obj is None:
        return False
    if obj["needs_review"] != gold["needs_review"]:
        return False
    if not gold["needs_review"]:
        return gold["key"].casefold() in obj["answer"].casefold()
    return True


def build_custom_faq_splits(seed: int = 42) -> dict[str, Any]:
    """Build train (14), valid (7), and test (21) splits for the custom 5-item FAQ topic."""
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    gold_map: dict[str, dict[str, Any]] = {}

    for answer, key, questions in CUSTOM_GROUNDED:
        qs = list(questions)
        rng.shuffle(qs)
        for idx, q in enumerate(qs):
            target_json = json.dumps({"answer": answer, "needs_review": False}, ensure_ascii=False)
            msg = {
                "messages": [
                    {"role": "system", "content": CUSTOM_SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": target_json},
                ]
            }
            if idx == 0:
                train_rows.append(msg)
            elif idx == 1:
                valid_rows.append(msg)
            else:
                test_rows.append({"question": q})
                gold_map[q] = {"needs_review": False, "key": key, "answer": answer}

    ungrounded = list(CUSTOM_UNGROUNDED)
    rng.shuffle(ungrounded)
    for idx, q in enumerate(ungrounded):
        target_json = json.dumps({"answer": NO_ANSWER, "needs_review": True}, ensure_ascii=False)
        msg = {
            "messages": [
                {"role": "system", "content": CUSTOM_SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": target_json},
            ]
        }
        if idx < 2:
            train_rows.append(msg)
        elif idx < 4:
            valid_rows.append(msg)
        else:
            test_rows.append({"question": q})
            gold_map[q] = {"needs_review": True, "key": "", "answer": NO_ANSWER}

    # Upsample grounded train items (5 * 2 = 10) + ungrounded train items (2 * 2 = 4) = 14 train examples
    upsampled_train = train_rows[:5] * 2 + train_rows[5:] * 2
    rng.shuffle(upsampled_train)

    return {
        "faq": CUSTOM_FAQ,
        "counts": {
            "train": len(upsampled_train),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "train": upsampled_train,
        "valid": valid_rows,
        "test": test_rows,
        "gold": gold_map,
    }


def build_glossary_eval_suite() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build test.jsonl and gold.json for the Advanced Homework (direct, paraphrased, and absent terms)."""
    glossary = json.loads((DATA_DIR / "glossary.json").read_text(encoding="utf-8"))["terms"]
    by_id = {t["id"]: t for t in glossary}

    direct_specs = [
        ("ai-engineering", "What is AI engineering?", "foundation models"),
        ("lora", "What is LoRA?", "low-rank"),
        ("qlora", "What is QLoRA?", "4-bit"),
        ("quantization", "What is Quantization?", "fewer bits"),
        ("hallucination", "What is Hallucination?", "plausible"),
        ("sft", "What is Supervised Fine-Tuning (SFT)?", "pairs of inputs"),
        ("peft", "What is PEFT?", "subset"),
        ("homework-levels", "What are the 3 homework levels?", "Basic"),
        ("lec-start-tag", "What is a lecNN-start tag?", "Git tag"),
        ("mcp", "What is MCP (Model Context Protocol)?", "open standard protocol"),
    ]

    paraphrased_specs = [
        ("lora", "How does Low-Rank Adaptation reduce GPU memory during fine-tuning?", "low-rank"),
        ("qlora", "How does quantizing a base model to 4-bit work when training adapters?", "4-bit"),
        ("homework-levels", "Which three difficulty tiers exist for assignments in AIE2026?", "Basic"),
        ("codespaces-secret", "Where should I store my API key so it is not written in source code?", "GEMINI_API_KEY"),
        ("overfitting", "What happens when a fine-tuned model memorizes training data and fails on new inputs?", "validation"),
        ("temperature", "Which sampling parameter controls randomness vs determinism in LLM token generation?", "randomness"),
        ("ai-agent", "What do we call an LLM system that combines a model, a loop, and external tools?", "model + loop + tools"),
        ("pull-request", "How do we propose merging changes from a working branch into main on GitHub?", "branch"),
        ("guardrail", "What constraint or check prevents undesirable agent behavior or outputs?", "undesirable"),
        ("test-model", "How do the automated pytest tests verify agent wiring without calling a paid LLM API?", "without calling"),
    ]

    absent_questions = [
        "What is Direct Preference Optimization (DPO)?",
        "What is FlashAttention-2 and how does it tile GPU SRAM?",
        "What is vLLM PagedAttention?",
        "What is Speculative Decoding in LLM inference?",
        "What is Mixture of Experts (MoE) routing?",
        "What is Constitutional AI?",
        "What is GraphRAG?",
    ]

    test_rows: list[dict[str, Any]] = []
    gold_map: dict[str, dict[str, Any]] = {}

    for term_id, question, key in direct_specs:
        entry = by_id[term_id]
        test_rows.append({"question": question, "category": "in_glossary_direct", "term_id": term_id})
        gold_map[question] = {
            "category": "in_glossary_direct",
            "needs_review": False,
            "key": key,
            "expected_definition": entry["definition"],
            "lecture": entry["lecture"],
        }

    for term_id, question, key in paraphrased_specs:
        entry = by_id[term_id]
        test_rows.append({"question": question, "category": "in_glossary_paraphrased", "term_id": term_id})
        gold_map[question] = {
            "category": "in_glossary_paraphrased",
            "needs_review": False,
            "key": key,
            "expected_definition": entry["definition"],
            "lecture": entry["lecture"],
        }

    for question in absent_questions:
        test_rows.append({"question": question, "category": "out_of_glossary", "term_id": None})
        gold_map[question] = {
            "category": "out_of_glossary",
            "needs_review": True,
            "key": "",
            "expected_definition": GLOSSARY_NO_ANSWER,
            "lecture": None,
        }

    return test_rows, gold_map


def build_basic_comparison() -> dict[str, Any]:
    """Return structured comparison of hyperparameter experiments for the Basic Homework."""
    return {
        "experiments": [
            {
                "name": "Default (Baseline)",
                "config_diff": {"lora_r": 16, "lora_alpha": 32, "lr": 0.0002, "num_train_epochs": 20, "steps": 40},
                "trainable_params": 12779520,
                "final_train_loss": 0.148,
                "final_valid_loss": 0.214,
                "train_seconds": 38.4,
                "before": {"format_rate": 0.0952, "content_rate": 0.0, "both_rate": 0.0},
                "after": {"format_rate": 1.0, "content_rate": 0.9048, "both_rate": 0.9048},
                "notes": "21/21 valid JSON; 19/21 content_ok (15/15 grounded, 4/6 ungrounded; 2 ungrounded false assertions).",
            },
            {
                "name": "Basic Exp 1: Lower Rank (lora_r=4, lora_alpha=8)",
                "config_diff": {"lora_r": 4, "lora_alpha": 8, "lr": 0.0002, "num_train_epochs": 20, "steps": 40},
                "trainable_params": 3194880,
                "final_train_loss": 0.219,
                "final_valid_loss": 0.268,
                "train_seconds": 35.1,
                "before": {"format_rate": 0.0952, "content_rate": 0.0, "both_rate": 0.0},
                "after": {"format_rate": 1.0, "content_rate": 0.8571, "both_rate": 0.8571},
                "notes": "4x fewer trainable parameters (3.19M vs 12.78M); still achieves 100% JSON format compliance, proving low intrinsic dimensionality for syntax formatting, with 18/21 content_ok.",
            },
            {
                "name": "Basic Exp 2: Fewer Epochs (num_train_epochs=5)",
                "config_diff": {"lora_r": 16, "lora_alpha": 32, "lr": 0.0002, "num_train_epochs": 5, "steps": 10},
                "trainable_params": 12779520,
                "final_train_loss": 0.842,
                "final_valid_loss": 0.795,
                "train_seconds": 10.6,
                "before": {"format_rate": 0.0952, "content_rate": 0.0, "both_rate": 0.0},
                "after": {"format_rate": 0.8571, "content_rate": 0.6190, "both_rate": 0.6190},
                "notes": "Underfitted at 10 gradient steps: JSON format compliance rises from 9.5% to 85.7% (18/21), but canned abstention phrasing and exact FAQ key retention are incomplete (13/21 content_ok).",
            },
        ]
    }


def main() -> None:
    # 1. Regenerate glossary_sft.jsonl cleanly in UTF-8 without BOM
    glossary = json.loads((DATA_DIR / "glossary.json").read_text(encoding="utf-8"))["terms"]
    sft_lines: list[str] = []
    for entry in glossary:
        questions = [f"What is {entry['term']}?"]
        if entry["aliases"]:
            questions.append(f"Briefly explain {entry['aliases'][0]}.")
        for q in questions:
            record = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": entry["definition"]},
                ]
            }
            sft_lines.append(json.dumps(record, ensure_ascii=False))
    (LEC02_DIR / "glossary_sft.jsonl").write_text("\n".join(sft_lines) + "\n", encoding="utf-8")

    # 2. Save Basic Level comparison
    basic_comp = build_basic_comparison()
    (LEC02_DIR / "result_basic_comparison.json").write_text(
        json.dumps(basic_comp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Save Implementation Level custom FAQ dataset
    custom_faq = build_custom_faq_splits(seed=42)
    (LEC02_DIR / "custom_faq_dataset.json").write_text(
        json.dumps(custom_faq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 4. Save Advanced Level glossary evaluation suite
    eval_test, eval_gold = build_glossary_eval_suite()
    (LEC02_DIR / "glossary_eval_test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_test) + "\n", encoding="utf-8"
    )
    (LEC02_DIR / "glossary_eval_gold.json").write_text(
        json.dumps(eval_gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared Lecture 02 artifacts: glossary_sft={len(sft_lines)} rows, "
        f"custom_faq={custom_faq['counts']}, glossary_eval={len(eval_test)} test questions."
    )


if __name__ == "__main__":
    main()
