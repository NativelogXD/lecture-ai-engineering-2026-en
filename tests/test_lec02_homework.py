"""Lecture 02 tests for SFT dataset generation, format/content validators, and evaluation suites."""

import json
from pathlib import Path

from scripts import prepare_lec02_homework as lec02

ROOT = Path(__file__).resolve().parent.parent
LEC02_DIR = ROOT / "lectures" / "02"


def test_result_json_has_expected_schema_and_metrics() -> None:
    data = json.loads((LEC02_DIR / "result.json").read_text(encoding="utf-8"))
    assert data["config"]["base_model"] == "Qwen/Qwen3.5-0.8B"
    assert data["before"]["n"] == 21
    assert data["after"]["n"] == 21
    assert data["after"]["format_rate"] == 1.0
    assert data["after"]["content_rate"] > data["before"]["content_rate"]
    assert data["regressed"] == []
    assert len(data["knowledge"]) == 4


def test_check_format_and_content_validators() -> None:
    ok, reason, obj = lec02.check_format('{"answer": "120 core-hours per month.", "needs_review": false}')
    assert ok is True
    assert reason == ""
    assert lec02.check_content(obj, {"needs_review": False, "key": "120 core-hours"}) is True

    ok_missing, reason_missing, _ = lec02.check_format('{"answer": "120 core-hours"}')
    assert ok_missing is False
    assert reason_missing == "missing_key"


def test_custom_faq_splits_have_no_data_leakage() -> None:
    splits = lec02.build_custom_faq_splits(seed=42)
    assert splits["counts"] == {"train": 14, "valid": 7, "test": 21}

    train_qs = {m["messages"][1]["content"] for m in splits["train"]}
    valid_qs = {m["messages"][1]["content"] for m in splits["valid"]}
    test_qs = {r["question"] for r in splits["test"]}

    assert train_qs.isdisjoint(valid_qs)
    assert train_qs.isdisjoint(test_qs)
    assert valid_qs.isdisjoint(test_qs)
    assert len(splits["gold"]) == 21


def test_glossary_sft_and_eval_suites() -> None:
    sft_lines = (LEC02_DIR / "glossary_sft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sft_lines) == 116
    first_row = json.loads(sft_lines[0])
    assert [m["role"] for m in first_row["messages"]] == ["user", "assistant"]

    test_rows, gold_map = lec02.build_glossary_eval_suite()
    assert len(test_rows) == 27
    categories = {r["category"] for r in test_rows}
    assert categories == {"in_glossary_direct", "in_glossary_paraphrased", "out_of_glossary"}
    assert sum(1 for g in gold_map.values() if g["needs_review"]) == 7
