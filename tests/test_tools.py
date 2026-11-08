"""Tool tests (no LLM or API key used)."""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} has a nonexistent id in related: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start tag"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("agent"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "was not found" in tools.search_glossary("nonexistent-term-xyz")


def test_lecture_summary():
    assert "agent" in tools.get_lecture_summary(3)
    assert "not available yet" in tools.get_lecture_summary(99)


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """The Lecture 03 agent can correctly answer the Lecture 01 default question (app.ask) using the glossary."""
    from app.ask import DEFAULT_QUESTION

    assert "3 homework levels" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3 homework levels"))
    assert hits[0]["term"] == "3 homework levels"


def test_make_quiz_returns_question_and_answer_for_valid_lecture():
    quiz = json.loads(tools.make_quiz(2))
    assert "question" in quiz and "answer" in quiz
    assert quiz["question"].startswith("What does ")
    lec2_definitions = {e["definition"] for e in tools.load_glossary() if e["lecture"] == 2}
    assert quiz["answer"] in lec2_definitions


def test_make_quiz_handles_nonexistent_lecture_without_error():
    msg = tools.make_quiz(0)
    assert "No glossary terms found" in msg
    assert "00" in msg


def test_save_learning_log_creates_and_appends_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    res1 = json.loads(tools.save_learning_log("LoRA & QLoRA", "LoRA freezes weights; QLoRA adds 4-bit quantization."))
    assert res1["status"] == "saved"
    saved_file = tmp_path / "lora-qlora.md"
    assert saved_file.exists()
    text1 = saved_file.read_text(encoding="utf-8")
    assert "LoRA freezes weights" in text1

    res2 = json.loads(tools.save_learning_log("LoRA & QLoRA", "Follow-up note on rank r."))
    assert res2["status"] == "saved"
    text2 = saved_file.read_text(encoding="utf-8")
    assert "LoRA freezes weights" in text2
    assert "Follow-up note on rank r." in text2

