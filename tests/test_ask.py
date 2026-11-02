"""Lecture 01 tests for app/ask.py (no LLM or API key used)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from app import ask


def _fake_response(
    text: str = "LoRA is Low-Rank Adaptation.",
    prompt_tokens: int = 12,
    output_tokens: int = 45,
    thoughts_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts_tokens,
            total_token_count=prompt_tokens + output_tokens + thoughts_tokens,
        ),
    )


def test_ask_prints_token_usage_and_returns_text(capsys: pytest.CaptureFixture[str]) -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response()

    with patch("app.ask.get_api_key", return_value="fake-key"), patch(
        "app.ask.genai.Client", return_value=fake_client
    ):
        answer = ask.ask("What is LoRA?", temperature=0.2)

    assert answer == "LoRA is Low-Rank Adaptation."
    out = capsys.readouterr().out
    assert "[tokens] input=12 output=45 thoughts=0 total=57" in out
    assert "latency=" in out
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["contents"] == "What is LoRA?"
    assert call_kwargs["config"].temperature == 0.2


def test_chat_retains_session_across_follow_up_questions(capsys: pytest.CaptureFixture[str]) -> None:
    fake_session = MagicMock()
    fake_session.send_message.side_effect = [
        _fake_response("Turn 1 answer", prompt_tokens=10, output_tokens=20),
        _fake_response("Turn 2 answer", prompt_tokens=38, output_tokens=25),
    ]
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_session

    with patch("app.ask.get_api_key", return_value="fake-key"), patch(
        "app.ask.genai.Client", return_value=fake_client
    ), patch("builtins.input", side_effect=["How does it differ from QLoRA?", ""]):
        ask.chat("What is LoRA?")

    assert fake_session.send_message.call_count == 2
    assert fake_session.send_message.call_args_list[0].args == ("What is LoRA?",)
    assert fake_session.send_message.call_args_list[1].args == ("How does it differ from QLoRA?",)
    out = capsys.readouterr().out
    assert "[turn 1]" in out
    assert "[turn 2]" in out
    assert "input=38 output=25" in out


def test_benchmark_runs_multiple_times_and_summarizes(capsys: pytest.CaptureFixture[str]) -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        _fake_response("Run 1", prompt_tokens=8, output_tokens=30),
        _fake_response("Run 2", prompt_tokens=8, output_tokens=32),
        _fake_response("Run 3", prompt_tokens=8, output_tokens=31),
    ]

    with patch("app.ask.get_api_key", return_value="fake-key"), patch(
        "app.ask.genai.Client", return_value=fake_client
    ):
        records = ask.benchmark("What is LoRA?", runs=3, temperature=0.7)

    assert len(records) == 3
    out = capsys.readouterr().out
    assert "[summary (3 runs)]" in out
    assert "output_tokens_median=31" in out


def test_ask_handles_api_error_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    fake_client = MagicMock()
    api_err = errors.APIError(code=429, response_json={"error": {"message": "RESOURCE_EXHAUSTED"}})
    fake_client.models.generate_content.side_effect = api_err

    with patch("app.ask.get_api_key", return_value="fake-key"), patch(
        "app.ask.genai.Client", return_value=fake_client
    ), pytest.raises(SystemExit) as exc_info:
        ask.ask("What is LoRA?")

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Rate limit reached. Wait about 1 minute" in out
