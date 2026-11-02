"""Lecture 01: minimal script that asks the LLM a single question or conducts a multi-turn chat.

Usage:
    uv run python -m app.ask "What is LoRA?"
    uv run python -m app.ask                                     # Default question (about this course)
    uv run python -m app.ask --chat "What is LoRA?"              # Multi-turn chat retaining conversation history
    uv run python -m app.ask --temperature 0.7 --runs 3 "What is LoRA?"  # Repeat N runs and summarize latency/tokens
    AIE_MODEL=gemini-3.5-flash uv run python -m app.ask "What is LoRA?"
"""

import argparse
import statistics
import time
from typing import Any

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# Default question: something specific to this course that a bare LLM does not know.
# Check the correct answer in data/glossary.json (id: "homework-levels").
# In Lecture 03, you will give the agent a glossary-search tool so it can answer accurately.
DEFAULT_QUESTION = "What are the 3 homework levels in the AIE2026 course?"


def _build_config(temperature: float | None = None) -> types.GenerateContentConfig | None:
    """Build an optional GenerateContentConfig when sampling parameters are specified."""
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def print_usage(response: Any, elapsed_sec: float | None = None) -> dict[str, int | float]:
    """Display input/output/thoughts/total token counts from response.usage_metadata and optional latency."""
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage is not None else 0
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage is not None else 0
    thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0) if usage is not None else 0
    total_tokens = (
        int(getattr(usage, "total_token_count", 0) or (prompt_tokens + output_tokens + thoughts_tokens))
        if usage is not None
        else 0
    )

    if usage is not None:
        latency_part = f" latency={elapsed_sec:.2f}s" if elapsed_sec is not None else ""
        print(
            f"[tokens] input={prompt_tokens} output={output_tokens} "
            f"thoughts={thoughts_tokens} total={total_tokens}{latency_part}"
        )
    elif elapsed_sec is not None:
        print(f"[latency] {elapsed_sec:.2f}s")

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed_sec, 4) if elapsed_sec is not None else 0.0,
    }


def ask(question: str, *, temperature: float | None = None) -> str:
    """Send a single-turn question to the configured Gemini model and return its text response."""
    client = genai.Client(api_key=get_api_key())
    config = _build_config(temperature)
    start_time = time.perf_counter()
    try:
        if config is not None:
            response = client.models.generate_content(model=MODEL, contents=question, config=config)
        else:
            response = client.models.generate_content(model=MODEL, contents=question)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    elapsed_sec = time.perf_counter() - start_time
    # Lecture 01 exercise & advanced homework: display input/output token counts and response time
    print_usage(response, elapsed_sec=elapsed_sec)
    return response.text or ""


def chat(first_question: str, *, temperature: float | None = None) -> None:
    """Conduct a multi-turn conversation while retaining chat history across follow-up questions."""
    client = genai.Client(api_key=get_api_key())
    config = _build_config(temperature)
    if config is not None:
        session = client.chats.create(model=MODEL, config=config)
    else:
        session = client.chats.create(model=MODEL)

    question = first_question
    turn = 1
    while question:
        start_time = time.perf_counter()
        try:
            response = session.send_message(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        elapsed_sec = time.perf_counter() - start_time

        print(f"\n[turn {turn}]")
        print(response.text or "")
        print_usage(response, elapsed_sec=elapsed_sec)

        try:
            question = input("\nFollow-up question (empty Enter to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        turn += 1


def benchmark(
    question: str,
    runs: int = 3,
    *,
    temperature: float | None = None,
) -> list[dict[str, int | float]]:
    """Run the same question multiple times and print per-run responses plus median latency/token statistics."""
    client = genai.Client(api_key=get_api_key())
    config = _build_config(temperature)
    records: list[dict[str, int | float]] = []

    for idx in range(1, runs + 1):
        print(f"\n--- Run {idx}/{runs} ---")
        start_time = time.perf_counter()
        try:
            if config is not None:
                response = client.models.generate_content(model=MODEL, contents=question, config=config)
            else:
                response = client.models.generate_content(model=MODEL, contents=question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        elapsed_sec = time.perf_counter() - start_time
        print(response.text or "")
        metrics = print_usage(response, elapsed_sec=elapsed_sec)
        records.append(metrics)

    if records:
        latencies = [float(r["elapsed_sec"]) for r in records]
        totals = [int(r["total_tokens"]) for r in records]
        outputs = [int(r["output_tokens"]) for r in records]
        print(
            f"\n[summary ({runs} runs)] "
            f"latency_median={statistics.median(latencies):.2f}s "
            f"(min={min(latencies):.2f}s, max={max(latencies):.2f}s) | "
            f"output_tokens_median={statistics.median(outputs):.0f} | "
            f"total_tokens_median={statistics.median(totals):.0f}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the LLM a single question, run multi-turn chat (--chat), or benchmark repeated runs (--runs)."
    )
    parser.add_argument("question", nargs="*", help="Question to ask the LLM (defaults to course-specific question)")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Continue with follow-up questions while retaining conversation history",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (for example, 0.0, 0.7, 1.5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to ask the same question (for Basic/Advanced comparison)",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() or DEFAULT_QUESTION
    temp_info = f" (temperature={args.temperature})" if args.temperature is not None else ""
    print(f"[model] {MODEL}{temp_info}\n[question] {question}\n")

    if args.chat:
        chat(question, temperature=args.temperature)
    elif args.runs > 1:
        benchmark(question, runs=args.runs, temperature=args.temperature)
    else:
        print(ask(question, temperature=args.temperature))


if __name__ == "__main__":
    main()
