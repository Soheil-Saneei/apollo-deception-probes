#!/usr/bin/env python3
"""Generate paired neutral and deferential responses through Ollama.

Each entry in the prompt file carries one underlying question and one
implicit-deference line. The neutral condition sends the bare question; the
deferential condition sends the question followed by that line. Every prompt
is sampled --samples times independently, so 30 questions produce 60 distinct
prompts and, at the default of 3 samples, 180 responses.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CONDITIONS = ("neutral", "deferential")


def build_prompt(item: dict, condition: str) -> str:
    question = item["question"].strip()
    if condition == "neutral":
        return question
    return f"{question} {item['deference_variant'].strip()}"


def query_ollama(base_url: str, model: str, prompt: str, think: bool) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False, "think": think}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="independent responses to generate per prompt (default: 3)",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the model's hidden reasoning pass (default: off)",
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")

    prompts = json.loads(args.prompts.read_text())
    args.outputs.mkdir(parents=True, exist_ok=True)
    run_path = args.outputs / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    total = len(prompts) * len(CONDITIONS) * args.samples
    done = 0

    try:
        with run_path.open("w") as output_file:
            for item in prompts:
                for condition in CONDITIONS:
                    prompt = build_prompt(item, condition)
                    for sample in range(1, args.samples + 1):
                        result = query_ollama(args.base_url, args.model, prompt, args.think)
                        record = {
                            "id": item["id"],
                            "class": item.get("class"),
                            "topic": item.get("topic"),
                            "condition": condition,
                            "sample": sample,
                            "model": args.model,
                            "think": args.think,
                            "prompt": prompt,
                            "response": result.get("response", ""),
                            "thinking": result.get("thinking", ""),
                            "eval_count": result.get("eval_count"),
                            "total_duration": result.get("total_duration"),
                        }
                        output_file.write(json.dumps(record) + "\n")
                        output_file.flush()
                        done += 1
                        print(f"[{done}/{total}] {item['id']} {condition} sample {sample}", flush=True)
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach Ollama at {args.base_url}: {error}") from error

    print(run_path)


if __name__ == "__main__":
    main()
