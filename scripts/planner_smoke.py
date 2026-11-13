"""Smoke-test the LLM planner on a few train questions (run on the GPU box)."""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ledgerai.dataset import Dataset  # noqa: E402
from ledgerai.llm import GemmaEngine  # noqa: E402
from ledgerai.pipeline import BaselineRouter  # noqa: E402
from ledgerai.planner import build_prompt, execute_plan, parse_plan  # noqa: E402
from ledgerai.scoring import compare, load_jsonl  # noqa: E402


def main() -> None:
    data = ROOT / "data/raw/train"
    questions = load_jsonl(data / "questions.jsonl")
    answers = {a["id"]: a for a in load_jsonl(data / "answers.jsonl")}
    # Pick 12 questions across three business shapes.
    wanted = {"biz_101": 4, "biz_110": 4, "biz_120": 4}
    picked = []
    for q in questions:
        if wanted.get(q["business_id"], 0) > 0:
            picked.append(q)
            wanted[q["business_id"]] -= 1
        if sum(wanted.values()) == 0:
            break

    print("loading model...", flush=True)
    t0 = time.time()
    engine = GemmaEngine(
        ROOT / "models/gemma-4-12b-it",
        max_model_len=8192,
        gpu_memory_utilization=0.45,
    )
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    router = BaselineRouter()
    datasets: dict[str, Dataset] = {}
    correct = wrong = abstain = 0
    for q in picked:
        if q["business_id"] not in datasets:
            datasets[q["business_id"]] = Dataset.load(data / "data" / q["business_id"])
        ds = datasets[q["business_id"]]
        as_of = date.fromisoformat(q["as_of_date"])
        answer = router.answer(q["question"], as_of, ds)
        source = "router"
        thinking = ""
        if answer.type == "abstain":
            t1 = time.time()
            gen = engine.chat(build_prompt(q["question"], as_of, ds))
            elapsed = time.time() - t1
            plan = parse_plan(gen.text)
            thinking = gen.text[:500].replace("\n", " ")
            if plan:
                answer = execute_plan(plan, as_of, ds)
            print(f"[{elapsed:.1f}s] {q['question'][:70]}", flush=True)
            print(f"  plan={plan} -> {answer.type} {answer.value}", flush=True)
            source = "llm"
        gold = answers[q["id"]]
        verdict = compare(answer.to_json(q["id"]), gold)
        if verdict.status == "correct":
            correct += 1
        elif verdict.status == "wrong":
            wrong += 1
        else:
            abstain += 1
        print(
            f"  {source} {verdict.status}: pred={answer.type} {answer.value} gold={gold['type']} {gold['answer']}",
            flush=True,
        )
    print(f"RESULT correct={correct} wrong={wrong} abstain={abstain}", flush=True)


if __name__ == "__main__":
    main()
