"""Evaluate the hybrid router + LLM planner on the business-level holdout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from holdout import split_businesses  # noqa: E402
from ledgerai.dataset import Dataset  # noqa: E402
from ledgerai.llm import GemmaEngine  # noqa: E402
from ledgerai.pipeline import BaselineRouter  # noqa: E402
from ledgerai.planner import build_prompt, execute_plan, parse_plan  # noqa: E402
from ledgerai.verifier import llm_eligible, verify_plan  # noqa: E402
from ledgerai.scoring import compare, load_jsonl, penalized_accuracy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/train"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default=str(ROOT / "eval/run_llm"))
    parser.add_argument("--model", default=str(ROOT / "models/gemma-4-12b-it"))
    parser.add_argument("--gpu-mem", type=float, default=0.45)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    _, holdout = split_businesses(data_dir / "data")
    questions = load_jsonl(data_dir / "questions.jsonl")
    answers = {a["id"]: a for a in load_jsonl(data_dir / "answers.jsonl")}
    rows = [q for q in questions if q["business_id"] in set(holdout)]
    if args.limit:
        rows = rows[: args.limit]

    print(f"loading model from {args.model}", flush=True)
    engine = GemmaEngine(
        args.model,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_mem,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds_path = out / "preds.jsonl"
    # Resume: keep every completed answer so a dropped session does not lose work.
    done: dict[str, dict] = {}
    if args.resume and preds_path.exists():
        for row in load_jsonl(preds_path):
            done[row["id"]] = row
        print(f"resume: {len(done)} answers already present", flush=True)
    if not args.resume:
        preds_path.write_text("", encoding="utf-8")

    router = BaselineRouter()
    datasets: dict[str, Dataset] = {}
    verdicts = []
    by_type = defaultdict(list)
    preds = []
    start = time.time()
    preds_file = open(preds_path, "a", encoding="utf-8")

    try:
        for idx, q in enumerate(rows, 1):
            if q["id"] in done:
                pred = done[q["id"]]
                gold = answers[q["id"]]
                verdict = compare(pred, gold)
                verdicts.append(verdict)
                by_type[gold["type"]].append(verdict)
                preds.append(pred)
                continue
            biz = q["business_id"]
            if biz not in datasets:
                datasets[biz] = Dataset.load(data_dir / "data" / biz)
            ds = datasets[biz]
            as_of = date.fromisoformat(q["as_of_date"])
            answer = router.answer(q["question"], as_of, ds)
            if answer.type == "abstain" and llm_eligible(q["question"]):
                gen = engine.chat(build_prompt(q["question"], as_of, ds))
                plan = parse_plan(gen.text)
                if plan is not None:
                    ok, reason, plan = verify_plan(plan, q["question"], as_of, ds)
                    if ok:
                        answer = execute_plan(plan, as_of, ds)
                    elif reason == "outside_data_range":
                        answer = __import__(
                            "ledgerai.pipeline", fromlist=["Answer"]
                        ).Answer("unanswerable", plan=reason)
            gold = answers[q["id"]]
            verdict = compare(answer.to_json(q["id"]), gold)
            verdicts.append(verdict)
            by_type[gold["type"]].append(verdict)
            pred = answer.to_json(q["id"])
            preds.append(pred)
            preds_file.write(json.dumps(pred, ensure_ascii=False) + "\n")
            preds_file.flush()
            print(
                f"{idx}/{len(rows)} {verdict.status} {q['business_id']} "
                f"{answer.type} {answer.value} | gold {gold['type']} {gold['answer']}",
                flush=True,
            )
            if idx % 25 == 0:
                partial = penalized_accuracy(verdicts)
                print(
                    f"PARTIAL {idx}/{len(rows)} pa={partial['pa']} "
                    f"cov={partial['coverage']} prec={partial['precision']}",
                    flush=True,
                )
    finally:
        preds_file.close()

    report = {
        "questions": len(rows),
        "seconds": round(time.time() - start, 1),
        "overall": penalized_accuracy(verdicts),
        "by_type": {k: penalized_accuracy(v) for k, v in sorted(by_type.items())},
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("REPORT", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
