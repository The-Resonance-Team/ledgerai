"""Offline entry point: read /data, write /result/submission.jsonl and logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ledgerai.dataset import Dataset  # noqa: E402
from ledgerai.llm import GemmaEngine  # noqa: E402
from ledgerai.pipeline import Answer, BaselineRouter  # noqa: E402
from ledgerai.planner import build_prompt, execute_plan, parse_plan  # noqa: E402
from ledgerai.verifier import llm_eligible, verify_plan  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_log(log_dir: Path, record: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{record['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data")
    parser.add_argument("--result", default="/result")
    parser.add_argument("--model", default="/code/models/gemma-4-12b-it")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data)
    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir / "logs"
    questions = load_jsonl(data_dir / "questions.jsonl")
    if args.limit:
        questions = questions[: args.limit]

    router = BaselineRouter()
    engine = None
    if args.use_llm and GemmaEngine.available(args.model):
        try:
            engine = GemmaEngine(
                args.model,
                max_model_len=int(os.environ.get("LEDGERAI_MAX_LEN", "8192")),
                gpu_memory_utilization=float(
                    os.environ.get("LEDGERAI_GPU_MEM", "0.85")
                ),
            )
        except Exception as exc:  # keep the deterministic path alive
            print(f"LLM unavailable, deterministic only: {exc}", flush=True)
            engine = None

    datasets: dict[str, Dataset] = {}
    outputs: list[dict] = []

    budget_deadline = time.time() + float(os.environ.get("LEDGERAI_BUDGET_S", "18000"))

    submission = open(result_dir / "submission.jsonl", "w", encoding="utf-8")
    try:
        for question in questions:
            start = time.time()
            qid = question["id"]
            business_id = question["business_id"]
            thinking = None
            code = None
            inputs = None
            output = None
            try:
                if engine is not None and time.time() > budget_deadline:
                    print("time budget reached; deterministic mode", flush=True)
                    engine = None
                if business_id not in datasets:
                    datasets[business_id] = Dataset.load(
                        data_dir / "data" / business_id
                    )
                ds = datasets[business_id]
                as_of = date.fromisoformat(question["as_of_date"])
                answer = router.answer(question["question"], as_of, ds)
                if (
                    answer.type == "abstain"
                    and engine is not None
                    and llm_eligible(question["question"])
                ):
                    messages = build_prompt(question["question"], as_of, ds)
                    generation = engine.chat(messages)
                    thinking = generation.text
                    plan = parse_plan(generation.text)
                    if plan is not None:
                        ok, reason, plan = verify_plan(
                            plan, question["question"], as_of, ds
                        )
                        if not ok:
                            answer = Answer(
                                "unanswerable"
                                if reason == "outside_data_range"
                                else "abstain",
                                plan=reason,
                            )
                        else:
                            answer = execute_plan(plan, as_of, ds)
                        code = "src/ledgerai/planner.py::execute_plan"
                        inputs = plan
                        output = answer.value
            except Exception as exc:  # never fail the whole run for one question
                answer = __import__("ledgerai.pipeline", fromlist=["Answer"]).Answer(
                    "abstain", plan="error", reason=str(exc)
                )

            if code is None:
                # Deterministic route: log the actual function and arguments.
                code = "src/ledgerai/pipeline.py::BaselineRouter.answer"
                inputs = {
                    "question": question["question"],
                    "business_id": business_id,
                    "as_of_date": question["as_of_date"],
                }
                output = answer.value

            record = answer.to_json(qid)
            submission.write(json.dumps(record, ensure_ascii=False) + "\n")
            submission.flush()
            outputs.append(record)
            write_log(
                log_dir,
                {
                    "id": qid,
                    "answer": {"type": answer.type, "answer": answer.value},
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "thinking": thinking,
                    "code": code,
                    "inputs": inputs,
                    "output": output,
                    "plan": answer.plan,
                },
            )
    finally:
        submission.close()

    print(f"wrote {len(outputs)} answers to {result_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
