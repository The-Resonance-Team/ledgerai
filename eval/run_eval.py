"""Run the baseline pipeline on the business-level holdout and report PA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from holdout import split_businesses  # noqa: E402
from ledgerai.dataset import Dataset  # noqa: E402
from ledgerai.normalize import strip_diacritics  # noqa: E402
from ledgerai.pipeline import BaselineRouter  # noqa: E402
from ledgerai.scoring import Verdict, compare, load_jsonl, penalized_accuracy  # noqa: E402


def _family(question: str) -> str:
    t = strip_diacritics(question).lower()
    if "ban chay" in t or "ban nhieu" in t or "hot nhat" in t:
        return "top_product"
    if "khach" in t and ("nhieu nhat" in t or "nhieu tien" in t):
        return "top_customer"
    if "huy" in t:
        return "cancel"
    if "tang truong" in t or "phan tram thay doi" in t:
        return "growth"
    if "trung binh" in t or "binh quan" in t:
        return "average"
    if "no" in t.split() or "chua tra" in t:
        return "debt"
    if "ton kho" in t or "ton" in t.split():
        return "inventory"
    if "nhap" in t:
        return "purchase"
    if "don" in t and ("bao nhieu" in t or "so don" in t):
        return "order_count"
    if "doanh thu" in t or "dthu" in t or "thu ve" in t:
        return "revenue"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/train"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "eval/run"))
    args = parser.parse_args()

    data_dir = Path(args.data)
    _, holdout = split_businesses(data_dir / "data")
    questions = load_jsonl(data_dir / "questions.jsonl")
    answers = {a["id"]: a for a in load_jsonl(data_dir / "answers.jsonl")}
    holdout_set = set(holdout)
    rows = [q for q in questions if q["business_id"] in holdout_set]
    if args.limit:
        rows = rows[: args.limit]

    router = BaselineRouter()
    datasets: dict[str, Dataset] = {}
    verdicts: list[Verdict] = []
    preds = []
    by_type: dict[str, list[Verdict]] = defaultdict(list)
    by_family: dict[str, list[Verdict]] = defaultdict(list)
    plan_counts = Counter()
    start = time.time()

    for q in rows:
        biz = q["business_id"]
        if biz not in datasets:
            datasets[biz] = Dataset.load(data_dir / "data" / biz)
        as_of = date.fromisoformat(q["as_of_date"])
        answer = router.answer(q["question"], as_of, datasets[biz])
        gold = answers[q["id"]]
        verdict = compare(answer.to_json(q["id"]), gold)
        verdicts.append(verdict)
        record = answer.to_json(q["id"])
        record["plan"] = answer.plan
        preds.append(record)
        by_type[gold["type"]].append(verdict)
        by_family[_family(q["question"])].append(verdict)
        plan_counts[answer.plan] += 1

    report = {
        "holdout": holdout,
        "questions": len(rows),
        "seconds": round(time.time() - start, 1),
        "overall": penalized_accuracy(verdicts),
        "by_type": {k: penalized_accuracy(v) for k, v in sorted(by_type.items())},
        "by_family": {k: penalized_accuracy(v) for k, v in sorted(by_family.items())},
        "plans": plan_counts.most_common(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out / "preds.jsonl", "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
