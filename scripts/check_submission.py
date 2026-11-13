"""Validate a submission.jsonl against the DataMate format rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VALID_TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def check_line(row: dict) -> list[str]:
    errors = []
    qid = row.get("id")
    if not isinstance(qid, str) or not qid:
        errors.append("missing or invalid id")
    rtype = row.get("type")
    if rtype not in VALID_TYPES:
        errors.append(f"invalid type {rtype!r}")
        return errors
    if rtype in {"abstain", "unanswerable"}:
        return errors
    if "answer" not in row:
        errors.append("missing answer")
        return errors
    answer = row["answer"]
    if rtype == "number":
        if not isinstance(answer, (int, float)) or isinstance(answer, bool):
            errors.append("number answer must be a JSON number")
    elif rtype == "percent":
        if not isinstance(answer, (int, float)) or isinstance(answer, bool):
            errors.append("percent answer must be a JSON number")
    elif rtype == "string":
        if not isinstance(answer, str):
            errors.append("string answer must be a string")
    elif rtype == "date":
        if not isinstance(answer, str) or not (
            DATE_RE.match(answer) or MONTH_RE.match(answer)
        ):
            errors.append("date answer must be YYYY-MM-DD or YYYY-MM")
    elif rtype == "boolean":
        if not isinstance(answer, bool):
            errors.append("boolean answer must be true/false")
    elif rtype == "list":
        if not isinstance(answer, list):
            errors.append("list answer must be a JSON array")
    elif rtype == "table":
        if (
            not isinstance(answer, dict)
            or "columns" not in answer
            or "rows" not in answer
        ):
            errors.append("table answer must have columns and rows")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in open(args.questions, encoding="utf-8")
        if line.strip()
    ]
    expected = {q["id"] for q in questions}
    rows = [
        json.loads(line)
        for line in open(args.submission, encoding="utf-8")
        if line.strip()
    ]
    seen = set()
    errors = []
    for row in rows:
        seen.add(row.get("id"))
        for error in check_line(row):
            errors.append(f"{row.get('id')}: {error}")
    missing = expected - seen
    extra = seen - expected
    print(
        f"rows={len(rows)} expected={len(expected)} missing={len(missing)} extra={len(extra)} errors={len(errors)}"
    )
    if missing:
        print("missing ids:", sorted(missing)[:10])
    if extra:
        print("extra ids:", sorted(extra)[:10])
    for error in errors[:30]:
        print("ERROR", error)
    raise SystemExit(1 if errors or missing or extra else 0)


if __name__ == "__main__":
    main()
