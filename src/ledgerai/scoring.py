"""Answer comparison and penalized accuracy."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .normalize import norm_key


@dataclass
class Verdict:
    status: str  # correct | wrong | abstain
    detail: str = ""


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _string_ok(pred: object, gold: object, aliases: list) -> bool:
    key = norm_key(pred)
    if not key:
        return False
    candidates = [gold] + list(aliases or [])
    for cand in candidates:
        if isinstance(cand, list):
            for item in cand:
                if key == norm_key(item):
                    return True
        elif key == norm_key(cand):
            return True
    return False


def _list_ok(pred: object, gold: object, aliases: list) -> bool:
    if not isinstance(pred, list) or not isinstance(gold, list):
        return False
    if len(pred) != len(gold):
        return False
    alias_lists = aliases if isinstance(aliases, list) else []
    for idx, (p, g) in enumerate(zip(pred, gold)):
        row_aliases = alias_lists[idx] if idx < len(alias_lists) else []
        if isinstance(g, (int, float)) and not isinstance(g, bool):
            pn = _as_number(p)
            if pn is None or abs(pn - float(g)) > 1e-9:
                return False
        elif isinstance(g, list):
            if not _list_ok(p, g, row_aliases):
                return False
        elif not _string_ok(p, g, row_aliases):
            return False
    return True


def _table_ok(pred: object, gold: object) -> bool:
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    gcols = [norm_key(c) for c in gold.get("columns", [])]
    pcols = [norm_key(c) for c in pred.get("columns", [])]
    if gcols and pcols and gcols != pcols:
        # Column names are compared by name; missing names fall back to position.
        if sorted(gcols) != sorted(pcols):
            return False
    grows = gold.get("rows", [])
    prows = pred.get("rows", [])
    if len(grows) != len(prows):
        return False

    def row_key(row):
        return tuple(norm_key(v) if isinstance(v, str) else str(v) for v in row)

    return sorted(row_key(r) for r in grows) == sorted(row_key(r) for r in prows)


def compare(pred: dict, gold: dict) -> Verdict:
    gtype = gold.get("type")
    ptype = pred.get("type")
    if ptype in {"abstain", None} or "answer" not in pred:
        return Verdict("abstain")
    if gtype == "unanswerable":
        return Verdict("correct" if ptype == "unanswerable" else "wrong")
    if ptype == "unanswerable":
        return Verdict("wrong")
    if ptype != gtype:
        return Verdict("wrong", f"type {ptype} != {gtype}")
    p = pred.get("answer")
    g = gold.get("answer")
    aliases = gold.get("alias", [])
    if gtype == "number":
        pn, gn = _as_number(p), _as_number(g)
        ok = pn is not None and gn is not None and abs(pn - gn) <= 1e-6
    elif gtype == "percent":
        pn, gn = _as_number(p), _as_number(g)
        ok = (
            pn is not None
            and gn is not None
            and abs(round(pn, 2) - round(gn, 2)) <= 1e-9
        )
    elif gtype == "string":
        ok = _string_ok(p, g, aliases)
    elif gtype == "list":
        ok = _list_ok(p, g, aliases)
    elif gtype == "date":
        ok = str(p) == str(g)
    elif gtype == "boolean":
        ok = bool(p) == bool(g)
    elif gtype == "table":
        ok = _table_ok(p, g)
    elif gtype == "unanswerable":
        ok = ptype == "unanswerable"
    else:
        ok = p == g
    return Verdict("correct" if ok else "wrong")


def penalized_accuracy(verdicts: list[Verdict]) -> dict:
    n = len(verdicts)
    if n == 0:
        return {"n": 0, "correct": 0, "wrong": 0, "abstain": 0, "pa": 0.0}
    correct = sum(1 for v in verdicts if v.status == "correct")
    wrong = sum(1 for v in verdicts if v.status == "wrong")
    abstain = n - correct - wrong
    answered = correct + wrong
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "answered": answered,
        "coverage": round(answered / n, 4),
        "precision": round(correct / answered, 4) if answered else 0.0,
        "pa": round((correct - wrong) / n, 4),
    }


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
