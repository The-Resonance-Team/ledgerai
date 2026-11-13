"""Rule-based verifier for LLM plans and deterministic answers."""

from __future__ import annotations

import re
from datetime import date

from .dataset import Dataset
from .normalize import norm_key
from .periods import Period, resolve_period


_PERIOD_METRICS = {
    "revenue",
    "order_count",
    "cancelled_count",
    "cancellation_rate",
    "top_products",
    "top_customers",
    "avg_order",
    "product_quantity",
    "quantity_by_unit",
}


def llm_eligible(question: str) -> bool:
    """Skip the LLM on list/name questions; its ranking order is unreliable."""
    t = norm_key(question)
    if any(
        phrase in t
        for phrase in (
            "liet ke",
            "danh sach",
            "la nhung gi",
            "nhung gi",
            "nhung mat hang nao",
            "nhung san pham nao",
        )
    ):
        return False
    if re.search(r"(?:^|\s)top\s*\d", t):
        return False
    if re.search(r"\b\d{1,2}\s+(?:mat hang|san pham|hang|khach|mon|sp)\b", t):
        return False
    if re.search(
        r"(?:mat hang|san pham|hang|khach|mon|sp|tho|nhan vien)\s+nao",
        t,
    ):
        return False
    if re.search(r"(?:^|\s)ai(?:\s|$)", t):
        return False
    return True


def _product_names(ds: Dataset) -> list[str]:
    names: list[str] = []
    if not ds.products.empty and "product" in ds.products.columns:
        names.extend(str(n) for n in ds.products["product"].dropna().unique())
    if not ds.lines.empty and "product" in ds.lines.columns:
        names.extend(str(n) for n in ds.lines["product"].dropna().unique())
    return list(dict.fromkeys(names))


def match_product(question: str, ds: Dataset, min_len: int = 5) -> str | None:
    """Longest product name that appears in the question."""
    q = norm_key(question)
    best: str | None = None
    best_len = 0
    for name in _product_names(ds):
        if norm_key(name) == "phi phuc vu":
            continue
        key = norm_key(name)
        if len(key) >= min_len and key in q and len(key) > best_len:
            best, best_len = str(name), len(key)
    # Alias spellings (full names, nicknames) map back to the canonical name.
    # Full aliases are longer than the question's short form, so accept a
    # long consecutive word run as well ("samsung galaxy a56 5g").
    for alias, canonical in (getattr(ds, "product_aliases", None) or {}).items():
        if norm_key(canonical) == "phi phuc vu":
            continue
        if len(alias) >= min_len and alias in q:
            score = len(alias)
        else:
            run_words, run_len = _word_run(alias, q)
            # Runs shorter than this are noise ("size m", "op lung").
            if run_words < 2 or run_len < 8:
                continue
            score = run_len
        if score > best_len:
            best, best_len = str(canonical), score
    return best


def _word_run(a: str, b: str) -> tuple[int, int]:
    """Longest consecutive word run shared by a and b. Returns (words, chars)."""
    words, length, _ = _word_run_next(a, b)
    return words, length


def _word_run_next(a: str, b: str) -> tuple[int, int, str | None]:
    """Like _word_run, plus the alias word right after a prefix run."""
    wa, wb = a.split(), b.split()
    if not wa or not wb:
        return 0, 0, None
    positions: dict[str, list[int]] = {}
    for j, w in enumerate(wb):
        positions.setdefault(w, []).append(j)
    best_words, best_len, best_next = 0, 0, None
    for i, w in enumerate(wa):
        for j in positions.get(w, []):
            k = 0
            while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                k += 1
            if k > best_words:
                best_words = k
                best_len = len(" ".join(wa[i : i + k]))
                best_next = wa[i + k] if i == 0 and i + k < len(wa) else None
    return best_words, best_len, best_next


def match_unit(question: str, ds: Dataset, min_len: int = 2) -> str | None:
    if ds.lines.empty or "unit" not in ds.lines.columns:
        lines_empty = True
    else:
        lines_empty = False
    q = norm_key(question)
    # "bao nhiêu" is a question word, not the unit "bao" (bag).
    q = re.sub(r"\b(?:bao\s*nhieu|bao\s*nhiu|bnhiu|bao\s*tien)\b", " ", q)
    best: str | None = None
    best_len = 0
    generic = {
        "don",
        "don hang",
        "bill",
        "hoa don",
        "lan",
        "luot",
        "phieu",
        "ky",
        "thang",
        "ngay",
    }
    seen: set[str] = set()
    if not lines_empty:
        for name in ds.lines["unit"].dropna().unique():
            key = norm_key(name)
            if not key or len(key) < min_len or key in generic or key in seen:
                continue
            seen.add(key)
            if (
                re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", q)
                and len(key) > best_len
            ):
                best, best_len = str(name), len(key)
    # Product masters carry the unit when order lines do not (dev_007).
    if not ds.products.empty and "unit" in ds.products.columns:
        for name in ds.products["unit"].dropna().unique():
            key = norm_key(name)
            if not key or len(key) < min_len or key in generic or key in seen:
                continue
            seen.add(key)
            if (
                re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", q)
                and len(key) > best_len
            ):
                best, best_len = str(name), len(key)
    return best


def match_category(question: str, ds: Dataset, min_len: int = 3) -> str | None:
    names: list[str] = []
    if not ds.products.empty and "category" in ds.products.columns:
        names.extend(str(n) for n in ds.products["category"].dropna().unique())
    if not ds.lines.empty and "category" in ds.lines.columns:
        names.extend(str(n) for n in ds.lines["category"].dropna().unique())
    q = norm_key(question)
    best: str | None = None
    best_len = 0
    for name in dict.fromkeys(names):
        key = norm_key(name)
        if len(key) >= min_len and key in q and len(key) > best_len:
            best, best_len = name, len(key)
    return best


def _period(plan: dict, as_of: date) -> Period | None:
    start = plan.get("period_start")
    end = plan.get("period_end")
    if start and end:
        try:
            return Period(date.fromisoformat(start), date.fromisoformat(end), "plan")
        except ValueError:
            return None
    expression = plan.get("period") or plan.get("time")
    return resolve_period(expression, as_of) if expression else None


def verify_plan(
    plan: dict, question: str, as_of: date, ds: Dataset
) -> tuple[bool, str, dict]:
    """Return (ok, reason, adjusted_plan)."""
    plan = dict(plan)
    plan.setdefault("filters", {})
    filters = plan["filters"] if isinstance(plan.get("filters"), dict) else {}
    plan["filters"] = filters
    metric = str(plan.get("metric") or "abstain").lower()
    q = norm_key(question)
    period = _period(plan, as_of)

    if metric == "abstain":
        return False, "llm_abstain", plan

    # Filters the semantic layer cannot express.
    if ("ke don" in q or "khong ke don" in q) and not filters.get("prescription"):
        return False, "unsupported_prescription_filter", plan
    if ("ca sang" in q or "ca chieu" in q) and not filters.get("shift"):
        return False, "unsupported_shift_filter", plan

    # Period is required for period metrics.
    if metric in _PERIOD_METRICS and period is None:
        return False, "missing_period", plan

    # A period wholly outside the data is unanswerable, not 0.
    if period is not None and len(ds.orders):
        dates = ds.orders["date"].dropna()
        if len(dates):
            data_min, data_max = dates.min(), dates.max()
            if period.end < data_min or period.start > data_max:
                return False, "outside_data_range", plan

    # Auto-add product/category filters the LLM missed.
    if metric in {"revenue", "top_products", "product_quantity"}:
        product = match_product(question, ds)
        if product:
            # The question names a known product; trust the exact catalog name
            # over the model's partial spelling.
            filters["product"] = product
    if metric in {"revenue", "top_products"} and not filters.get("category"):
        category = match_category(question, ds)
        if category:
            filters["category"] = category
    if metric == "quantity_by_unit":
        if not filters.get("unit"):
            unit = match_unit(question, ds)
            if unit:
                filters["unit"] = unit
        if "size m" in q:
            filters["size"] = "m"

    # Question type consistency.
    asks_day = "ngay nao" in q or "ngay may" in q
    asks_month = "thang nao" in q
    if (asks_day or asks_month) and metric in {"revenue", "order_count"}:
        return False, "type_mismatch_date_question", plan
    asks_percent = any(k in q for k in ("phan tram", "ti le", "%"))
    if asks_percent and metric not in {"cancellation_rate"}:
        return False, "type_mismatch_percent_question", plan
    asks_list = any(k in q for k in ("liet ke", "top ", "danh sach"))
    if asks_list and metric not in {"top_products", "top_customers"}:
        return False, "type_mismatch_list_question", plan

    return True, "ok", plan
