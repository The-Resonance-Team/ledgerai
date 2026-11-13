"""Conservative baseline router: question -> deterministic metric -> answer.

The baseline answers only narrow, unambiguous families. Everything else
abstains, because a wrong answer costs -1 while abstaining costs 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .dataset import Dataset
from .normalize import norm_key, parse_number, strip_diacritics
from .periods import (
    Period,
    _month_end,
    extract_month_refs,
    is_tet_period,
    previous_period,
    resolve_period,
    tet_period,
)


@dataclass
class Answer:
    type: str
    value: object = None
    confidence: float = 0.0
    plan: str = ""
    reason: str = ""

    def to_json(self, qid: str) -> dict:
        out = {"id": qid, "type": self.type}
        if self.type not in {"abstain", "unanswerable"}:
            out["answer"] = self.value
        return out


def _has(text: str, *keys: str) -> bool:
    return any(k in text for k in keys)


def _words(text: str, *words: str) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text)
        for word in words
    )


def _has_period(text: str) -> bool:
    return bool(
        re.search(
            r"hom nay|hom qua|hnay|hqua|hwa|bua qua|bua hom qua|bua nay|hom ni|"
            r"thang|thang ni|th nay|th truoc|th roi|tuan|quy|ngay|nam|nam ni|gan nhat|dau nam|tet|"
            r"\d{1,2}[-/]\d{4}|\d{1,2}/\d{2}(?!\d)|\bth\s*\d{1,2}\b",
            text,
        )
    )


def _is_yoy(raw: str, t: str, period: Period | None) -> bool:
    """True when the comparison base is the same period last year."""
    if any(
        phrase in t
        for phrase in (
            "cung ky nam",
            "cung thang nam",
            "nam ngoai",
            "nam roi",
            "nam truoc",
        )
    ):
        return True
    if period is None:
        return False
    refs: list[tuple[int, int]] = []
    for match in re.finditer(r"(?:th|thang)?\s*(\d{1,2})\s*[/\-]\s*(\d{2,4})", raw):
        month, year = int(match.group(1)), int(match.group(2))
        if year < 100:
            year += 2000
        if 1 <= month <= 12:
            refs.append((year, month))
    return (
        len(refs) >= 2
        and len({month for _, month in refs}) == 1
        and len({year for year, _ in refs}) >= 2
    )


def _match_service(question: str, ds: Dataset) -> str | None:
    """Longest service description from an orders table that appears in the question."""
    if ds.orders.empty or "service" not in ds.orders.columns:
        return None
    q = norm_key(question)
    variants: list[str] = []
    if "dau lap" in q:
        variants.append("nhot lap")
    if "dau may" in q or "nhot may" in q:
        variants.append("nhot may")
    services = [norm_key(name) for name in ds.orders["service"].dropna().unique()]
    for variant in variants:
        if any(variant in key for key in services):
            return variant
    best: str | None = None
    best_len = 0
    for name in ds.orders["service"].dropna().unique():
        key = norm_key(name)
        if len(key) >= 4 and key in q and len(key) > best_len:
            best, best_len = str(name), len(key)
    return best


def _period_growth(
    ds: Dataset, period: Period | None, prior: Period | None, metric: str
) -> float | None:
    """Percent change between two explicit periods (e.g., Tết YoY)."""
    if period is None or prior is None:
        return None
    if metric == "orders":
        current, old = ds.order_count(period), ds.order_count(prior)
    else:
        current, old = ds.revenue(period), ds.revenue(prior)
    if not old:
        return None
    return round(100.0 * (current - old) / old, 2)


def _count_k(text: str) -> int:
    m = re.search(
        r"(?:top|liet ke|lay|cho|tim)\s*(?<![0-9])(\d{1,2})(?![0-9/])\s*"
        r"(?:mat hang|mawtj hang|món|mon|san pham|hang|khach)?",
        text,
    )
    if not m:
        m = re.search(
            r"(?<![0-9])(\d{1,2})(?![0-9/])\s*"
            r"(?:mat hang|mawtj hang|món|mon|san pham|hangf|hang|khach)",
            text,
        )
    if m:
        return max(1, min(50, int(m.group(1))))
    return 1


def _round_money(value: float) -> int:
    return int(round(value))


_AMOUNT = r"(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(k|nghin|nghìn|tr|trieu|triệu)?"


def _amount_value(text: str, start: int) -> float | None:
    m = re.match(r"\s*" + _AMOUNT, text[start:])
    if not m:
        return None
    value = parse_number(m.group(0))
    if value is None or value < 1000:
        return None
    return value


def _money_thresholds(raw: str) -> tuple[float | None, float | None]:
    """Read revenue thresholds such as 'từ 55.000 trở lên' or 'dưới 100.000'."""
    lower: float | None = None
    upper: float | None = None

    m = re.search(
        r"(?:tu|dat|it nhat)\s+(?:muc\s*)?(?:dthu\s*|doanh thu\s*)?(?:tu\s*)?",
        raw,
    )
    if m:
        value = _amount_value(raw, m.end())
        if value is not None:
            lower = value
    if lower is None:
        m = re.search(r"(?:≥|>=)\s*(?:muc\s*)?", raw)
        if m:
            lower = _amount_value(raw, m.end())
    if lower is None:
        m = re.search(r"(?:tren|hon|qua)\s+(?:muc\s*)?", raw)
        if m:
            value = _amount_value(raw, m.end())
            if value is not None:
                lower = value + 1  # "trên N" does not include N (rulebook §2)
    if lower is None:
        m = re.search(r"(?:duoi|nho hon)\s+(?:muc\s*)?", raw)
        if m:
            value = _amount_value(raw, m.end())
            if value is not None:
                upper = value - 1
    return lower, upper


def _match_inventory(question: str, ds: Dataset) -> str | None:
    if ds.inventory is None or ds.inventory.empty:
        return None
    q = norm_key(question)
    best: str | None = None
    best_len = 0
    for name in ds.inventory["product"].dropna().unique():
        key = norm_key(name)
        if len(key) >= 4 and key in q and len(key) > best_len:
            best, best_len = str(name), len(key)
    return best


# Filters the baseline cannot apply. A revenue question with one of these
# keywords must abstain instead of returning the unfiltered total.
_FILTER_WORDS = (
    "mat hang",
    "san pham",
    "hang hoa",
    "thuoc",
    "hoat chat",
    "nhom hang",
    "nhom ",
    "khach hang",
    "khach ",
    "nhan vien",
    "hinh thuc",
    "kenh",
    "shopee",
    "tiktok",
    "cod",
    "chuyen khoan",
    "tien mat",
    "ca sang",
    "ca chieu",
    "the loai",
    "loai hang",
    "nha cung cap",
    "zalo",
    "tai quay",
    "quay",
    "khong ke don",
    "ke don",
    "hoat chat",
    "thanh toan",
    "cong no",
    "tra truoc",
    "tra sau",
)

_BOOLEAN_WORDS = (
    "co tang lien tuc",
    "tang lien tuc",
    "co luon cao hon",
    "luon cao hon",
    "co phai",
    "dung khong",
    "phai khong",
    "co dung",
    "tang hay giam",
    "cao hon khong",
    "thap hon khong",
)


class BaselineRouter:
    # A zero in these plans almost always means a failed join or a missing
    # column, not a real zero. Abstaining scores 0; a wrong zero scores -1.
    _ZERO_ABSTAIN_PLANS = {
        "revenue_total",
        "revenue_filtered",
        "revenue_product",
        "revenue_category",
        "revenue_customer_group",
        "revenue_monthly_list",
        "revenue_monthly_sequence",
        "revenue_employee",
        "revenue_payment",
        "revenue_shift",
        "debt_total",
        "debt_customers",
        "debt_collection",
        "profit",
        "purchase_total",
        "avg_order_value",
        "paid_total",
        "shift_revenue",
    }

    def answer(self, question: str, as_of: date, ds: Dataset) -> Answer:
        result = self._answer(question, as_of, ds)
        if (
            result.type == "number"
            and result.value == 0
            and result.plan in self._ZERO_ABSTAIN_PLANS
        ):
            return Answer("abstain", plan=f"{result.plan}_zero_guard")
        return result

    def _answer(self, question: str, as_of: date, ds: Dataset) -> Answer:
        raw = strip_diacritics(str(question or "")).lower()
        # Frequent misspellings in the question set.
        raw = re.sub(r"\b(toonf|tonf)\b", "ton", raw)
        raw = re.sub(r"\b(tieenf|tienf)\b", "tien", raw)
        raw = re.sub(r"\b(treen)\b", "tren", raw)
        raw = re.sub(r"\b(thangs)\b", "thang", raw)
        raw = re.sub(r"\b(quys)\b", "quy", raw)
        raw = re.sub(r"\b(nhaats|nhut|nhuts)\b", "nhat", raw)
        raw = re.sub(r"\b(quaan)\b", "quan", raw)
        raw = re.sub(r"\b(mooix)\b", "moi", raw)
        raw = re.sub(r"\b(leen)\b", "len", raw)
        raw = re.sub(r"\b(sangs)\b", "sang", raw)
        raw = re.sub(r"\b(ddown)\b", "don", raw)
        raw = re.sub(r"\b(nhoms)\b", "nhom", raw)
        raw = re.sub(r"\b(khachs)\b", "khach", raw)
        raw = re.sub(r"\b(huyr|huyf)\b", "huy", raw)
        raw = re.sub(r"\b(roa)\b", "ra", raw)
        t = norm_key(raw)
        period = resolve_period(raw, as_of) if _has_period(raw) else None
        k = _count_k(t)
        if (
            period is not None
            and not _words(t, "ton")
            and not _has(t, "trong kho", "con lai")
        ):
            date_range = ds.date_range()
            if date_range and (
                period.end < date_range[0] or period.start > date_range[1]
            ):
                return Answer("unanswerable", plan="outside_data_range")

        # Boolean revenue trend over consecutive months.
        if _has(t, "tang lien tuc", "luon cao hon", "co tang lien tuc") and _has(
            t, "doanh thu", "dthu"
        ):
            refs = extract_month_refs(raw, as_of)
            if len(refs) >= 2:
                (y, m), (end_y, end_m) = refs[0], refs[-1]
                months = []
                while (y, m) <= (end_y, end_m) and len(months) <= 24:
                    months.append((y, m))
                    m += 1
                    if m == 13:
                        m = 1
                        y += 1
                values = [
                    ds.revenue(resolve_period(f"thang {mo}/{yr}", as_of))
                    for yr, mo in months
                ]
                increasing = all(
                    values[i + 1] > values[i] for i in range(len(values) - 1)
                )
                return Answer(
                    "boolean",
                    bool(increasing),
                    confidence=0.8,
                    plan="revenue_trend_boolean",
                )

        # Boolean trend questions are not handled by the baseline.
        growth_language = _has(t, *_BOOLEAN_WORDS) or _has(
            t,
            "len hay xuong",
            "cao hay thap",
            "len xuong",
            "tang dam",
            "hon kem",
            "cao hon",
            "thap hon",
        )
        if growth_language and not _has(t, "ca sang", "ca chieu"):
            asks_percent = _has(raw, "%", "phan tram") or bool(
                re.search(r"(?:bao nhieu|bao nhiu|bnhiu|may|mi|bn|nhiu|nhieu)\s*%", raw)
            )
            if asks_percent and period is not None:
                if _has(t, "so don", "don hang", "so bill", "so hoa don"):
                    metric = "orders"
                elif _has(t, "doanh thu", "dthu", "tien ban", "tien hang", "thu ve"):
                    metric = "revenue"
                else:
                    return Answer("abstain", plan="boolean_trend")
                previous = previous_period(period)
                bounds = ds.date_range()
                if bounds and previous.end < bounds[0]:
                    return Answer("unanswerable", plan="growth_no_prior_period")
                if _is_yoy(raw, t, period):
                    if is_tet_period(period):
                        # Same lunar window last year; a calendar shift
                        # would miss Tết 2025 (verified exact on train).
                        prior = tet_period(period.start.year - 1)
                        value = _period_growth(ds, period, prior, metric)
                    else:
                        value = ds.growth_yoy(period, metric)
                    plan = "revenue_growth_yoy"
                else:
                    value = ds.growth(period, metric)
                    plan = "revenue_growth"
                if value is not None:
                    return Answer("percent", value, confidence=0.75, plan=plan)
            return Answer("abstain", plan="boolean_trend")

        # Channel revenue / share (Zalo OA, at the counter, phone).
        if period is not None and _has(t, "zalo", "tai quay", "dien thoai"):
            found: list[tuple[int, str]] = []
            for needle, value in (
                ("zalo", "zalo"),
                ("tai quay", "store"),
                ("dien thoai", "phone"),
            ):
                pos = t.find(needle)
                if pos >= 0:
                    found.append((pos, value))
            found.sort()
            if len(found) >= 2 and (_has(t, " va ", " voi ") or "," in str(question)):
                values = [ds.revenue_channel(period, value) for _, value in found]
                if all(value is not None for value in values):
                    return Answer(
                        "list",
                        [_round_money(value) for value in values],
                        confidence=0.75,
                        plan="channel_revenue_list",
                    )
            channel = found[0][1] if found else "zalo"
            if _has(t, "ti trong", "chiem"):
                share = ds.channel_share(period, channel)
                if share is not None:
                    return Answer(
                        "percent",
                        share,
                        confidence=0.8,
                        plan="channel_revenue_share",
                    )
            if _has(t, "doanh thu", "dthu", "tien", "ban"):
                value = ds.revenue_channel(period, channel)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="channel_revenue",
                    )

        # Category share of revenue, e.g. "nhóm điện chiếm bao nhiêu %".
        # Customer tiers ("nhóm sỉ/lẻ/thân thiết" + khách) belong to the
        # customer-group branch below, not to product categories.
        if (
            _has(t, "chiem", "ti trong", "dong gop")
            and period is not None
            and not (
                _has(t, "khach", "khach hang")
                and _words(t, "nhom si", "nhom le", "nhom than thiet")
            )
        ):
            from .verifier import match_category

            category = match_category(question, ds)
            if category:
                share = ds.revenue_share(category, period)
                if share is not None:
                    return Answer(
                        "percent",
                        share,
                        confidence=0.8,
                        plan="category_revenue_share",
                    )

        # Profit and expenses.
        if period is not None and _has(
            t,
            "lai gop",
            "lai rong",
            "loi nhuan",
            "loi that",
            "lai that",
            "loi nhieu",
            "lai nhieu",
        ):
            if _has(t, "san pham", "mat hang", "mon", "hang nao", "sp nao") and _has(
                t, "nhat", "cao nhat", "nhieu nhat"
            ):
                top = ds.top_products_profit(period, k=k)
                if top:
                    if k > 1:
                        return Answer(
                            "list",
                            [name for name, _ in top],
                            confidence=0.75,
                            plan="top_products_profit",
                        )
                    return Answer(
                        "string",
                        top[0][0],
                        confidence=0.75,
                        plan="top_product_profit",
                    )
            if _has(
                t, "lai rong", "loi that", "lai that", "tru chi phi", "trừ chi phí"
            ):
                value = ds.net_profit(period)
            else:
                value = ds.gross_profit(period)
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.75,
                    plan="profit",
                )
        if (
            period is not None
            and _has(t, "chi phi", "chi phi")
            and _has(t, "bao nhieu", "bnhiu", "tong", "het")
            and not _has(t, "lai", "loi")
        ):
            value = ds.expense_total(period)
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="expense_total",
                )

        # Payment-method revenue and share. Ledgers use Vietnamese labels
        # ("Chuyển khoản") or English codes ("transfer"), so each method
        # carries needles for both vocabularies.
        payment_methods: list[tuple[str, ...]] = []
        for needles in (
            ("chuyen khoan", "transfer", "ck"),
            ("tien mat", "tien tuoi", "cash", "tm"),
            ("vi dien tu", "vi dien", "ewallet", "momo", "zalo"),
            ("the tin dung", "the ngan hang", "the", "card"),
            ("cod", "thu ho"),
            ("ghi no", "debt", "cong no"),
        ):
            # Word boundaries for every needle: "thu ho" is a substring
            # of "doanh thu hom nay", and "the" of many unrelated words.
            if any(_words(t, needle) for needle in needles):
                payment_methods.append(needles)
        payment_keyword = payment_methods[0][0] if payment_methods else None
        if (
            len(payment_methods) >= 2
            and period is not None
            and _has(t, "moi loai", "tung loai", "tung hinh thuc", "moi hinh thuc")
        ):
            # The reference lists values in the order the methods appear in
            # the question, not in a canonical method order.
            def first_pos(needles: tuple[str, ...]) -> int:
                positions = [
                    match.start()
                    for needle in needles
                    for match in re.finditer(
                        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", t
                    )
                ]
                return min(positions) if positions else 10**9

            values = []
            for needles in sorted(payment_methods, key=first_pos):
                value = ds.revenue_payment(period, needles[0], synonyms=needles)
                if value is None:
                    break
                values.append(_round_money(value))
            else:
                return Answer(
                    "list",
                    values,
                    confidence=0.75,
                    plan="payment_revenue_list",
                )
            return Answer("abstain", plan="payment_revenue_list")
        if payment_keyword and period is not None:
            if _has(t, "ti trong", "chiem"):
                share = ds.payment_share(
                    period, payment_keyword, synonyms=payment_methods[0]
                )
                if share is not None:
                    return Answer(
                        "percent",
                        share,
                        confidence=0.8,
                        plan="payment_share",
                    )
            if _has(t, "doanh thu", "dthu", "tien", "ban"):
                value = ds.revenue_payment(
                    period, payment_keyword, synonyms=payment_methods[0]
                )
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="payment_revenue",
                    )
                # A payment-filtered question with no payment data must not
                # fall through to unrelated count branches.
                return Answer("abstain", plan="payment_no_data")

        # Discounts given to customers.
        if (
            period is not None
            and _has(t, "giam gia", "voucher", "chiet khau", "giam cho khach")
            and _has(t, "bao nhieu", "bnhiu", "tong", "het")
        ):
            value = ds.discount_total(period)
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="discount_total",
                )

        # Debt summaries.
        debt_as_of = period.end if period is not None else as_of
        if _words(t, "no") or _has(t, "cong no"):
            if _has(t, "kem so tien", "so tien cua tung", "bang 3 khach"):
                return Answer("abstain", plan="debt_detail_table")
            if _has(t, "thu hoi", "thu duoc", "tra no", "thu tien no", "nop"):
                return Answer("abstain", plan="debt_collection")
            if _has(t, "tuoi no", "tren 15") or re.search(
                r"0\s*[–-]\s*7|8\s*[–-]\s*15", raw
            ):
                buckets = ds.debt_age_buckets(debt_as_of)
                if buckets is not None:
                    return Answer(
                        "list",
                        [_round_money(value) for value in buckets],
                        confidence=0.75,
                        plan="debt_age_buckets",
                    )
            if _has(
                t, "bao nhieu khach", "so khach", "may khach", "bao nhieu nguoi"
            ) and not _has(t, "tra bot", "tra no", "thu duoc"):
                # Debtor counts need the payment ledger, like debt totals.
                return Answer("abstain", plan="debt_customers_abstain")
            if _has(t, "top", "liet ke", "danh sach") and _has(t, "khach"):
                if _has(t, "kem so tien", "so no cua tung", "va so no"):
                    return Answer("abstain", plan="top_debtors_detail")
                top = ds.top_debtors(debt_as_of, k=k)
                if top:
                    return Answer(
                        "list",
                        [name for name, _ in top],
                        confidence=0.75,
                        plan="top_debtors",
                    )
                return Answer("abstain", plan="top_debtors")
            if _has(
                t, "tong", "bao nhieu tien", "con lai", "con no", "no con lai"
            ) and not _has(t, "tra bot", "tra no", "thu duoc", "nop"):
                # Outstanding debt needs the payment ledger; summing the
                # per-order debt column is wrong whenever later payments exist.
                return Answer("abstain", plan="debt_total_abstain")

        # Delivery rounds.
        if _has(
            t,
            "giao hang",
            "giao cho khach",
            "xe giao",
            "xe cho",
            "da giao",
            "cho giao",
        ) and _has(
            t,
            "dot giao",
            "luot giao",
            "chuyen giao",
            "dot hang",
            "so dot",
            "so chuyen",
            "may dot",
            "bn dot",
        ):
            count = ds.delivery_count(period)
            if count is not None:
                return Answer("number", count, confidence=0.8, plan="delivery_count")

        # Revenue by active ingredient (thuốc có hoạt chất X).
        if period is not None and _has(t, "hoat chat", "thanh phan"):
            ingredient = ds.match_ingredient(question)
            if ingredient and _has(
                t,
                "doanh thu",
                "dthu",
                "revenue",
                "thu duoc",
                "tien",
                "ban duoc",
                "ban dc",
            ):
                value = ds.revenue_ingredient(period, ingredient)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="revenue_ingredient",
                    )

        # Revenue of a customer group (sỉ / lẻ / thân thiết).
        if period is not None and _words(t, "nhom si", "nhom le", "nhom than thiet"):
            group = (
                "sỉ"
                if _has(t, "nhom si")
                else ("thân thiết" if _has(t, "nhom than thiet") else "lẻ")
            )
            # A group the ledger never uses makes the question unanswerable
            # (train gold precedent: no "thân thiết" tier in that shop).
            known_groups = set()
            if not ds.customers.empty and "group" in ds.customers.columns:
                known_groups.update(ds.customers["group"].map(norm_key).unique())
            if "customer_group" in ds.orders.columns:
                known_groups.update(
                    key
                    for key in ds.orders["customer_group"].map(norm_key).unique()
                    if key
                )
            if norm_key(group) not in known_groups:
                return Answer("unanswerable", plan="customer_group_unknown")
            part = ds.revenue_filtered(period, customer_group=group)
            if _has(t, "ti trong", "phan tram", "chiem") or "%" in raw:
                total = ds.revenue(period)
                if total > 0:
                    return Answer(
                        "percent",
                        round(100.0 * part / total, 2),
                        confidence=0.8,
                        plan="customer_group_share",
                    )
            if _has(t, "doanh thu", "dthu", "thu ve", "tien", "ban"):
                return Answer(
                    "number",
                    _round_money(part),
                    confidence=0.8,
                    plan="revenue_customer_group",
                )

        # Return/cancel reasons ("vì lỗi", "vì mua nhầm", "bên nào nhiều
        # hơn") are not recorded in the ledger (train gold: unanswerable).
        # This check must run before any count branch: questions like "bên
        # nào nhiều đơn hơn" contain no "bao nhiêu đơn" trigger.
        if _has(t, "vi loi", "vi mua nham", "ly do tra", "nguyen nhan tra") or (
            _has(t, "tra hang", "tra lai", "doi tra", "huy don")
            and _has(t, "ben nao", "loai nao", "nhieu hon", "so sanh", "vi ")
        ):
            return Answer("unanswerable", plan="return_reason_unknown")

        # Order counts filtered by category or status.
        if _has(t, "bao nhieu don", "may don", "so don", "bn don", "bao nhieu bill"):
            if _has(t, "trang thai", "tinh trang"):
                if _words(t, "huy"):
                    return Answer(
                        "number",
                        ds.cancelled_count(period),
                        confidence=0.8,
                        plan="cancelled_count",
                    )
                if _has(t, "tra lai", "tra hang", "hoan tra", "hoan hang"):
                    return Answer(
                        "number",
                        ds.returned_count(period),
                        confidence=0.8,
                        plan="returned_count",
                    )
                status = None
                if _has(t, "dang giao", "van chuyen", "dang gui"):
                    status = "shipping"
                elif _has(t, "hoan thanh", "da xong", "thanh cong", "da lay"):
                    status = "completed"
                elif _has(t, "cho xu ly", "dang xu ly", "dang chuan bi"):
                    status = "other"
                if status:
                    count = ds.orders_by_status(period, status)
                    if count is not None:
                        return Answer(
                            "number",
                            count,
                            confidence=0.8,
                            plan="orders_by_status",
                        )
                return Answer("abstain", plan="order_count_status")
            if _has(t, "tra lai", "tra hang", "hoan tra", "doi tra"):
                return Answer(
                    "number",
                    ds.returned_count(period),
                    confidence=0.8,
                    plan="returned_count",
                )
            from .verifier import match_category

            category = match_category(question, ds)
            if category and period is not None:
                count = ds.orders_with_category(period, category)
                if count is not None:
                    return Answer(
                        "number", count, confidence=0.8, plan="orders_by_category"
                    )
            group = re.search(
                r"nhom\s+([a-z0-9 ]{2,30}?)(?=\s+(?:co|ban|mua|thang|trong|"
                r"cua|la|gom|bao|chie|nhieu|may)\b|$)",
                t,
            )
            if group and period is not None:
                count = ds.orders_with_product_keyword(period, norm_key(group.group(1)))
                if count is not None:
                    return Answer(
                        "number",
                        count,
                        confidence=0.75,
                        plan="orders_by_product_keyword",
                    )

        # Customer counts by number of orders.
        threshold = re.search(r"(?:tu|it nhat)\s*(\d{1,3})\s*don", raw)
        if threshold and _has(t, "khach"):
            minimum = int(threshold.group(1))
            if (
                _has(t, "chiem", "ti le", "ty le", "phan tram", "bao nhieu %")
                or "%" in raw
            ):
                share = ds.customer_order_share(period, minimum)
                if share is not None:
                    return Answer(
                        "percent",
                        share,
                        confidence=0.8,
                        plan="customer_order_share",
                    )
            count = ds.customers_with_orders(period, minimum)
            if count is not None:
                return Answer(
                    "number",
                    count,
                    confidence=0.8,
                    plan="customers_with_orders",
                )
        if _has(t, "khach") and _has(
            t, "mua nhieu don nhat", "nhieu don nhat", "mua nhieu don"
        ):
            top = ds.top_customer_orders(period, k=1)
            if top:
                return Answer(
                    "string", top[0][0], confidence=0.7, plan="top_customer_orders"
                )

        # Prescription vs over-the-counter revenue.
        if _has(t, "ke don", "thuoc ke don"):
            if period is not None:
                is_otc = _has(
                    t,
                    "khong ke don",
                    "ko ke don",
                    "k ke don",
                    "hong ke don",
                    "khong can ke don",
                    "tru thuoc ke don",
                    "tru ke don",
                )
                if _has(t, "chiem", "ti trong", "phan tram") or "%" in raw:
                    share = ds.prescription_share(period)
                    if share is not None:
                        return Answer(
                            "percent",
                            share,
                            confidence=0.8,
                            plan="prescription_share",
                        )
                if _has(t, "doanh thu", "dthu", "thu ve", "tien", "ban duoc", "ban dc"):
                    value = ds.revenue_prescription(period, not is_otc)
                    if value is not None:
                        return Answer(
                            "number",
                            _round_money(value),
                            confidence=0.8,
                            plan="prescription_revenue",
                        )

        # Quantity-weighted purchase average price (rulebook §5).
        if _has(t, "nhap", "gia nhap", "gia mua", "nhap vao") and _has(
            t, "binh quan", "trung binh"
        ):
            from .verifier import match_product

            product = match_product(question, ds)
            if product and period is not None:
                value = ds.purchase_avg_price(product, period)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="purchase_avg_price",
                    )
            return Answer("abstain", plan="purchase_avg_price")

        # Rulebook §5 price-change effect: compare the N days from the price
        # change with the N days immediately before it, in base units.
        if (
            _has(t, "doi gia", "tang gia", "len gia", "giam gia", "thay doi gia")
            and _has(t, "ngay", "ke tu")
            and _has(raw, "%", "phan tram", "may phan tram", "bao nhieu phan tram")
            and not _has(t, "voucher", "chiet khau")
        ):
            from .verifier import match_product

            product = match_product(question, ds)
            days_match = re.search(
                r"(\d{1,2})\s*ngay\s*(?:ke tu|ke tu ngay|sau|tinh tu)", t
            )
            date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
            if product and days_match and date_match:
                n = int(days_match.group(1))
                day = date(
                    int(date_match.group(3)),
                    int(date_match.group(2)),
                    int(date_match.group(1)),
                )
                after = ds.product_quantity(
                    Period(day, day + timedelta(days=n - 1)), product
                )
                before = ds.product_quantity(
                    Period(day - timedelta(days=n), day - timedelta(days=1)),
                    product,
                )
                if before and before > 0 and after is not None:
                    value = round(100.0 * (after - before) / before, 2)
                    return Answer(
                        "percent",
                        value,
                        confidence=0.85,
                        plan="price_change_effect",
                    )
            return Answer("abstain", plan="price_change_comparison")

        # Percent / comparison questions need a comparison metric; the revenue
        # gap makes them unreliable, so abstain instead of risking -1.
        # Cancellation-rate questions also ask for a percentage and are handled below.
        if not (_words(t, "huy") and _has(t, "don", "dh")) and not _has(
            t, "ca sang", "ca chieu"
        ):
            asks_percent = _has(t, "phan tram", "ti le", "ty le", "ti trong") or bool(
                re.search(r"(?:bao nhieu|may|bn)\s*%", raw)
            )
            compares = _has(
                t,
                "tang truong",
                "thay doi",
                "hon hay kem",
                "hown hay kem",
                "cao hon",
                "thap hon",
                "so voi",
                "cung ky",
                "tang bao nhieu",
                "giam bao nhieu",
            )
            if asks_percent or compares:
                growth_words = _has(
                    t,
                    "thay doi",
                    "tang truong",
                    "hon hay kem",
                    "tang bao nhieu",
                    "giam bao nhieu",
                    "tang hay giam",
                    "may %",
                    "bao nhieu %",
                )
                years = [
                    int(value)
                    for value in re.findall(r"thang\s+\d+\s+nam\s+(\d{4})", t)
                ]
                years.extend(int(value) for value in re.findall(r"\b(20[0-2]\d)\b", t))
                single_month = (
                    period is not None
                    and period.start.year == period.end.year
                    and period.start.month == period.end.month
                )
                yoy = _is_yoy(raw, t, period) or (
                    single_month
                    and any(year != period.start.year for year in years)
                    and _has(t, "so voi", "tang truong", "thay doi", "so vs")
                )
                if yoy and period is not None:
                    metric = (
                        "orders"
                        if _has(t, "so don", "don hang", "so bill")
                        else "revenue"
                    )
                    if is_tet_period(period):
                        prior = tet_period(period.start.year - 1)
                        value = _period_growth(ds, period, prior, metric)
                    else:
                        value = ds.growth_yoy(period, metric)
                    if value is not None:
                        return Answer(
                            "percent",
                            value,
                            confidence=0.75,
                            plan="revenue_growth_yoy",
                        )
                    return Answer("abstain", plan="growth_no_base")
                share_words = _has(t, "chiem", "ti trong", "dong gop", "cung ky nam")
                if (
                    growth_words
                    and not share_words
                    and period is not None
                    and not _has(t, "ca sang", "ca chieu", "gia ", "gia ban", "doi gia")
                ):
                    previous = previous_period(period)
                    bounds = ds.date_range()
                    if bounds and previous.end < bounds[0]:
                        return Answer("unanswerable", plan="growth_no_prior_period")
                    metric = (
                        "orders"
                        if _has(t, "so don", "don hang", "so bill")
                        else "revenue"
                    )
                    value = ds.growth(period, metric)
                    if value is not None:
                        return Answer(
                            "percent",
                            value,
                            confidence=0.75,
                            plan="revenue_growth",
                        )
                    return Answer("abstain", plan="growth_no_base")
                return Answer("abstain", plan="percent_or_comparison")

        # Threshold counts: orders or days whose revenue passes a floor.
        if (
            _has(t, "doanh thu", "dthu", "tien hang", "tien ban")
            and not _has(t, "nhap hang")
            and (
                _words(t, "tu", "tren", "duoi", "hon", "qua")
                or _has(t, "tro len", "tro xuong", "dat muc", "dat ")
                or _has(raw, "≥", ">=", "<=", ">", "<")
            )
        ):
            lower, upper = _money_thresholds(raw)
            if lower is not None or upper is not None:
                if period is None:
                    return Answer("abstain", plan="threshold_count")
                if _has(t, "bao nhieu ngay", "may ngay", "so ngay", "nhieu ngay"):
                    return Answer(
                        "number",
                        ds.count_days_by_revenue(period, lower, upper),
                        confidence=0.85,
                        plan="days_above_revenue",
                    )
                if _words(t, "don", "bill") or _has(t, "hoa don"):
                    return Answer(
                        "number",
                        ds.count_orders_by_revenue(period, lower, upper),
                        confidence=0.85,
                        plan="orders_above_revenue",
                    )

        # "Liệt kê" with a non-product criterion is not handled.
        if _has(t, "liet ke") and not _has(t, "ban chay", "ban nhieu", "hot nhat"):
            return Answer("abstain", plan="list_other")

        # 1. "Ngày nào / tháng nào ... cao nhất" -> date.
        if _has(t, "ngay nao") and _has(
            t,
            "cao nhat",
            "nhieu nhat",
            "nhiu nhat",
            "dat nhat",
            "ban dac nhat",
            "ban dat nhat",
            "nhieu tien nhat",
            "nhiu tien nhat",
        ):
            if period is None or not _has(
                t, "doanh thu", "dthu", "ban duoc", "thu ve", "tien"
            ):
                return Answer("abstain", plan="top_day")
            top = ds.top_days(period, k=1)
            if not top:
                return Answer("abstain", plan="top_day")
            return Answer(
                "date", top[0][0].isoformat(), confidence=0.8, plan="top_revenue_day"
            )

        if _has(t, "thang nao") and _has(
            t, "cao nhat", "nhieu nhat", "nhiu nhat", "lon nhat", "nhieu tien nhat"
        ):
            if period is None or not _has(
                t, "doanh thu", "dthu", "ban duoc", "ban dc", "thu ve", "nhiu tien"
            ):
                return Answer("abstain", plan="top_month")
            top = ds.top_months(period, k=1)
            if not top:
                return Answer("abstain", plan="top_month")
            return Answer("date", top[0][0], confidence=0.75, plan="top_revenue_month")

        # 2. Ranking products.
        if (
            _has(
                t,
                "ban chay nhat",
                "ban nhieu nhat",
                "ban dc nhieu nhat",
                "ban duoc nhieu",
                "ban ra nhieu",
                "ban dc nhieu",
                "hot nhat",
                "ban chay",
                "chay nhat",
                "ban nhieu",
                "nhieu ly nhat",
                "nhieu kg nhat",
                "nhieu hop nhat",
                "nhieu chai nhat",
                "nhieu lon nhat",
                "nhieu dia nhat",
                "nhieu phan nhat",
                "nhieu goi nhat",
                "nhieu cay nhat",
                "nhieu bao nhat",
            )
            and not _has(t, "size m", "size l")
            and not _has(
                t,
                "bao nhieu tien",
                "bao nhiu tien",
                "nhiu tien",
                "nhieu tien",
                "doanh thu",
                "dthu",
            )
        ):
            if period is None:
                return Answer("abstain", plan="top_product")
            from .verifier import match_category

            category = match_category(question, ds)
            if category:
                top = ds.top_products_filtered(
                    period, by="quantity", k=k, category=category
                )
            else:
                top = ds.top_products(period, by="quantity", k=k)
            if not top:
                return Answer("abstain", plan="top_product")
            if ds.mixed_bulk_units([name for name, _ in top]):
                return Answer("abstain", plan="top_products_mixed_units")
            if k > 1 or _has(t, "liet ke", "top", "danh sach"):
                return Answer(
                    "list",
                    [name for name, _ in top],
                    confidence=0.7,
                    plan="top_products_quantity",
                )
            return Answer(
                "string", top[0][0], confidence=0.75, plan="top_product_quantity"
            )

        # 2b. Ranking products by revenue.
        if (
            period is not None
            and _has(
                t,
                "thu ve nhieu tien nhat",
                "doanh thu nhieu nhat",
                "ban duoc nhieu tien nhat",
                "ban dc nhieu tien nhat",
            )
            and _has(t, "mat hang", "san pham", "hang nao", "mon nao", "sp nao")
        ):
            from .verifier import match_category

            category = match_category(question, ds)
            top = ds.top_products_filtered(
                period, by="line_amount", k=k, category=category
            )
            if top:
                return Answer(
                    "string", top[0][0], confidence=0.75, plan="top_product_revenue"
                )

        # 2e. Special rankings from the public rulebook (§3).
        if period is not None:
            if _has(t, "trung thanh") and _has(t, "khach"):
                top = ds.top_customers_days(period)
                if top:
                    return Answer(
                        "string", top[0][0], confidence=0.7, plan="top_customer_days"
                    )
            if _words(t, "sop") and _has(t, "khach"):
                top = ds.top_customers_avg(period)
                if top:
                    return Answer(
                        "string", top[0][0], confidence=0.7, plan="top_customer_avg"
                    )
            if _has(t, "doi y") and _has(t, "khach"):
                top = ds.top_customers_cancelled(period)
                if top:
                    return Answer(
                        "string",
                        top[0][0],
                        confidence=0.7,
                        plan="top_customer_cancelled",
                    )
            if _has(t, "ua chuong", "ua thich") and _has(
                t, "hang", "san pham", "mat hang", "mon", "sp"
            ):
                top = ds.top_products_orders(period, k=max(1, k))
                if top:
                    if k > 1 or _has(t, "liet ke", "top", "danh sach"):
                        return Answer(
                            "list",
                            [name for name, _ in top],
                            confidence=0.7,
                            plan="top_products_orders",
                        )
                    return Answer(
                        "string",
                        top[0][0],
                        confidence=0.7,
                        plan="top_product_orders",
                    )
            if _has(t, "gio cao diem", "cao diem"):
                hour = ds.peak_hour(period)
                if hour is not None:
                    return Answer("number", hour, confidence=0.7, plan="peak_hour")
            if _has(t, "dat khach") or (
                _has(t, "ngay nao") and _has(t, "nhieu don nhat")
            ):
                day = ds.top_day_orders(period)
                if day:
                    return Answer(
                        "date", day.isoformat(), confidence=0.7, plan="top_day_orders"
                    )

        # 3. Ranking customers.
        if _has(
            t,
            "khach mua nhieu nhat",
            "khach mua nhieu tien nhat",
            "khach hang mua nhieu",
            "khach nao mua nhieu",
            "ai mua hang nhieu",
            "ai mua nhieu",
            "mua nhieu tien nhat",
            "mua nhieu nhat",
            "mua hang nhieu tien nhat",
            "mua hang nhiu tien nhat",
            "mua nhiu tien nhat",
            "ai mua nhieu tien nhat",
            "ai mua hang nhieu tien nhat",
            "ai mua hang nhiu tien nhat",
            "tong tien mua cao nhat",
            "tong tien mua hang cao nhat",
            "tien mua hang cao nhat",
            "mua hang cao nhat",
            "mua cao nhat",
            "khach mua cao nhat",
            "chi tieu nhieu nhat",
            "chi tieu nhiu nhat",
            "3 khach",
        ) and not _has(t, "cong no", "no "):
            if period is None:
                return Answer("abstain", plan="top_customer")
            top = ds.top_customers(period, k=max(1, k))
            if not top:
                return Answer("abstain", plan="top_customer")
            if k > 1 or _has(t, "liet ke", "danh sach", "top"):
                return Answer(
                    "list",
                    [name for name, _ in top],
                    confidence=0.7,
                    plan="top_customers_revenue",
                )
            return Answer(
                "string", top[0][0], confidence=0.65, plan="top_customer_revenue"
            )

        # 3b. Cups by size, e.g. "bán được bao nhiêu ly size M".
        if _has(t, "size m", "size l") and _has(t, "ly", "coc"):
            if period is None:
                return Answer("abstain", plan="quantity_by_size")
            size = "m" if "size m" in t else "l"
            value = ds.quantity_by_unit(period, unit="ly", size=size)
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="quantity_by_size",
                )

        # 3c. Total cups, e.g. "tuần trước bán ra mấy ly".
        if (
            _has(t, "ly", "coc")
            and not _has(t, "size m", "size l", "loai", "nao", "vi ", "huong")
            and _has(
                t,
                "ban duoc",
                "ban dc",
                "ban ra",
                "may ly",
                "bn ly",
                "bao nhieu ly",
            )
        ):
            from .verifier import match_product

            if match_product(question, ds) is None:
                if period is None:
                    return Answer("abstain", plan="quantity_by_unit")
                value = ds.quantity_by_unit(period, unit="ly")
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="quantity_by_unit",
                    )

        # 4. Cancellation rate / count.
        if (
            _words(t, "huy")
            and _has(t, "don", "dh")
            and (_has(raw, "%") or _has(t, "ti le", "ty le", "phan tram", "chiem"))
            or _has(
                t,
                "ti le huy",
                "ty le huy",
                "ti le don huy",
                "ty le don huy",
                "ti le huy don",
            )
        ):
            if period is None:
                return Answer("abstain", plan="cancel_rate")
            return Answer(
                "percent",
                ds.cancellation_rate(period),
                confidence=0.85,
                plan="cancellation_rate",
            )
        # A cancellation "vì <lý do>" (late delivery, out of stock, ...)
        # needs a reason the ledger does not record. Train gold marks the
        # identical "huỷ vì giao trễ" question unanswerable.
        if _words(t, "huy") and _has(
            t,
            "giao tre",
            "giao muon",
            "ship tre",
            "ship muon",
            "giao hang tre",
            "van chuyen tre",
            "giao cham",
            "ship cham",
        ):
            return Answer("unanswerable", plan="cancel_reason_unknown")
        if _words(t, "huy") and _has(t, "don", "dh", "hoa don", "bill"):
            if period is None:
                return Answer("abstain", plan="cancel_count")
            return Answer(
                "number",
                ds.cancelled_count(period),
                confidence=0.8,
                plan="cancelled_count",
            )

        # 4a. Products present in at least N distinct orders.
        if _has(
            t,
            "xuat hien",
            "co mat",
            "duoc mua o",
            "duoc mua tai",
            "mua o",
            "mua tai",
        ) and _has(t, "don", "don hang"):
            match = re.search(r"(?:tu|it nhat)?\s*(\d{1,4})\s*don", raw)
            if match and period is not None:
                return Answer(
                    "number",
                    ds.products_in_orders(period, int(match.group(1))),
                    confidence=0.8,
                    plan="products_in_orders",
                )

        # 4b. Products that did not sell in the period.
        if _has(t, "bao nhieu mat hang", "may mat hang", "so mat hang") and (
            _words(t, "e")
            or _has(t, "khong ban", "ko ban", "ban cham", "e han", "khong ai mua")
        ):
            if period is not None:
                return Answer(
                    "number",
                    ds.unsold_count(period),
                    confidence=0.8,
                    plan="unsold_products",
                )

        # 4b. Closing stock value at cost. "ton" needs a word boundary:
        # "tổng" (total) contains it as a substring.
        if (
            _has(t, "gia von", "gia tri")
            and (
                _words(t, "ton")
                or _has(t, "ton kho", "ton cuoi", "trong kho", "tien hang")
            )
            and _has(t, "bao nhieu", "bnhiu", "may")
        ):
            value = ds.inventory_value()
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="inventory_value",
                )

        # 4b1. Closing stock as [full packs, loose units]. Pack words are
        # open-ended (hộp, xe, cuộn, cây, ...); the chẵn/lẻ pair is the
        # signature. Customer tiers ("khách lẻ") never pair with "chẵn".
        if _has(
            t,
            "hop chan",
            "goi chan",
            "thung chan",
            "bao chan",
            "xe chan",
            "cuon chan",
            "cay chan",
        ) or (
            _has(t, "chan")
            and _words(t, "le")
            and (_words(t, "con") or _has(t, "ton", "kho", "cuoi ngay", "hom nay"))
            and not _has(t, "khach")
        ):
            stock_product = _match_inventory(question, ds)
            if stock_product:
                packs = ds.stock_packs(stock_product)
                if packs is not None:
                    return Answer(
                        "list",
                        packs,
                        confidence=0.75,
                        plan="stock_packs",
                    )
            return Answer("abstain", plan="stock_packs")

        # 4c. Closing stock, direct from the inventory snapshot.
        if (
            _words(t, "ton")
            or _has(t, "ton kho", "ton cuoi", "trong kho", "con bao nhieu", "con lai")
        ) and not (
            _has(t, "gia tri", "gia von", "nhap", "don vi lon", "hop chan", "le")
            or _words(t, "ban")
        ):
            stock_product = _match_inventory(question, ds)
            if stock_product:
                value = ds.stock(stock_product)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="stock_on_hand",
                    )

        # 4a2. Delivered quantity from delivery notes ("xe đã giao ...").
        # Must run before sales-quantity branches: the same product/quantity
        # words appear, but delivery notes are the source of truth.
        # Must also run before the stock branches below: "tổng cộng" does
        # not contain "tồn" (word boundary), but older plans misrouted here.
        if _has(
            t,
            "xe da giao",
            "da giao",
            "duoc giao",
            "da cho di",
            "duoc cho",
            "cho di",
            "giao duoc",
            "cho di",
        ) and not _has(t, "khong giao duoc", "khong giao hang", "chua giao"):
            from .verifier import match_product as _match_product_dl

            delivered_product = _match_product_dl(question, ds)
            if delivered_product and _has(
                t, "bao nhieu", "may", "bnhiu", "tong cong", "tat ca"
            ):
                # "Các đơn đặt tháng X ... đã chở đi": the month selects the
                # ORDER date; delivery only needs to be on/before as_of.
                order_period = None
                if _has(t, "don dat", "don hang dat", "dat thang", "don thang"):
                    refs = extract_month_refs(raw, as_of)
                    if refs:
                        year, month = refs[0]
                        order_period = Period(
                            date(year, month, 1),
                            _month_end(year, month),
                            f"tháng {month:02d}/{year}",
                        )
                if order_period is not None:
                    delivered = ds.delivered_quantity_for_orders(
                        order_period, as_of, delivered_product
                    )
                else:
                    # "Tới hôm nay ... tổng cộng" is cumulative up to as_of,
                    # not just today's deliveries.
                    bounds = ds.date_range()
                    if _has(t, "toi hom nay", "den hom nay", "tinh toi") and bounds:
                        cumulative = Period(bounds[0], as_of, "tới hôm nay")
                    else:
                        cumulative = period
                    if cumulative is None and bounds:
                        cumulative = Period(bounds[0], as_of, "tới hôm nay")
                    delivered = (
                        ds.delivered_quantity(cumulative, delivered_product)
                        if cumulative is not None
                        else None
                    )
                if delivered is not None:
                    return Answer(
                        "number",
                        _round_money(delivered),
                        confidence=0.75,
                        plan="delivered_quantity",
                    )
            return Answer("abstain", plan="delivered_quantity")

        # 4b. Product-aware revenue and quantity.
        from .verifier import match_product

        product = match_product(question, ds)
        if product and period is not None:
            price_words = _has(
                t,
                "gia ban",
                "don gia",
                "gia bao nhieu",
                "bao nhieu mot",
                "voi don gia",
                "gia cua",
                "ban voi gia",
            )
            if (
                price_words
                and period.start == period.end
                and not _has(t, "nhap", "binh quan", "trung binh")
            ):
                value = ds.price_on_date(product, period.start)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="price_on_date",
                    )
            price_inventory = _has(
                t,
                "gia ban",
                "don gia",
                "gia bao nhieu",
                "bao nhieu mot",
                "gia nhap",
                "nhap vao",
                "gia binh quan",
                "ton kho",
                "ton cuoi",
                "con lai",
                "con bao nhieu",
                "cuoi ngay",
                "trong kho",
                "het han",
            )
            money_words = _has(
                t,
                "doanh thu",
                "dthu",
                "thu ve",
                "tien ban",
                "tieng ban",
                "tieng bang",
                "tieenf",
                "tienf",
                "bao nhieu tien",
                "bao nhiu tien",
                "bnhiu tien",
                "bn tien",
                "bnhieu tien",
                "nhiu tien",
                "may tien",
                "thu duoc",
            )
            if money_words and not price_inventory:
                value = ds.revenue_filtered(period, product=product)
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="revenue_product",
                )
            # "Còn bao nhiêu" asks what REMAINS (stock), not what sold.
            # With no inventory ledger the question is unanswerable.
            # "con" needs a word boundary: "tổng cộng" contains it.
            if (
                _words(t, "con")
                and not _has(t, "ban duoc", "ban ra", "ban dc", "ban chay", "ban nhieu")
                and not _has(t, "no", "cong no", "tra no", "thu no")
                and not (_has(t, "chan") and _words(t, "le"))
            ):
                if ds.inventory is None or ds.inventory.empty:
                    return Answer("unanswerable", plan="stock_no_inventory")
                value = ds.stock(product)
                if value is None:
                    return Answer("unanswerable", plan="stock_unknown_product")
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="stock_on_hand",
                )
            quantity_words = _has(
                t, "ban duoc", "ban ra", "ban dc", "may ", "bao nhieu", "so luong"
            )
            has_unit = _words(
                t,
                "kg",
                "tan",
                "cay",
                "cuon",
                "bao",
                "cai",
                "hop",
                "chai",
                "binh",
                "lon",
                "ly",
                "dia",
                "phan",
                "noi",
                "goi",
                "tup",
                "lo",
                "vien",
                "mieng",
                "thung",
                "ket",
                "m3",
                "met",
                "bo",
                "chiec",
            )
            if not price_inventory and quantity_words and (has_unit or "so luong" in t):
                value = ds.product_quantity(period, product=product)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="product_quantity",
                    )

        # 4b2. Quantity by unit when no product is named.
        if (
            period is not None
            and _has(t, "bao nhieu", "may", "bnhiu", "bao nhiu")
            and not _has(
                t,
                "tien",
                "doanh thu",
                "dthu",
                "gia",
                "ton",
                "trong kho",
                "con lai",
                "don hang",
                "so don",
                "bao nhieu don",
                "may don",
                "hoa don",
                "bill",
            )
            and not (_words(t, "xe") and _words(t, "chan") and _words(t, "le"))
        ):
            from .verifier import match_unit

            unit = match_unit(question, ds)
            if unit:
                value = ds.quantity_by_unit(period, unit)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="quantity_by_unit",
                    )

        # 4c. Category revenue, e.g. "nhóm cát đá bán được bao nhiêu tiền".
        if period is not None:
            from .verifier import match_category

            category = match_category(question, ds)
            if (
                category
                and _has(
                    t,
                    "doanh thu",
                    "dthu",
                    "thu ve",
                    "tien ban",
                    "ban duoc",
                    "ban dc",
                    "nhiu tien",
                    "bnhiu tien",
                )
                and not _has(t, "chiem", "ti trong", "dong gop", "phan tram", "may %")
            ):
                value = ds.revenue_filtered(period, category=category)
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="revenue_category",
                )

        # 5. Growth / change percent: the baseline revenue gap makes this
        # unreliable, so abstain instead of risking -1.
        if _has(
            t,
            "tang truong",
            "phan tram thay doi",
            "thay doi bao nhieu",
            "tang bao nhieu phan tram",
            "giam bao nhieu phan tram",
            "tang hay giam",
        ) and not _has(t, "ca sang", "ca chieu"):
            return Answer("abstain", plan="growth")

        # 6. Average order value.
        if _has(
            t,
            "trung binh moi don",
            "binh quan moi don",
            "gia tri trung binh mot don",
            "trung binh mot don",
            "trung binh tren mot don",
            "binh quan moi bill",
            "trung binh moi bill",
            "binh quan mot bill",
            "binh quan tren mot bill",
            "trung binh 1 don",
            "binh quan 1 don",
            "trung binh mot dh",
            "binh quan mot dh",
            "trung binh 1 phieu",
            "binh quan 1 phieu",
            "binh quan don hang",
            "binh quan tren 1 dh",
            "binh quan 1 dh",
            "trung binh 1 dh",
            "trung binh moi dh",
            "binh quan moi dh",
            "gia tri trung binh don",
            "trung binhf",
            "binh quanf",
            "trung binhf mot ddown",
            "trung binh mot ddown",
            "trung binhf 1 ddown",
        ) or re.search(
            r"(?:trung binh|binh quan)[a-z ]{0,20}"
            r"(?:mot|moi|1|moi|mooix|moix)\s*"
            r"(?:don|dh|bill|phieu|hoa don|order)",
            t,
        ):
            if period is None:
                return Answer("abstain", plan="avg_order")
            count = ds.order_count(period)
            if count == 0:
                return Answer("abstain", plan="avg_order")
            return Answer(
                "number",
                _round_money(ds.revenue(period) / count),
                confidence=0.8,
                plan="avg_order_value",
            )

        # 7. Debt: needs paid/outstanding semantics that vary by schema.
        if _has(t, "con no", "cong no", "chua tra", "no tien", "khach no"):
            return Answer("abstain", plan="debt_total")

        # 6b. Shift (ca sáng / ca chiều) revenue and order counts.
        if _has(t, "ca sang", "ca chieu"):
            compare_at = min(
                [
                    position
                    for position in (t.find("so voi"), t.find("so vs"))
                    if position != -1
                ]
                or [-1]
            )
            if compare_at != -1:
                before, after = t[:compare_at], t[compare_at:]
                if "ca chieu" in before:
                    shift = "afternoon"
                elif "ca sang" in before:
                    shift = "morning"
                elif "ca sang" in after:
                    shift = "afternoon"
                elif "ca chieu" in after:
                    shift = "morning"
                else:
                    shift = "morning"
            else:
                pos_sang = t.find("ca sang")
                pos_chieu = t.find("ca chieu")
                if pos_chieu == -1:
                    shift = "morning"
                elif pos_sang == -1:
                    shift = "afternoon"
                else:
                    shift = "morning" if pos_sang < pos_chieu else "afternoon"
            other = "afternoon" if shift == "morning" else "morning"
            compares = _has(
                t,
                "so voi",
                "chenh",
                "hon",
                "kem",
                "tang",
                "giam",
                "may %",
                "bao nhieu %",
            )
            if compares:
                if _words(t, "don", "bill") or _has(t, "hoa don"):
                    base = ds.orders_shift(period, other)
                    value = ds.orders_shift(period, shift)
                else:
                    base = ds.revenue_shift(period, other)
                    value = ds.revenue_shift(period, shift)
                if base and value is not None and base > 0:
                    return Answer(
                        "percent",
                        round(100.0 * (value - base) / base, 2),
                        confidence=0.8,
                        plan="shift_change",
                    )
            if _words(t, "don", "bill") or _has(t, "hoa don"):
                count = ds.orders_shift(period, shift)
                if count is not None:
                    return Answer(
                        "number", count, confidence=0.8, plan="shift_order_count"
                    )
            if _has(t, "doanh thu", "dthu", "tien", "ban"):
                value = ds.revenue_shift(period, shift)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="shift_revenue",
                    )

        # 6c. Employee revenue, ticket counts and top seller.
        if period is not None:
            is_employee_q = _has(t, "nhan vien", "nv ") or _words(t, "tho")
            if (
                is_employee_q
                and _has(t, "ai", "nao")
                and _has(t, "nhat", "cao nhat", "gioi nhat", "cham chi")
                and not _has(t, "it nhat", "thap nhat")
            ):
                if _has(t, "cham chi", "ngay lam"):
                    name = ds.top_employee_days(period)
                    plan = "top_employee_days"
                elif _has(t, "luot", "phieu"):
                    name = ds.top_employee_tickets(period)
                    plan = "top_employee_tickets"
                else:
                    name = ds.top_employee(period)
                    plan = "top_employee"
                if name:
                    return Answer("string", name, confidence=0.75, plan=plan)
            employee = ds.match_employee(question) if is_employee_q else None
            if employee is None and is_employee_q:
                code = re.search(r"\bnv\s*0*(\d+)\b", raw)
                if code:
                    employee = f"NV{int(code.group(1)):03d}"
            if employee:
                if _has(t, "luot", "phieu") and _has(
                    t, "bao nhieu", "may", "bnhiu", "nhiu", "nhieu"
                ):
                    count = ds.employee_order_count(period, employee)
                    if count is not None:
                        return Answer(
                            "number",
                            count,
                            confidence=0.75,
                            plan="employee_order_count",
                        )
                value = ds.revenue_employee(period, employee)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="employee_revenue",
                    )

        # 6d. Service ticket count by job keyword ("thay dầu láp ...").
        if (
            period is not None
            and _has(t, "luot", "phieu")
            and _has(t, "bao nhieu", "may", "bnhiu", "nhiu", "nhieu")
            and not _has(t, "nhan vien", "nv ", "ton", "trong kho", "con lai")
            and not _words(t, "le", "chan")
        ):
            service = _match_service(question, ds)
            count = None
            if service:
                count = ds.orders_with_service_keyword(period, norm_key(service))
            if count is None:
                from .verifier import match_product

                product = match_product(question, ds)
                if product:
                    count = ds.orders_with_product_keyword(period, norm_key(product))
            if count is not None:
                return Answer(
                    "number",
                    count,
                    confidence=0.75,
                    plan="service_ticket_count",
                )

        # 6e. Commission (hoa hồng) by employee.
        if period is not None and _has(t, "hoa hong", "hue hong", "commission"):
            if _has(t, "cao nhat", "nhieu nhat", "nhiu nhat") and _has(
                t, "ai", "nao", "tho"
            ):
                ranking = ds.commission_by_employee(period)
                if ranking:
                    return Answer(
                        "string",
                        ranking[0][0],
                        confidence=0.75,
                        plan="top_commission",
                    )
            employee = ds.match_employee(question)
            if employee:
                value = ds.commission(period, employee)
                if value is not None:
                    return Answer(
                        "number",
                        _round_money(value),
                        confidence=0.8,
                        plan="employee_commission",
                    )

        # 7b. Supplier ranking.
        if _has(t, "nha cung cap", "nhaf cung cap", "ncc") and _has(
            t,
            "nhieu tien nhat",
            "nhieu tien hang nhat",
            "cao nhat",
            "nhieu nhat",
            "lon nhat",
        ):
            if period is not None:
                supplier = ds.top_supplier(period)
                if supplier:
                    return Answer(
                        "string",
                        supplier,
                        confidence=0.75,
                        plan="top_supplier",
                    )

        # 7c. Trade-in, paid and warranty ledgers.
        if _has(t, "thu may cu") and _has(
            t, "bao nhieu", "bnhiu", "may", "tien", "tong", "cong"
        ):
            value = (
                ds.tradein_total(period) if period is not None else ds.tradein_total()
            )
            if value is not None:
                return Answer(
                    "number",
                    _round_money(value),
                    confidence=0.8,
                    plan="tradein_total",
                )
        if _has(t, "bao hanh") and _has(
            t, "luot", "nuot", "lan", "phieu", "may luot", "bao nhieu", "may"
        ):
            count = ds.warranty_count(period)
            if count is not None:
                return Answer("number", count, confidence=0.8, plan="warranty_count")
        if (
            period is not None
            and _has(t, "khach phai tra", "tien khach tra", "khach da tra", "khach tra")
            and not _has(t, "chua tra", "con no", "no ", "gop")
        ):
            return Answer(
                "number",
                _round_money(ds.paid_total(period)),
                confidence=0.8,
                plan="paid_total",
            )

        # 7d. Imported quantity and import-minus-sales.
        from .verifier import match_product as _match_product

        import_product = _match_product(question, ds)
        if period is not None and import_product and _has(t, "nhap"):
            purchased = ds.purchase_quantity(import_product, period)
            if purchased is not None:
                if _has(t, "tru", "tru bot", "tru so") and _has(t, "ban"):
                    sold = ds.product_quantity(period, import_product)
                    if sold is not None:
                        return Answer(
                            "number",
                            _round_money(purchased - sold),
                            confidence=0.8,
                            plan="import_minus_sales",
                        )
                if _has(t, "bao nhieu", "bnhiu", "may"):
                    return Answer(
                        "number",
                        _round_money(purchased),
                        confidence=0.8,
                        plan="purchase_quantity",
                    )

        # 8. Purchase totals, only when the semantic layer found priced purchases.
        if _has(
            t,
            "nhap hang",
            "phieu nhap",
            "don nhap",
            "tien nhap",
            "tien hang nhap",
            "gia tri nhap",
            "nhap vao",
            "bo ra bao nhieu tien",
        ):
            if period is None or _has(
                t,
                "gia nhap",
                "binh quan",
                "trung binh",
                "nha cung cap",
                "mat hang",
                "san pham",
            ):
                return Answer("abstain", plan="purchase_total")
            total = ds.purchase_total(period)
            if total <= 0:
                return Answer("abstain", plan="purchase_total")
            return Answer(
                "number",
                _round_money(total),
                confidence=0.8,
                plan="purchase_total",
            )

        # 9. Order count.
        if _has(
            t,
            "so don",
            "bao nhieu don",
            "bao nhiu don",
            "bn don",
            "may don",
            "nhieu don",
            "nhiu don",
            "ddown",
            "dh",
            "so luong don",
            "so hoa don",
        ):
            if period is None:
                return Answer("abstain", plan="order_count")
            if _has(t, *_FILTER_WORDS) or _has(
                t, "co dthu", "co doanh thu", "doanh thu tu", "tu 55", "tren ", "duoi "
            ):
                return Answer("abstain", plan="order_count_filtered")
            return Answer(
                "number", ds.order_count(period), confidence=0.8, plan="order_count"
            )

        # 9b. Explicit month range -> list of every month in the range.
        m = re.search(
            r"tu\s+thang\s*(\d{1,2})/(\d{4})\s+(?:den|toi|qua)\s+thang\s*(\d{1,2})/(\d{4})",
            raw,
        )
        if m and _has(t, "doanh thu", "dthu", "thu ve", "moi thang", "tung thang"):
            year, month = int(m.group(2)), int(m.group(1))
            end_year, end_month = int(m.group(4)), int(m.group(3))
            months = []
            while (year, month) <= (end_year, end_month) and len(months) <= 24:
                months.append((year, month))
                month += 1
                if month == 13:
                    month = 1
                    year += 1
            values = [
                _round_money(ds.revenue(resolve_period(f"thang {mo}/{y}", as_of)))
                for y, mo in months
            ]
            return Answer("list", values, confidence=0.75, plan="revenue_monthly_range")

        # 9c. Multiple explicit months asked for money -> list of monthly values.
        refs = extract_month_refs(raw, as_of)
        if (
            len(refs) >= 2
            and _has(t, "doanh thu", "dthu", "thu ve")
            and _has(t, "bao nhieu", "la bao nhieu", "ghi", "lần lượt", "lan luot")
            and not _has(
                t, "tang", "giam", "cao hon", "thap hon", "lien tuc", "so sanh"
            )
        ):
            values = []
            for year, month in sorted(refs):
                p = resolve_period(f"thang {month}/{year}", as_of)
                if p is None:
                    return Answer("abstain", plan="revenue_monthly_list")
                values.append(_round_money(ds.revenue(p)))
            return Answer("list", values, confidence=0.75, plan="revenue_monthly_list")

        # 9c. "N tháng liền nhau kết thúc ở tháng M/YYYY" -> list of values.
        m = re.search(
            r"(\d{1,2})\s*thang\s*lien nhau\s*(?:ket thuc|tinh den|den)\s*"
            r"(?:o|tai|vao)?\s*thang\s*(\d{1,2})/(\d{4})",
            raw,
        )
        if m and _has(t, "doanh thu", "dthu", "thu ve"):
            count = int(m.group(1))
            year, month = int(m.group(3)), int(m.group(2))
            if not 1 <= count <= 12:
                return Answer("abstain", plan="revenue_monthly_sequence")
            months = []
            for _ in range(count):
                months.append((year, month))
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
            months.reverse()
            values = []
            for y, mo in months:
                value = ds.revenue(resolve_period(f"thang {mo}/{y}", as_of))
                values.append(_round_money(value))
            return Answer(
                "list", values, confidence=0.75, plan="revenue_monthly_sequence"
            )

        # 10. Total revenue, only when no filter is present.
        if _has(
            t,
            "gia ban",
            "don gia",
            "gia bao nhieu",
            "bao nhieu mot",
            "gia nhap",
            "gia binh quan",
        ):
            return Answer("abstain", plan="price_question")
        if _has(raw, "%", "phan tram") or re.search(
            r"(?:bao nhieu|bao nhiu|bnhiu|may|mi|bn|nhiu|nhieu)\s*%", raw
        ):
            return Answer("abstain", plan="percent_or_comparison")

        if _has(
            t,
            "doanh thu",
            "dthu",
            "thu ve",
            "ban duoc bao nhieu tien",
            "ban dc bao nhieu tien",
            "tien ban",
            "nhiu tien",
            "bnhiu tien",
            "bn tien",
            "bnhieu tien",
        ) or (
            _has(t, "ban duoc", "ban dc", "ban ra")
            and _has(t, "bao nhieu tien", "bao nhiu tien", "bnhiu tien", "bn tien")
        ):
            if period is None:
                return Answer("abstain", plan="revenue")
            if _has(t, "nhap", "mua vao", "tien nhap"):
                return Answer("abstain", plan="revenue_purchase_guard")
            if _has(t, *_FILTER_WORDS):
                return Answer("abstain", plan="revenue_filtered")
            return Answer(
                "number",
                _round_money(ds.revenue(period)),
                confidence=0.85,
                plan="revenue_total",
            )

        return Answer("abstain", plan="unsupported")
