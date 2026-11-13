"""Resolve Vietnamese time expressions against an as-of date."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from .normalize import norm_key, strip_diacritics


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str = ""

    def contains(self, d: date | None) -> bool:
        if d is None:
            return False
        try:
            return self.start <= d <= self.end
        except TypeError:
            return False

    def days(self) -> int:
        return (self.end - self.start).days + 1


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


# Lunar New Year (mùng 1 Tết). The gold convention for "dịp Tết" is the
# 17-day window [Tết − 7, Tết + 9] (23 tháng Chạp → mùng 9 Tết), verified
# exact against train gold on both YoY and prior-window comparisons.
TET_DATES = {
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
}

TET_PRE_DAYS = 7
TET_POST_DAYS = 9


def tet_period(year: int) -> Period | None:
    """The 17-day Tết season window for a lunar year."""
    new_year = TET_DATES.get(year)
    if new_year is None:
        return None
    return Period(
        new_year - timedelta(days=TET_PRE_DAYS),
        new_year + timedelta(days=TET_POST_DAYS),
        f"dịp tết {year}",
    )


def is_tet_period(period: Period | None) -> bool:
    if period is None:
        return False
    for year in TET_DATES:
        expected = tet_period(year)
        if (
            expected is not None
            and period.start == expected.start
            and period.end == expected.end
        ):
            return True
    return False


def _month_range(year: int, month: int, today: date, clamp: bool = False) -> Period:
    start = date(year, month, 1)
    end = _month_end(year, month)
    if clamp and start <= today < end:
        end = today
    return Period(start, end, f"tháng {month:02d}/{year}")


def _prev_month(period: Period) -> Period:
    """Previous month with the same day span when the period is partial."""
    if period.start.day == 1:
        y, m = (
            (period.start.year, period.start.month - 1)
            if period.start.month > 1
            else (period.start.year - 1, 12)
        )
        last_day = _month_end(y, m).day
        # Full month -> full previous month. Partial month -> same day count.
        current_end = _month_end(period.start.year, period.start.month)
        if period.end == current_end:
            end_day = last_day
        else:
            end_day = min(period.end.day, last_day)
        return Period(date(y, m, 1), date(y, m, end_day), "tháng trước")
    prev_end = period.start - timedelta(days=1)
    span = period.days()
    prev_start = prev_end - timedelta(days=span - 1)
    return Period(prev_start, prev_end, "kỳ trước")


def previous_period(period: Period) -> Period:
    """Previous comparable period: same length, ending the day before."""
    return _prev_month(period)


def _parse_day_month(text: str, today: date) -> tuple[int, int, int] | None:
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{4}))?", text)
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
    year = int(m.group(3)) if m.group(3) else today.year
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    if not m.group(3) and date(year, month, min(day, 28)) > today:
        year -= 1
    return day, month, year  # type: ignore[return-value]


def _parse_month_year(text: str, today: date) -> tuple[int, int] | None:
    m = re.search(r"thang\s*(\d{1,2})\s*(?:[/\-.]\s*|\s+(?:nam\s+)?)(\d{4})", text)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        return (year, month) if 1 <= month <= 12 else None
    m = re.search(r"thang\s*(\d{1,2})", text)
    if not m:
        return None
    month = int(m.group(1))
    if month < 1 or month > 12:
        return None
    year = today.year
    if date(year, month, 1) > today:
        year -= 1
    return year, month


def extract_month_refs(text: str, today: date) -> list[tuple[int, int]]:
    """Find all month references such as '4/2026', 'tháng 5/2026'.

    Also accepts dash separators ('11-2025'), 2-digit years ('2/26'),
    and bare 'tháng X' mentions, which take the year of the nearest dated
    reference in the same string (question lists like 'th8, th9, th10/2025').
    """
    dated: list[tuple[int, int, int]] = []  # (pos, year, month)
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(?:thang\s*)?(\d{1,2})\s*[/\-]\s*(\d{4}|\d{2})(?!\d)", text):
        month, year_raw = int(m.group(1)), m.group(2)
        year = int(year_raw) if len(year_raw) == 4 else 2000 + int(year_raw)
        if 1 <= month <= 12:
            dated.append((m.start(), year, month))
            spans.append((m.start(), m.end()))
    ordered: list[tuple[int, tuple[int, int]]] = [
        (pos, (year, month)) for pos, year, month in dated
    ]
    for m in re.finditer(r"th(?:ang)?\s*(\d{1,2})(?![\d\s]*[/\-])", text):
        month = int(m.group(1))
        if not 1 <= month <= 12:
            continue
        # Skip months already inside a dated reference span.
        if any(start <= m.start() < end for start, end in spans):
            continue
        if dated:
            year = min(dated, key=lambda item: abs(item[0] - m.start()))[1]
        else:
            year = today.year
            if date(year, month, 1) > today:
                year -= 1
        ordered.append((m.start(), (year, month)))
    ordered.sort(key=lambda item: item[0])
    return [ym for _, ym in ordered]


def resolve_period(text: str, as_of: date) -> Period | None:
    """Best-effort resolution of a Vietnamese time expression."""
    t = strip_diacritics(str(text or "")).lower()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # Common abbreviations and typos in the train questions.
    t = re.sub(r"\bhnay\b|\bhn\b", "hom nay", t)
    t = re.sub(r"\bhqua\b|\bhwa\b|\bbua qua\b|\bbua hom qua\b|\bhom wa\b", "hom qua", t)
    t = re.sub(r"\btuan trc\b|\btuan truoc\b|\btuan roi\b", "tuan truoc", t)
    t = re.sub(r"\bthang trc\b|\bthang roi\b", "thang truoc", t)
    t = re.sub(r"\bthangs\b|\bthangf\b", "thang", t)
    t = re.sub(r"\bnawm\b|\bnamf\b", "nam", t)
    t = re.sub(r"\bbua nay\b|\bhom ni\b|\bhnay\b", "hom nay", t)
    t = re.sub(r"\bthang ni\b", "thang nay", t)
    t = re.sub(r"\bth\s+nay\b", "thang nay", t)
    t = re.sub(r"\bth\s+truoc\b|\bth\s+roi\b", "thang truoc", t)
    t = re.sub(r"\bnam ni\b", "nam nay", t)
    today = as_of

    # "Từ đầu tháng (này) đến hết hôm nay" is month-to-date, not today.
    if re.search(
        r"tu\s+(?:dau thang(?: nay)?|mung 1 thang(?: nay)?|ngay 1 thang(?: nay)?)"
        r".*(?:den|toi)\s+(?:het\s+)?(?:hom nay|hnay|bay gio)",
        t,
    ):
        return Period(date(today.year, today.month, 1), today, "từ đầu tháng")

    # Range "tu X den Y" / "tu X qua Y". Must run before "hom nay" so that
    # "tu ngay 1/1 den het hom nay" keeps both ends.
    range_match = re.search(r"tu\s+(.+?)\s+(?:den|toi|qua)\s+(.+)$", t)
    if range_match:
        left = resolve_period(range_match.group(1), today)
        right = resolve_period(range_match.group(2), today)
        if left and right:
            return Period(
                min(left.start, right.start), max(left.end, right.end), "khoảng"
            )

    # "Tu dau nam den hom nay" is a range, not a single day.
    if "dau nam" in t or "tu dau nam" in t:
        return Period(date(today.year, 1, 1), today, "từ đầu năm")

    # "Dịp Tết (năm ...)": the 17-day lunar season [Tết − 7, Tết + 9].
    # Must run before year/month parsing ("tet 2026" has a bare year).
    # The anchor is the CURRENT Tet when "năm nay" appears anywhere, even
    # alongside "năm trước" (which then describes the comparison base).
    if re.search(r"(?<![a-z0-9])tet(?![a-z0-9])", t):
        year_match = re.search(r"\b(20[0-2]\d)\b", t)
        if year_match:
            year = int(year_match.group(1))
        elif re.search(r"nam\s+nay", t):
            year = today.year
        elif re.search(
            r"tet\s+(?:nam\s+)?(?:truoc|ngoai|roi|trc)|nam\s+(?:truoc|ngoai|roi)\s+(?:tet|den tet)",
            t,
        ):
            year = today.year - 1
        else:
            year = today.year
        tet = tet_period(year)
        if tet is not None:
            return tet

    if "hom nay" in t:
        return Period(today, today, "hôm nay")
    if "hom qua" in t:
        d = today - timedelta(days=1)
        return Period(d, d, "hôm qua")

    # Week.
    if "tuan nay" in t:
        monday = today - timedelta(days=today.weekday())
        return Period(monday, today, "tuần này")
    if "tuan truoc" in t or "tuan roi" in t:
        this_monday = today - timedelta(days=today.weekday())
        return Period(
            this_monday - timedelta(days=7),
            this_monday - timedelta(days=1),
            "tuần trước",
        )

    # Quarter ("quý 2/2025", "Q1/2026") must be parsed before M/YYYY.
    q = re.search(
        r"(?:quy|\bq)\s*(\d)(?:\s*/\s*(\d{4})|\s+nam\s+(\d{4})|\s+(\d{4}))?", t
    )
    if q:
        quarter = int(q.group(1))
        year_text = q.group(2) or q.group(3) or q.group(4)
        year = int(year_text) if year_text else today.year
        if not year_text and date(year, (quarter - 1) * 3 + 1, 1) > today:
            year -= 1
        start = date(year, (quarter - 1) * 3 + 1, 1)
        end = date(year, quarter * 3, _month_end(year, quarter * 3).day)
        return Period(start, end, f"quý {quarter}/{year}")

    # Multiple explicit months form a range, e.g. "tháng 11/2025, 12/2025 và 1/2026".
    refs = extract_month_refs(t, today)
    if len(refs) >= 2:
        # Year-over-year phrasing ("T4/2025 ... T4/2026") points at the later
        # month, not a 13-month range.
        if len({month for _, month in refs}) == 1 and any(
            phrase in t
            for phrase in ("so voi", "so vs", "cung ky", "tang truong", "thay doi")
        ):
            year, month = max(refs, key=lambda ref: ref[0])
            return _month_range(year, month, today)
        starts = [date(y, m, 1) for y, m in refs]
        ends = [_month_end(y, m) for y, m in refs]
        return Period(min(starts), max(ends), "khoảng tháng")

    # Short month forms such as "Th3", "th 5" (no year).
    short_month = re.search(r"\bth\s*(\d{1,2})\b(?!\s*[/-]\s*\d)", t)
    if short_month:
        month = int(short_month.group(1))
        if 1 <= month <= 12:
            year = today.year
            if date(year, month, 1) > today:
                year -= 1
            return _month_range(year, month, today)

    # Month names without a year ("tháng ba", "tháng mười hai").
    month_names = {
        "mot": 1,
        "hai": 2,
        "ba": 3,
        "bon": 4,
        "tu": 4,
        "nam": 5,
        "sau": 6,
        "bay": 7,
        "tam": 8,
        "chin": 9,
        "muoi": 10,
        "muoi mot": 11,
        "muoi hai": 12,
    }
    month_match = re.search(
        r"thang\s+(muoi mot|muoi hai|mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi)"
        r"(?:\s+(?:nam\s+)?(\d{4}))?",
        t,
    )
    if month_match and not re.search(r"thang\s+\d", t):
        month = month_names[month_match.group(1)]
        if month_match.group(2):
            year = int(month_match.group(2))
        else:
            year = today.year
            if date(year, month, 1) > today:
                year -= 1
        return _month_range(year, month, today)

    # Explicit month references win over "tháng này / tháng trước".
    bare = re.search(
        r"(?<![\d/])(?:t|th)?\s*(\d{1,2})\s*[/-]\s*(\d{4}|\d{2})(?![\d/])",
        t,
    )
    if bare:
        month = int(bare.group(1))
        year = int(bare.group(2))
        if year < 100:
            year += 2000
        if 1 <= month <= 12:
            return _month_range(year, month, today)

    my = _parse_month_year(t, today)

    if my:
        return _month_range(my[0], my[1], today)

    # Month.
    if "thang nay" in t:
        return _month_range(today.year, today.month, today, clamp=True)
    if "thang truoc" in t or "thang roi" in t:
        y, m = (
            (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        )
        return Period(date(y, m, 1), _month_end(y, m), "tháng trước")

    # Year.
    if "nam nay" in t:
        return Period(date(today.year, 1, 1), today, "từ đầu năm")

    # N days.
    nd = re.search(r"(\d+)\s*ngay\s*(?:gan nhat|truoc|qua khu)?", t)
    if nd:
        n = int(nd.group(1))
        if n > 0:
            return Period(today - timedelta(days=n - 1), today, f"{n} ngày gần nhất")

    # Explicit day.
    dm = _parse_day_month(t, today)
    if dm:
        day, month, year = dm
        try:
            d = date(year, month, day)
        except ValueError:
            return None
        return Period(d, d, d.isoformat())

    return None


def period_for_question(text: str, as_of: date) -> Period | None:
    return resolve_period(text, as_of)
