"""Text, number and date normalization for messy Vietnamese business data."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

import pandas as pd


_DIACRITICS = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)


def strip_diacritics(text: str) -> str:
    return text.translate(_DIACRITICS)


def _split_camel(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    # "8mg" and "8MG" must normalize the same way.
    return re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)


def norm_key(text: object) -> str:
    """Lowercase, diacritic-free, punctuation-free key for fuzzy matching."""
    s = strip_diacritics(_split_camel(str(text or ""))).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_col(name: object) -> str:
    """Normalize a column name for matching: snake_case, no diacritics."""
    s = strip_diacritics(_split_camel(str(name or ""))).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def parse_number(value: object) -> float | None:
    """Parse Vietnamese/English number strings. Returns None when not numeric.

    Handles: 1.234.567 ; 1,234,567 ; 560,000 ; 12,4 ; (123) ; 1 234 ; 4.82
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-", "--"}:
        return None
    lower = s.lower()
    # Vietnamese shorthand: 1tr440 = 1.44 million, 2tr5 = 2.5 million.
    short = re.match(r"^(-?\d+)\s*tr\s*(\d{1,3})$", lower)
    if short:
        sign = -1.0 if short.group(1).startswith("-") else 1.0
        left = abs(int(short.group(1)))
        right = short.group(2)
        return sign * (left + int(right) / (10 ** len(right))) * 1_000_000.0
    multiplier = 1.0
    if lower.endswith("k") or "nghìn" in lower or "ngan" in lower:
        multiplier = 1_000.0
    elif lower.endswith("tr") or "triệu" in lower or "trieu" in lower:
        multiplier = 1_000_000.0
    elif lower.endswith("cu"):
        # Slang "củ" (normalized to "cu") means one million VND.
        multiplier = 1_000_000.0
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace("−", "-").replace("–", "-").replace(" ", "").replace("\u00a0", "")
    s = re.sub(r"[^0-9,.\-]", "", s)
    if s in {"", "-", ".", ","}:
        return None
    # Decide decimal separator.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and (len(parts[1]) in {1, 2} or multiplier > 1):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # 1.234 is thousands in Vietnamese data; 12.4 stays decimal.
            s = s.replace(".", "")
    try:
        num = float(s)
    except ValueError:
        return None
    num *= multiplier
    return -num if negative else num


_DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d.%m.%y",
)


def parse_date(value: object) -> date | None:
    """Parse common Vietnamese/Excel dates. Returns None when not a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 20000 <= float(value) <= 60000:  # Excel serial date
            return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()
        return None
    s = str(value).strip()
    if s.isdigit() and 20000 <= int(s) <= 60000:
        return (datetime(1899, 12, 30) + timedelta(days=int(s))).date()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None
    s = s.replace("\u00a0", " ").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # "1 thg 6 2025", "1 tháng 6 năm 2025"
    m = re.match(
        r"^(\d{1,2})\s*(?:thg|tháng|thang)\s*(\d{1,2})[\s,/]*(?:năm|nam)?\s*(\d{4})$",
        s,
        re.I,
    )
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    # Excel sometimes stores dates as "2025-06-01 00:00:00"
    if " " in s:
        head = s.split(" ")[0]
        if head != s:
            return parse_date(head)
    return None


def parse_hour(value: object) -> int | None:
    """Hour of day from a timestamp or 'HH:MM' string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.hour
    s = str(value).strip()
    m = re.search(r"(\d{1,2}):\d{2}", s)
    if not m:
        return None
    hour = int(m.group(1))
    return hour if 0 <= hour <= 23 else None


def is_date_like(series: pd.Series, threshold: float = 0.8) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    good = sum(1 for v in sample if parse_date(v) is not None)
    return good / len(sample) >= threshold


def is_number_like(series: pd.Series, threshold: float = 0.9) -> bool:
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    good = sum(1 for v in sample if parse_number(v) is not None)
    return good / len(sample) >= threshold


def to_date_series(series: pd.Series) -> pd.Series:
    return series.map(parse_date)


def to_number_series(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype("float64")
