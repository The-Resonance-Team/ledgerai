"""Load CSV, Excel and SQLite files into pandas DataFrames."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import norm_col, parse_date, parse_number


_SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".sqlite", ".db"}


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    source: str
    sheet: str | None = None
    meta: dict = field(default_factory=dict)


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            with open(path, encoding=encoding, newline="") as fh:
                sample = fh.read(8192)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ","
            return _frame_from_raw(
                pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=sep,
                    header=None,
                    dtype=str,
                    keep_default_na=False,
                )
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise last_error or ValueError(f"Cannot read {path}")


_HEADER_WORDS = {
    "ma",
    "ten",
    "ngay",
    "so luong",
    "thanh tien",
    "trang thai",
    "gia",
    "don vi",
    "khach",
    "hang",
    "san pham",
    "nhap",
    "xuat",
    "ton",
    "ghi chu",
    "stt",
    "don gia",
    "chiet khau",
    "tong",
    "so phieu",
    "so hoa don",
    "ma hoa don",
    "ma hang",
    "ten hang",
    "so tien",
    "con no",
    "da tra",
}


def _header_score(row: list[str]) -> float:
    cells = [str(c).strip() for c in row if str(c).strip()]
    if len(cells) < 3:
        return -1.0
    score = 0.0
    for cell in cells:
        key = norm_col(cell).replace("_", " ")
        if any(word in key for word in _HEADER_WORDS):
            score += 2.0
        elif parse_date(cell) is not None or parse_number(cell) is not None:
            score -= 0.5
        else:
            score += 0.2
    return score


def _frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Find the header row in a sheet that may start with title rows."""
    best_idx, best_score = 0, -1.0
    for idx in range(min(15, len(raw))):
        score = _header_score(raw.iloc[idx].tolist())
        if score > best_score:
            best_idx, best_score = idx, score
    header = [str(c).strip() for c in raw.iloc[best_idx].tolist()]
    df = raw.iloc[best_idx + 1 :].copy()
    seen: dict[str, int] = {}
    columns = []
    for i, name in enumerate(header):
        base = name if name and name.lower() != "nan" else f"col_{i}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        columns.append(base if count == 1 else f"{base}_{count}")
    df.columns = columns
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def _read_excel(path: Path) -> list[Table]:
    tables: list[Table] = []
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        raw = book.parse(sheet, header=None, dtype=str, keep_default_na=False)
        if raw.empty:
            continue
        df = _frame_from_raw(raw)
        # Drop empty rows and columns produced by merged title cells.
        df = df.loc[
            :, [c for c in df.columns if df[c].astype(str).str.strip().ne("").any()]
        ]
        if df.empty:
            continue
        tables.append(Table(name=path.stem, df=df, source=str(path), sheet=str(sheet)))
    return tables


def _read_sqlite(path: Path) -> list[Table]:
    tables: list[Table] = []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for name in names:
            df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn, dtype=str)
            if not df.empty:
                tables.append(Table(name=name, df=df, source=str(path)))
    finally:
        conn.close()
    return tables


def load_business(biz_dir: Path) -> list[Table]:
    """Load every supported file under a business directory."""
    tables: list[Table] = []
    for path in sorted(Path(biz_dir).iterdir()):
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED or path.name.startswith("."):
            continue
        if suffix in {".csv", ".tsv"}:
            df = _read_csv(path)
            tables.append(Table(name=path.stem, df=df, source=str(path)))
        elif suffix in {".xlsx", ".xls"}:
            tables.extend(_read_excel(path))
        elif suffix in {".sqlite", ".db"}:
            tables.extend(_read_sqlite(path))
    for table in tables:
        table.df.columns = [norm_col(c) for c in table.df.columns]
    return tables
