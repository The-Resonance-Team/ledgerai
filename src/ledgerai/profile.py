"""Infer table and column roles for a business data package."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .loaders import Table
from .normalize import is_date_like, is_number_like, norm_key, parse_date, parse_number

# Column-name hints, checked in order. The first match wins.
_ROLE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    (
        "order_id",
        (
            "ma hoa don",
            "ma don",
            "ma bill",
            "bill",
            "mahd",
            "ma hd",
            "hd",
            "order_id",
            "order id",
            "so hoa don",
            "so don",
            "don so",
            "so ct",
            "so phieu",
            "ma phieu",
            "ma chung tu",
            "so chung tu",
        ),
    ),
    ("purchase_id", ("ma phieu nhap", "ma nhap", "purchase_id", "so phieu nhap")),
    (
        "product_id",
        (
            "ma san pham",
            "ma hang",
            "ma mat hang",
            "ma h hoa",
            "ma hang hoa",
            "ma hang ban",
            "ma san pham ban",
            "ma sp",
            "id sp",
            "ma thuoc",
            "product_id",
            "sku",
        ),
    ),
    ("customer_id", ("ma khach", "ma kh", "customer_id")),
    ("supplier_id", ("ma nha cung cap", "ma ncc", "supplier_id")),
    ("employee_id", ("ma nhan vien", "ma nv", "employee_id")),
    ("date", ("ngay", "thoi gian", "date", "created_at", "created", "transaction")),
    ("status", ("trang thai", "status", "ket qua", "tinh trang", "xu ly", "ghi chu tt", "ghi chu trang thai")),
    ("channel", ("kenh ban", "kenh", "channel", "hinh thuc ban", "nguon")),
    ("paid", ("da tra", "da thanh toan", "khach da tra", "khach tra", "paid")),
    (
        "payment",
        (
            "hinh thuc thanh toan",
            "phuong thuc thanh toan",
            "phuong thuc",
            "cach thanh toan",
            "loai tt",
            "pttt",
        ),
    ),
    (
        "debt",
        ("chua tra", "con no", "cong no", "phai thu", "con phai thu", "no", "debt"),
    ),
    (
        "total",
        ("tong tien", "tong cong", "phai tra", "can tra", "total"),
    ),
    (
        "amount",
        (
            "thanh tien",
            "doanh thu",
            "tien hang",
            "tien ban",
            "tri gia",
            "so tien",
            "tong bill",
            "tong don",
            "tien chi",
            "amount",
            "gia tri",
        ),
    ),
    ("cost", ("gia von", "cost", "lai gop")),
    ("discount", ("giam gia", "chiet khau", "discount")),
    ("price", ("gia ban", "don gia", "gia", "price")),
    (
        "base_quantity",
        ("quy doi", "quy ra", "so luong le", "sl ban", "so luong co ban"),
    ),
    ("quantity", ("so luong", "quy ra", "quantity", "sl", "qty")),
    ("unit", ("don vi", "unit", "dvt")),
    ("category", ("nhom hang", "loai hang", "loai san pham", "nhom", "danh muc", "category")),
    (
        "product",
        (
            "ten hang",
            "ten sp",
            "noi dung",
            "san pham",
            "mat hang",
            "hang hoa",
            "hang ban",
            "hang",
            "thuoc",
            "lam gi",
            "product",
            "item",
        ),
    ),
    (
        "customer",
        (
            "khach hang",
            "khach",
            "ten kh",
            "ten khach",
            "doi tac",
            "ben mua",
            "nguoi mua",
            "customer",
        ),
    ),
    ("supplier", ("nha cung cap", "ncc", "supplier")),
    ("employee", ("nhan vien", "nguoi ban", "employee", "staff", "tho", "ho ten")),
    ("phone", ("dien thoai", "sdt", "phone")),
    ("address", ("dia chi", "address")),
    ("note", ("ghi chu", "note", "dien giai")),
]


@dataclass
class ColumnProfile:
    name: str
    role: str = "other"
    kind: str = "text"
    missing: float = 0.0
    unique: int = 0
    minimum: object = None
    maximum: object = None
    sample: list = field(default_factory=list)


@dataclass
class TableProfile:
    name: str
    source: str
    sheet: str | None
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    roles: dict = field(default_factory=dict)
    date_range: list = field(default_factory=list)


@dataclass
class BusinessProfile:
    business_id: str
    tables: list[TableProfile] = field(default_factory=list)


def _matches_hint(key: str, hint: str) -> bool:
    """Match a hint phrase on word boundaries (not substrings)."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", key))


def _hint_role(name: str, series: pd.Series) -> str:
    key = norm_key(name)
    # Conversion tables: "Đơn vị quy đổi" is the pack unit, not a quantity.
    if "ti le quy doi" in key or "he so quy doi" in key or "ty le quy doi" in key:
        return "factor"
    if "don vi quy doi" in key or "don vi dong goi" in key:
        return "pack_unit"
    if "don vi co ban" in key:
        return "base_unit"
    # "nhóm hàng hóa" is a category, not a product name.
    if "nhom hang" in key or key.startswith("nhom"):
        return "category"
    if key == "tien" or key == "tt":
        return "amount"
    unique = series.replace("", pd.NA).dropna().nunique()
    for role, hints in _ROLE_HINTS:
        if any(_matches_hint(key, h) for h in hints):
            # A product hint on a low-cardinality flag column is a false positive.
            if role == "product" and unique <= 3:
                continue
            if role == "product" and ("khach" in key or "customer" in key):
                continue
            return role
    if is_date_like(series):
        return "date"
    if is_number_like(series, threshold=0.95):
        return "number"
    return "other"


def _kind(series: pd.Series, role: str) -> str:
    if role in {"date"}:
        return "date"
    if role in {"amount", "price", "cost", "debt", "quantity", "number", "discount"}:
        return "number"
    if is_date_like(series):
        return "date"
    if is_number_like(series, threshold=0.95):
        return "number"
    if series.nunique(dropna=True) <= max(20, len(series) // 100):
        return "categorical"
    return "text"


def profile_table(table: Table, max_sample: int = 5) -> TableProfile:
    df = table.df
    rows = len(df)
    columns: list[ColumnProfile] = []
    roles: dict = {}
    dates: list = []
    for col in df.columns:
        series = df[col]
        role = _hint_role(col, series)
        kind = _kind(series, role)
        nonnull = series.replace("", pd.NA).dropna()
        col_profile = ColumnProfile(
            name=col,
            role=role,
            kind=kind,
            missing=round(1 - (len(nonnull) / rows), 4) if rows else 0.0,
            unique=int(nonnull.nunique()) if len(nonnull) else 0,
            sample=[str(v)[:80] for v in nonnull.head(max_sample).tolist()],
        )
        if kind == "date" and len(nonnull):
            parsed = [parse_date(v) for v in nonnull.head(500)]
            parsed = [d for d in parsed if d]
            if parsed:
                col_profile.minimum = min(parsed).isoformat()
                col_profile.maximum = max(parsed).isoformat()
                dates.extend(parsed)
        elif kind == "number" and len(nonnull):
            nums = [parse_number(v) for v in nonnull.head(500)]
            nums = [n for n in nums if n is not None]
            if nums:
                col_profile.minimum = min(nums)
                col_profile.maximum = max(nums)
        if role != "other":
            if role not in roles:
                roles[role] = col
            elif role == "date":
                # Prefer a full date column ("ngay...") over a time-only one ("gio").
                old_key = norm_key(roles[role])
                new_key = norm_key(col)
                if "ngay" in new_key and "ngay" not in old_key:
                    roles[role] = col
            elif role == "category":
                # Prefer the business group ("nhóm hàng") over the generic type.
                old_key = norm_key(roles[role])
                new_key = norm_key(col)
                if ("nhom" in new_key or "loai san pham" in new_key) and not (
                    "nhom" in old_key or "loai san pham" in old_key
                ):
                    roles[role] = col
            elif role == "amount":
                # Prefer the net line total over pre-discount gross columns.
                old_key = norm_key(roles[role])
                new_key = norm_key(col)
                preferred = ("thanh tien", "sau giam", "phai thu", "tong tien")
                if any(p in new_key for p in preferred) and not any(
                    p in old_key for p in preferred
                ):
                    roles[role] = col
        columns.append(col_profile)
    return TableProfile(
        name=table.name,
        source=table.source,
        sheet=table.sheet,
        rows=rows,
        columns=columns,
        roles=roles,
        date_range=[min(dates).isoformat(), max(dates).isoformat()] if dates else [],
    )


def infer_table_role(table: TableProfile) -> str:
    name = norm_key(f"{table.name} {table.sheet or ''}")
    words = set(name.split())
    roles = set(table.roles)
    column_keys = {norm_key(column.name) for column in table.columns}
    if (
        "ti le quy doi" in column_keys
        or "he so quy doi" in column_keys
        or ("quy" in words and "doi" in words and "product" in roles)
    ):
        return "conversions"
    if (
        any("ton cuoi" in key for key in column_keys)
        and ("kho" in words or "ton" in words)
        and ("product_id" in roles or "product" in roles)
    ):
        return "inventory"
    if "ton kho" in name or "inventory" in name or "ton" in words or "toncuoi" in words:
        return "inventory"
    if "nhap" in name or "purchase" in name:
        return "purchase_lines" if "product" in roles else "purchases"
    if "thu may cu" in name or "tradein" in name:
        return "tradeins"
    if "bao hanh" in name or "warranty" in name:
        return "warranty"
    if "chi phi" in name or "expense" in name:
        return "expenses"
    if "thanh toan" in name or "payment" in name or "tra no" in name:
        return "payments"
    if "san pham" in name or "thuoc" in name or "product" in name or "danh muc" in name:
        return "products"
    if "khach" in name or "customer" in name:
        return "customers"
    if "nhan vien" in name or "employee" in name or "tho" in words:
        return "employees"
    if "product" in roles and ("quantity" in roles or "base_quantity" in roles):
        return "order_lines"
    if "product_id" in roles and ("quantity" in roles or "base_quantity" in roles):
        return "order_lines"
    if "product" in roles and "category" in roles:
        return "products"
    if "order_id" in roles and ("amount" in roles or "total" in roles):
        return "orders"
    if "date" in roles and "amount" in roles:
        return "orders"
    return "other"


def profile_business(biz_dir: Path, tables: list[Table]) -> BusinessProfile:
    profiles = [profile_table(t) for t in tables]
    return BusinessProfile(business_id=Path(biz_dir).name, tables=profiles)


def save_profile(profile: BusinessProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8"
    )
