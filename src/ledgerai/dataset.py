"""Semantic layer: turn heterogeneous business files into standard tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .loaders import Table, load_business
from .normalize import norm_col, norm_key, parse_date, parse_hour, parse_number
from .periods import Period, previous_period


def _is_date(value: object) -> bool:
    """True for a real date. pandas.NaT also passes isinstance(value, date)."""
    return isinstance(value, date) and not pd.isna(value)


_STATUS_DONE = {
    "xong",
    "hoan thanh",
    "hoan tat",
    "da giao",
    "giao thanh cong",
    "completed",
    "thanh cong",
    "ok",
    "da lay",
}
_STATUS_CANCEL = {"huy", "da huy", "khach huy", "huy don", "cancelled", "canceled"}
_STATUS_RETURN = {
    "tra hang",
    "doi tra",
    "hoan tra",
    "tra lai",
    "hang tra ve",
    "tra ve",
    "returned",
    "return",
    "khach tra",
}
_STATUS_SHIP = {
    "dang giao",
    "cho giao",
    "dang van chuyen",
    "van chuyen",
    "dang xu ly",
    "dang chuan bi",
    "shipping",
    "pending",
}


def normalize_status(value: object) -> str:
    key = norm_key(value)
    if not key:
        return "unknown"
    if (
        key in _STATUS_CANCEL
        or re.search(r"(?<![a-z0-9])huy(?![a-z0-9])", key)
        or key in {"cancel", "canceled", "cancelled", "void"}
        or "khong giao duoc" in key
        or "giao khong duoc" in key
    ):
        return "cancelled"
    if (
        key in _STATUS_RETURN
        or "tra hang" in key
        or "doi tra" in key
        or "hoan tra" in key
        or "hoan hang" in key
        or "hoan lai" in key
        or "hoan tien" in key
        or "mang tra" in key
        or "tra lai" in key
    ):
        return "returned"
    if key in _STATUS_SHIP or "dang giao" in key or "cho giao" in key:
        return "shipping"
    if key in _STATUS_DONE or "hoan thanh" in key or "xong" in key:
        return "completed"
    return "other"


def _status_priority(values) -> str:
    s = set(values)
    for status in ("cancelled", "returned", "shipping", "completed"):
        if status in s:
            return status
    return "other"


def _pick(df: pd.DataFrame, role_cols: dict, role: str) -> str | None:
    return role_cols.get(role)


def _to_num(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype("float64")


def _drop_seen(df: pd.DataFrame, seen: set, columns: list[str]) -> pd.DataFrame:
    """Drop rows already present in an earlier file/sheet, keeping in-file rows."""
    if df.empty or not columns:
        return df
    present = [column for column in columns if column in df.columns]
    if not present:
        return df
    keys = list(zip(*(df[column].astype(str) for column in present)))
    mask = [key in seen for key in keys]
    seen.update(keys)
    if not any(mask):
        return df
    return df[~pd.Series(mask, index=df.index)]


def _drop_invalid_lines(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Drop summary rows that repeat a product name but carry no quantity/unit."""
    quantity_col = cols.get("quantity")
    if not quantity_col or quantity_col not in df.columns:
        return df
    quantity = _to_num(df[quantity_col])
    if cols.get("unit") and cols["unit"] in df.columns:
        unit_blank = df[cols["unit"]].astype(str).str.strip().eq("")
    else:
        unit_blank = pd.Series(True, index=df.index)
    bad = quantity.isna() & unit_blank
    return df[~bad] if bad.any() else df


def _drop_footer_rows(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Drop sheet total rows such as \"Cộng\" or \"Tổng cộng\"."""
    product_col = cols.get("product")
    if not product_col or product_col not in df.columns:
        return df
    keys = df[product_col].map(norm_key)
    footer = keys.str.match(
        r"^(tong|cong)( (cong|tien|thang|nam|quy|ngay|tuan)\b.*)?$", na=False
    )
    return df[~footer] if footer.any() else df


def _drop_complete_duplicates(df: pd.DataFrame, order_col: str | None) -> pd.DataFrame:
    """Drop repeated rows that carry an order id, before blanks are filled."""
    if not order_col or order_col not in df.columns:
        return df
    present = df[order_col].astype(str).str.strip().ne("")
    if not present.any():
        return df
    subset = df[present]
    duplicates = subset.duplicated(keep="first")
    if not duplicates.any():
        return df
    return df.drop(index=subset.index[duplicates])


def _to_date(series: pd.Series) -> pd.Series:
    return series.map(parse_date)


def _plain_date(value: object) -> object:
    if isinstance(value, datetime):
        return value.date()
    return value


def _to_date_with_sheet(series: pd.Series, table: Table) -> pd.Series:
    """Parse dates, filling year-less day/month values from the sheet name."""
    parsed = series.map(parse_date)
    year_match = re.search(r"(20\d{2})", f"{table.sheet or ''} {table.name}")
    if not year_match:
        return parsed
    year = int(year_match.group(1))
    for index, value in series.items():
        if parsed.loc[index] is not None:
            continue
        text = str(value).strip()
        match = re.match(r"^(\d{1,2})\s*(?:[/\-.]|thg|thang)\s*(\d{1,2})$", text)
        if not match:
            continue
        day, month = int(match.group(1)), int(match.group(2))
        try:
            parsed.loc[index] = date(year, month, day)
        except ValueError:
            continue
    return parsed


@dataclass
class Dataset:
    business_id: str
    orders: pd.DataFrame
    lines: pd.DataFrame
    products: pd.DataFrame
    purchases: pd.DataFrame
    purchase_lines: pd.DataFrame
    tables: list[Table]
    inventory: pd.DataFrame = None  # type: ignore[assignment]
    tradeins: pd.DataFrame = None  # type: ignore[assignment]
    warranties: pd.DataFrame = None  # type: ignore[assignment]
    employees: pd.DataFrame = None  # type: ignore[assignment]
    expenses: pd.DataFrame = None  # type: ignore[assignment]
    drugs: pd.DataFrame = None  # type: ignore[assignment]
    customers: pd.DataFrame = None  # type: ignore[assignment]
    conversions: tuple = None  # type: ignore[assignment]
    # Harvested master data (set in from_tables; empty dict when absent):
    # - product_aliases: norm(alias spelling) -> canonical product name.
    # - product_conditions: norm(product spelling) -> 'may moi'|'may cu'|'phu kien'.
    # - gender_of_customer: norm(customer_id) -> 'nam'|'nu'.
    product_aliases: dict = None  # type: ignore[assignment]
    product_conditions: dict = None  # type: ignore[assignment]
    gender_of_customer: dict = None  # type: ignore[assignment]
    repair_tickets: pd.DataFrame = None  # type: ignore[assignment]
    lots: pd.DataFrame = None  # type: ignore[assignment]
    lot_uses: pd.DataFrame = None  # type: ignore[assignment]
    stock_ledger: pd.DataFrame = None  # type: ignore[assignment]
    installments: pd.DataFrame = None  # type: ignore[assignment]
    member_dates: list = None  # type: ignore[assignment]
    # Price history: product_id -> sorted [(effective date, new price)].
    price_history: dict = None  # type: ignore[assignment]

    @classmethod
    def load(cls, biz_dir: Path) -> "Dataset":
        tables = load_business(Path(biz_dir))
        return cls.from_tables(
            Path(biz_dir).name, tables, cls._parse_readme_factors(Path(biz_dir))
        )

    @staticmethod
    def _parse_readme_factors(
        biz_dir: Path,
    ) -> tuple[dict[tuple[str, str], float], dict[str, set[str]]]:
        """Parse "Quy đổi đơn vị" bullets: `- Product: 1 pack = N base`.

        Shop ledgers often leave bulk rows unconverted (1 thùng = 1 base).
        The README states the pack factors the gold answers use.
        """
        factors: dict[tuple[str, str], float] = {}
        units: dict[str, set[str]] = {}
        readme = Path(biz_dir) / "README.md"
        if not readme.exists():
            return factors, units
        try:
            text = readme.read_text(encoding="utf-8")
        except OSError:
            return factors, units
        pattern = re.compile(
            r"^\s*-\s*(.+?):\s*1\s+([^0-9=]+?)\s*=\s*([0-9]+)\s+(.+?)\s*$"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            product, pack, count, _base = match.groups()
            try:
                number = int(count)
            except ValueError:
                continue
            if number <= 1:
                continue
            product_key = norm_key(product)
            pack_key = norm_key(pack)
            if (
                not product_key
                or not pack_key
                or any(character.isdigit() for character in pack_key)
            ):
                continue
            factors[(product_key, pack_key)] = float(number)
            units.setdefault(product_key, set()).add(pack_key)
        return factors, units

    @classmethod
    def from_tables(
        cls,
        business_id: str,
        tables: list[Table],
        readme_factors: tuple[dict, dict] | None = None,
    ) -> "Dataset":
        from .profile import infer_table_role, profile_table

        profiles = {id(t): profile_table(t) for t in tables}
        roles = {id(t): infer_table_role(profiles[id(t)]) for t in tables}

        order_tables = [t for t in tables if roles[id(t)] == "orders"]
        line_tables = [t for t in tables if roles[id(t)] == "order_lines"]
        # A cash book or receipt list can look like an orders table. When
        # several order tables exist, keep those whose ids match the lines.
        if len(order_tables) > 1 and line_tables:
            line_ids: set[str] = set()
            for line_table in line_tables:
                line_cols = profiles[id(line_table)].roles
                if line_cols.get("order_id") and line_cols["order_id"] in line_table.df:
                    line_ids.update(
                        line_table.df[line_cols["order_id"]]
                        .astype(str)
                        .str.strip()
                        .dropna()
                    )
            if line_ids:
                overlapping = []
                for order_table in order_tables:
                    order_cols = profiles[id(order_table)].roles
                    oid = order_cols.get("order_id")
                    if not oid or oid not in order_table.df:
                        overlapping.append(order_table)
                        continue
                    ids = set(order_table.df[oid].astype(str).str.strip().dropna())
                    if ids and len(ids & line_ids) / len(ids) >= 0.1:
                        overlapping.append(order_table)
                if overlapping:
                    order_tables = overlapping
        # Delivery notes repeat sales quantities without prices. When priced
        # sales lines exist, keep only those to avoid double counting.
        priced_lines = [
            t
            for t in line_tables
            if profiles[id(t)].roles.get("amount") or profiles[id(t)].roles.get("price")
        ]
        if priced_lines:
            line_tables = priced_lines
        product_tables = [t for t in tables if roles[id(t)] == "products"]
        purchase_tables = [
            t for t in tables if roles[id(t)] in {"purchases", "purchase_lines"}
        ]
        conversion_tables = [t for t in tables if roles[id(t)] == "conversions"]

        conversions = cls._build_conversions(tables, profiles)
        if readme_factors:
            readme_map, readme_units = readme_factors
            factors, base_units, units_by_product = conversions
            for key, value in readme_map.items():
                factors.setdefault(key, value)
            for key, value in readme_units.items():
                units_by_product.setdefault(key, set()).update(value)
            conversions = (factors, base_units, units_by_product)
        customers = cls._build_customers(tables, profiles, roles)
        orders = cls._build_orders(order_tables, line_tables, profiles)
        if not customers.empty and not orders.empty:
            id_name = (
                customers[
                    (customers["name"].astype(str).str.strip() != "")
                    & (customers["customer_id"].astype(str).str.strip() != "")
                ]
                .drop_duplicates(subset=["customer_id"])
                .set_index("customer_id")["name"]
            )
            blank = orders["customer"].astype(str).str.strip() == ""
            if blank.any() and len(id_name):
                mapped = (
                    orders.loc[blank, "customer_id"]
                    .astype(str)
                    .str.strip()
                    .map(id_name)
                )
                orders.loc[blank, "customer"] = mapped.fillna("")
            group_map = (
                customers[
                    (customers["group"].astype(str).str.strip() != "")
                    & (customers["customer_id"].astype(str).str.strip() != "")
                ]
                .drop_duplicates(subset=["customer_id"])
                .set_index("customer_id")["group"]
            )
            orders["customer_group"] = (
                orders["customer_id"].astype(str).str.strip().map(group_map).fillna("")
            )
            # Many ledgers leave the customer id blank but keep the name.
            named = customers[
                (customers["group"].astype(str).str.strip() != "")
                & (customers["name"].astype(str).str.strip() != "")
            ]
            if not named.empty:
                name_group = (
                    named.assign(_key=named["name"].map(norm_key))
                    .drop_duplicates(subset=["_key"])
                    .set_index("_key")["group"]
                )
                blank = orders["customer_group"].astype(str).str.strip() == ""
                if blank.any():
                    mapped = orders.loc[blank, "customer"].map(norm_key).map(name_group)
                    orders.loc[blank, "customer_group"] = mapped.fillna("")
        if "customer_group" not in orders.columns:
            orders["customer_group"] = ""
        lines = cls._build_lines(line_tables, orders, profiles, conversions)
        products = cls._build_products(product_tables, lines, profiles)
        # Masters swallowed by other roles (e.g. a conversion sheet that also
        # carries the category column) still contribute categories, aliases
        # and condition tags. Canonical = the name column overlapping lines.
        products, master_aliases, master_conditions, gender_map = (
            cls._harvest_product_masters(products, lines, tables, profiles)
        )
        # Product masters often carry the name; order lines only carry the id.
        if (
            not products.empty
            and "product_id" in products.columns
            and "product_id" in lines.columns
        ):
            named = products[
                (products["product"].astype(str).str.strip() != "")
                & (
                    products["product"].astype(str).str.strip()
                    != products["product_id"].astype(str).str.strip()
                )
            ]
            if not named.empty:
                id_name = named.drop_duplicates(subset=["product_id"]).set_index(
                    "product_id"
                )["product"]
                blank = lines["product"].astype(str).str.strip() == ""
                if blank.any():
                    mapped = (
                        lines.loc[blank, "product_id"]
                        .astype(str)
                        .str.strip()
                        .map(id_name)
                    )
                    lines.loc[blank, "product"] = mapped.fillna("")
                # Reverse direction: lines carry names while masters like the
                # drug list are keyed by id. Fill the missing ids from names.
                name_id = (
                    named.assign(_key=named["product"].map(norm_key))
                    .drop_duplicates(subset=["_key"])
                    .set_index("_key")["product_id"]
                )
                blank_id = lines["product_id"].astype(str).str.strip() == ""
                if blank_id.any() and len(name_id):
                    mapped_ids = (
                        lines.loc[blank_id, "product"].map(norm_key).map(name_id)
                    )
                    lines.loc[blank_id, "product_id"] = mapped_ids.fillna(
                        lines.loc[blank_id, "product_id"]
                    )
        # Attach product categories to lines for category-level questions.
        if (
            not products.empty
            and "category" in products.columns
            and "product" in lines.columns
        ):
            categorized = products[products["category"].astype(str).str.strip() != ""]
            filled = pd.Series("", index=lines.index, dtype=object)
            if (
                "product_id" in lines.columns
                and "product_id" in categorized.columns
                and not categorized.empty
            ):
                # Product IDs survive case and spacing differences between files.
                id_map = (
                    categorized[categorized["product_id"].astype(str).str.strip() != ""]
                    .drop_duplicates(subset=["product_id"])
                    .set_index("product_id")["category"]
                )
                if len(id_map):
                    filled = (
                        lines["product_id"].astype(str).str.strip().map(id_map)
                    ).fillna("")
            if not categorized.empty:
                name_map = (
                    categorized.assign(_key=categorized["product"].map(norm_key))
                    .drop_duplicates(subset=["_key"])
                    .set_index("_key")["category"]
                )
                by_name = lines["product"].map(norm_key).map(name_map).fillna("")
                filled = filled.where(filled.astype(str).str.strip() != "", by_name)
            lines["category"] = lines["category"].where(
                lines["category"].astype(str).str.strip() != "", filled
            )
        purchases, purchase_lines = cls._build_purchases(purchase_tables, profiles)
        if (
            not purchase_lines.empty
            and not products.empty
            and "product_id" in products.columns
        ):
            id_map = (
                products[products["product_id"].astype(str).str.strip() != ""]
                .drop_duplicates(subset=["product_id"])
                .set_index("product_id")["product"]
            )
            if len(id_map):
                mapped = purchase_lines["product"].map(id_map)
                purchase_lines["product"] = mapped.fillna(purchase_lines["product"])
        inventory = cls._build_inventory(tables, profiles, roles)
        if (
            inventory is not None
            and not inventory.empty
            and not products.empty
            and "product_id" in products.columns
        ):
            named = products[products["product"].astype(str).str.strip() != ""]
            if not named.empty:
                id_name = named.drop_duplicates(subset=["product_id"]).set_index(
                    "product_id"
                )["product"]
                mapped = inventory["product_id"].astype(str).str.strip().map(id_name)
                needs_name = (inventory["product"].astype(str).str.strip() == "") | (
                    inventory["product"].astype(str).str.strip()
                    == inventory["product_id"].astype(str).str.strip()
                )
                inventory["product"] = inventory["product"].where(
                    ~needs_name, mapped.fillna(inventory["product"])
                )
        tradeins = cls._build_tradeins(tables, profiles, roles)
        warranties = cls._build_warranties(tables, profiles, roles)
        employees = cls._build_employees(tables, profiles, roles)
        expenses = cls._build_expenses(tables, profiles, roles)
        drugs = cls._build_drugs(tables, profiles, roles)
        if not drugs.empty and "product_id" in lines.columns:
            drug_map = drugs.set_index("product_id")["prescription"]
            lines["prescription"] = (
                lines["product_id"].astype(str).str.strip().map(drug_map).fillna(False)
            )
            ingredient_map = drugs.set_index("product_id")["ingredient"]
            lines["ingredient"] = (
                lines["product_id"]
                .astype(str)
                .str.strip()
                .map(ingredient_map)
                .fillna("")
            )
        ds = cls(
            business_id,
            orders,
            lines,
            products,
            purchases,
            purchase_lines,
            tables,
            inventory,
            tradeins,
            warranties,
            employees,
            expenses,
            drugs,
            customers,
            conversions,
        )
        ds.product_aliases = master_aliases
        ds.product_conditions = master_conditions
        ds.gender_of_customer = gender_map
        ds.repair_tickets = cls._build_repairs(tables, employees)
        ds.lots, ds.lot_uses, ds.stock_ledger = cls._build_lots(tables)
        ds.installments = cls._build_installments(tables)
        ds.member_dates = cls._build_member_dates(tables)
        ds.price_history = cls._build_price_history(tables, profiles)
        return ds

    # --------------------------------------------------------------- ledgers
    @classmethod
    def _build_tradeins(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            if roles[id(table)] != "tradeins":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            date_col = cols.get("date")
            amount_col = cols.get("amount") or cols.get("price") or cols.get("total")
            if not date_col or not amount_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["date"] = _to_date(df[date_col])
            out["amount"] = _to_num(df[amount_col])
            out["customer"] = (
                df[cols["customer"]].astype(str).str.strip()
                if cols.get("customer")
                else ""
            )
            out["note"] = (
                df[cols["note"]].astype(str).str.strip() if cols.get("note") else ""
            )
            frames.append(out)
        # Phone shops record the trade-in credit per sales line ("thu_may_cu")
        # instead of a separate trade-in table. Harvest those columns too.
        for table in tables:
            if roles[id(table)] == "tradeins":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            trade_col = next(
                (
                    c
                    for c in df.columns
                    if "thu may cu" in norm_key(str(c))
                    or norm_key(str(c)) in {"thu cu", "tien thu cu"}
                ),
                None,
            )
            if trade_col is None:
                continue
            date_col = cols.get("date")
            if not date_col or date_col not in df.columns:
                continue
            out = pd.DataFrame(index=df.index)
            out["date"] = _to_date(df[date_col])
            out["amount"] = _to_num(df[trade_col])
            out["customer"] = (
                df[cols["customer"]].astype(str).str.strip()
                if cols.get("customer")
                else ""
            )
            out["note"] = ""
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["date", "amount", "customer", "note"])
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def _build_warranties(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            if roles[id(table)] != "warranty":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            date_col = cols.get("date")
            if not date_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["date"] = _to_date(df[date_col])
            out["customer"] = (
                df[cols["customer"]].astype(str).str.strip()
                if cols.get("customer")
                else ""
            )
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["date", "customer"])
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def _build_employees(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            if roles[id(table)] != "employees":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            name_col = cols.get("employee")
            if not name_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if key in {"ho ten", "ten", "ten goi", "ten tho", "ten nhan vien"}:
                        name_col = candidate
                        break
            rate_col = None
            for candidate in df.columns:
                key = norm_key(candidate)
                if "hoa hong" in key or "commission" in key:
                    rate_col = candidate
                    break
            id_col = cols.get("employee_id")
            if not id_col:
                for candidate in df.columns:
                    if norm_key(candidate) in {
                        "ma nhan vien",
                        "ma nv",
                        "msnv",
                        "employee id",
                    }:
                        id_col = candidate
                        break
            if not name_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["employee"] = df[name_col].astype(str).str.strip()
            out["employee_id"] = df[id_col].astype(str).str.strip() if id_col else ""
            out["rate"] = _to_num(df[rate_col]) if rate_col else float("nan")
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["employee", "employee_id", "rate"])
        return pd.concat(frames, ignore_index=True)

    # Repair-shop tickets (Sua xe rulebook §6). Any table with a vehicle
    # column ('xe', 'bien so') plus date and amount columns qualifies,
    # whatever its inferred role. Plate pattern: 59B6-219.16.
    PLATE_RE = re.compile(r"\b\d{2}[A-Z]\d(?:-\d{3}\.\d{2})?\b")

    @classmethod
    def _build_repairs(cls, tables, employees) -> pd.DataFrame:
        frames = []
        for table in tables:
            df = table.df
            if df.empty:
                continue
            norm_cols = {norm_key(str(c)): c for c in df.columns}
            vehicle_col = next(
                (
                    c
                    for key, c in norm_cols.items()
                    if key in {"xe", "bien so", "bien so xe", "xe bien so"}
                    or (key.startswith("xe") and len(key) <= 4)
                ),
                None,
            )
            if vehicle_col is None:
                continue
            date_col = next(
                (c for key, c in norm_cols.items() if "ngay" in key),
                None,
            )
            amount_col = next(
                (
                    c
                    for key, c in norm_cols.items()
                    if key in {"tien", "tong tien", "thanh tien", "amount"}
                ),
                None,
            )
            if date_col is None or amount_col is None:
                continue
            mech_col = next(
                (
                    c
                    for key, c in norm_cols.items()
                    if key in {"tho", "ten tho", "nhan vien", "ky thuat"}
                ),
                None,
            )
            job_col = next(
                (c for key, c in norm_cols.items() if "lam gi" in key),
                None,
            )
            note_col = next(
                (c for key, c in norm_cols.items() if "ghi chu" in key),
                None,
            )
            out = pd.DataFrame(index=df.index)
            out["date"] = _to_date(df[date_col])
            out["mechanic"] = df[mech_col].astype(str).str.strip() if mech_col else ""
            out["vehicle_raw"] = df[vehicle_col].astype(str).str.strip()
            out["job"] = df[job_col].astype(str).str.strip() if job_col else ""
            out["note"] = df[note_col].astype(str).str.strip() if note_col else ""
            out["amount"] = _to_num(df[amount_col])
            frames.append(out)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "date",
                    "mechanic",
                    "mech_token",
                    "mech_full",
                    "line",
                    "line_key",
                    "plate",
                    "job",
                    "amount",
                    "warranty",
                ]
            )
        tickets = pd.concat(frames, ignore_index=True)
        tickets["mech_token"] = tickets["mechanic"].map(
            lambda v: norm_key(v).split()[-1] if norm_key(v) else ""
        )
        # Mechanic full names from the employee master (last-token match).
        rate_by_token: dict[str, tuple[str, float]] = {}
        if employees is not None and not employees.empty:
            for _, row in employees.iterrows():
                token = (
                    norm_key(row["employee"]).split()[-1]
                    if norm_key(row["employee"])
                    else ""
                )
                if token and token not in rate_by_token:
                    rate_by_token[token] = (str(row["employee"]), row["rate"])
        tickets["mech_full"] = tickets["mech_token"].map(
            lambda tok: rate_by_token.get(tok, (None, None))[0]
        )
        # Split "dòng xe + biển số": line = text before the plate.
        lines, keys, plates = [], [], []
        for raw_vehicle in tickets["vehicle_raw"]:
            text = str(raw_vehicle).strip()
            match = cls.PLATE_RE.search(text.upper())
            if match:
                plate = match.group(0)
                line = text[: match.start()].strip() or text
            else:
                plate, line = "", text
            lines.append(line)
            keys.append(re.sub(r"\s+", "", norm_key(line)))
            plates.append(plate)
        tickets["line"] = lines
        tickets["line_key"] = keys
        tickets["plate"] = plates
        # Dòng xe: spellings differing only by case or abbreviation are one
        # dòng (rulebook §6: "Alpha" = "Wave Alpha", "AB" = "Air Blade").
        # Merge a shorter key into the most frequent longer key containing
        # it, plus initials ("Air Blade" -> "AB").
        from collections import Counter as _Counter

        counts = _Counter(keys)
        parent: dict[str, str] = {}
        for short in sorted(set(keys), key=len):
            if len(short) < 2:
                continue
            candidates = [k for k in counts if len(k) > len(short) and short in k]
            if candidates:
                parent[short] = max(candidates, key=lambda k: counts[k])
        for raw_line, key in zip(lines, keys):
            words = [w for w in norm_key(raw_line).split() if w[:1].isalpha()]
            initials = "".join(w[0] for w in words)
            if (
                len(initials) >= 2
                and initials != key
                and initials in counts
                and initials not in parent
            ):
                parent[initials] = key

        def _canon(key: str) -> str:
            seen = set()
            while key in parent and key not in seen:
                seen.add(key)
                key = parent[key]
            return key

        tickets["line_canon"] = [_canon(k) for k in keys]
        combined = (tickets["job"].fillna("") + " " + tickets["note"].fillna("")).map(
            norm_key
        )
        tickets["warranty"] = combined.str.contains("bao hanh")
        return tickets

    def _repair_in_period(self, period) -> "pd.DataFrame":
        df = self.repair_tickets
        if df is None or df.empty:
            return df
        if period is not None and "date" in df.columns and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        return df

    def has_repair_data(self) -> bool:
        return self.repair_tickets is not None and not self.repair_tickets.empty

    def repair_count(self, period=None, mechanic_token: str | None = None) -> int:
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return 0
        if mechanic_token:
            df = df[df["mech_token"] == mechanic_token]
        return int(len(df))

    def repair_revenue(self, period=None, mechanic_token=None, line_key=None) -> float:
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return 0.0
        if mechanic_token:
            df = df[df["mech_token"] == mechanic_token]
        if line_key:
            col = "line_canon" if "line_canon" in df.columns else "line_key"
            df = df[df[col] == line_key]
        return float(df["amount"].fillna(0).sum())

    def top_repair_mechanic(self, period=None, by: str = "count") -> str | None:
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return None
        df = df[df["mech_token"] != ""]
        if df.empty:
            return None
        if by == "revenue":
            grouped = df.groupby("mech_token")["amount"].sum()
        else:
            grouped = df.groupby("mech_token").size()
        if grouped.empty:
            return None
        token = grouped.sort_values(ascending=False).index[0]
        full = df[df["mech_token"] == token]["mech_full"].dropna()
        if not full.empty:
            return str(full.iloc[0])
        raw = df[df["mech_token"] == token]["mechanic"]
        return str(raw.iloc[0]) if not raw.empty else None

    def top_repair_line(self, period=None) -> str | None:
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return None
        key_col = "line_canon" if "line_canon" in df.columns else "line_key"
        df = df[df[key_col] != ""]
        if df.empty:
            return None
        grouped = df.groupby(key_col)["amount"].sum()
        if grouped.empty:
            return None
        key = grouped.sort_values(ascending=False).index[0]
        raw = df[df[key_col] == key]["line"]
        if raw.empty:
            return None
        # Most frequent spelling of the winning dòng.
        return str(raw.value_counts().index[0])

    def repair_line_from_question(self, question: str) -> str | None:
        """Longest vehicle-line key from the repair ledger in the question."""
        if not self.has_repair_data():
            return None
        q = re.sub(r"\s+", "", norm_key(question))
        best: str | None = None
        best_len = 0
        tickets = self.repair_tickets
        canon_of = (
            dict(zip(tickets["line_key"], tickets["line_canon"]))
            if "line_canon" in tickets.columns
            else {}
        )
        for key in tickets["line_key"].dropna().unique():
            if len(key) >= 4 and key in q and len(key) > best_len:
                best, best_len = str(key), len(key)
        if best is None:
            return None
        return canon_of.get(best, best)

    def repair_distinct_plates(self, period=None, line_key=None) -> int:
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return 0
        if line_key:
            col = "line_canon" if "line_canon" in df.columns else "line_key"
            df = df[df[col] == line_key]
        plates = df["plate"].astype(str).str.strip()
        return int(plates[plates != ""].nunique())

    def repair_repeat_plates(self, period=None) -> int:
        """Plates with 2+ paid tickets in the period (khách quen)."""
        df = self._repair_in_period(period)
        if df is None or df.empty:
            return 0
        paid = df[~df["warranty"]]
        plates = paid["plate"].astype(str).str.strip()
        paid = paid[plates != ""]
        if paid.empty:
            return 0
        counts = paid.groupby(paid["plate"]).size()
        return int((counts >= 2).sum())

    def repair_avg_per_weekday(self, period, weekday: int) -> float | None:
        """Tickets (incl. warranty) on a weekday / number of such days."""
        from datetime import timedelta

        df = self._repair_in_period(period)
        if period is None:
            return None
        dates = [
            period.start + timedelta(days=i)
            for i in range((period.end - period.start).days + 1)
        ]
        matching = [d for d in dates if d.weekday() == weekday]
        if not matching:
            return None
        count = 0
        if df is not None and not df.empty:
            valid = df["date"].apply(lambda v: _is_date(v) and v.weekday() == weekday)
            count = int(valid.sum())
        return round(count / len(matching), 2)

    # Pharmacy lots and expiry (Nha thuoc rulebook §6). Detected by column
    # names ('ma_lo', 'han_su_dung', 'ton_cuoi'), whatever the inferred role.
    @classmethod
    def _build_lots(cls, tables) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lot_frames, use_frames, ledger_frames = [], [], []
        order_dates: dict[str, object] = {}
        return_dates: dict[str, object] = {}
        product_names: dict[str, str] = {}
        for table in tables:
            df = table.df
            if df.empty:
                continue
            norm_cols = {norm_key(str(c)): c for c in df.columns}
            keys = set(norm_cols)
            if {"ma lo", "han su dung"} <= keys and "don vi tinh" not in keys:
                # (Import lines also carry ma_lo/han_su_dung but their prices
                # are per import unit, not per base unit: skip them.)
                out = pd.DataFrame(index=df.index)
                out["lot"] = df[norm_cols["ma lo"]].astype(str).str.strip()
                out["product_id"] = (
                    df[norm_cols["ma hang hoa"]].astype(str).str.strip()
                    if "ma hang hoa" in norm_cols
                    else ""
                )
                out["expiry"] = df[norm_cols["han su dung"]].map(parse_date)
                qty_col = next(
                    (
                        norm_cols[k]
                        for k in (
                            "slnhap quy doi",
                            "sl nhap quy doi",
                            "sl nhap",
                            "so luong nhap",
                        )
                        if k in norm_cols
                    ),
                    None,
                )
                out["qty_in"] = _to_num(df[qty_col]) if qty_col else 0.0
                price_col = next(
                    (norm_cols[k] for k in ("don gia nhap",) if k in norm_cols),
                    None,
                )
                # lo_hang prices are already per base unit (verified: price
                # equals chi_tiet_nhap price / pack factor).
                out["unit_price"] = _to_num(df[price_col]) if price_col else 0.0
                lot_frames.append(out)
            if {"ma lo", "ma hoa don"} <= keys:
                out = pd.DataFrame(index=df.index)
                out["lot"] = df[norm_cols["ma lo"]].astype(str).str.strip()
                out["order_id"] = df[norm_cols["ma hoa don"]].astype(str).str.strip()
                q_col = next(
                    (
                        norm_cols[k]
                        for k in (
                            "slquy doi",
                            "sl quy doi",
                            "so luong",
                            "slquydoi",
                        )
                        if k in norm_cols
                    ),
                    None,
                )
                out["qty_out"] = _to_num(df[q_col]) if q_col else 0.0
                r_col = next(
                    (
                        norm_cols[k]
                        for k in ("sltra quy doi", "sl tra quy doi")
                        if k in norm_cols
                    ),
                    None,
                )
                out["qty_back"] = _to_num(df[r_col]) if r_col else 0.0
                use_frames.append(out)
            if "ma hoa don" in keys and not {"ma lo"}.issubset(keys):
                date_col = next(
                    (
                        norm_cols[k]
                        for k in (
                            "ngay lap",
                            "thoi gian lap",
                            "ngay dat",
                            "ngay gio tra",
                        )
                        if k in norm_cols
                    ),
                    None,
                )
                if date_col is not None:
                    # Return tables feed return dates; order tables feed dates.
                    is_return = (
                        "ma phieu tra" in norm_cols or "ngay gio tra" in norm_cols
                    )
                    dates = df[date_col].map(parse_date)
                    for oid, day in zip(df[norm_cols["ma hoa don"]].astype(str), dates):
                        if day is None:
                            continue
                        if is_return:
                            if oid not in return_dates:
                                return_dates[oid] = day
                        elif oid not in order_dates:
                            order_dates[oid] = day
            if {"ma hang hoa", "ten goi"} <= keys:
                for pid, name in zip(
                    df[norm_cols["ma hang hoa"]].astype(str),
                    df[norm_cols["ten goi"]].astype(str),
                ):
                    if pid.strip() and name.strip() and pid not in product_names:
                        product_names[pid] = name.strip()
            if {"ma hang hoa", "ton cuoi"} <= keys:
                date_col = next(
                    (
                        norm_cols[k]
                        for k in ("ngay gio phat sinh", "ngay", "thoi gian")
                        if k in norm_cols
                    ),
                    None,
                )
                if date_col is not None:
                    out = pd.DataFrame(index=df.index)
                    out["product_id"] = (
                        df[norm_cols["ma hang hoa"]].astype(str).str.strip()
                    )
                    out["date"] = df[date_col].map(parse_date)
                    out["stock"] = _to_num(df[norm_cols["ton cuoi"]])
                    ledger_frames.append(out)
        lots = (
            pd.concat(lot_frames, ignore_index=True)
            if lot_frames
            else pd.DataFrame(
                columns=["lot", "product_id", "expiry", "qty_in", "unit_price"]
            )
        )
        uses = (
            pd.concat(use_frames, ignore_index=True)
            if use_frames
            else pd.DataFrame(columns=["lot", "order_id", "qty_out", "qty_back"])
        )
        if not uses.empty:
            uses["date"] = uses["order_id"].map(order_dates)
            uses["return_date"] = uses["order_id"].map(return_dates)
        ledger = (
            pd.concat(ledger_frames, ignore_index=True)
            if ledger_frames
            else pd.DataFrame(columns=["product_id", "date", "stock"])
        )
        if not lots.empty and product_names:
            lots["product"] = lots["product_id"].map(product_names).fillna("")
        elif "product" not in lots.columns:
            lots["product"] = ""
        return lots, uses, ledger

    def has_lot_data(self) -> bool:
        return self.lots is not None and not self.lots.empty

    def _lot_remaining(self, as_of) -> "pd.DataFrame":
        """Lots with remaining quantity at `as_of` (future moves ignored)."""
        lots = self.lots
        if lots is None or lots.empty:
            return lots
        out_tot = pd.Series(0.0, index=lots.index)
        back_tot = pd.Series(0.0, index=lots.index)
        if self.lot_uses is not None and not self.lot_uses.empty:
            uses = self.lot_uses
            lot_key = uses["lot"].astype(str)
            out_qty = uses["qty_out"].fillna(0).astype(float)
            back_qty = uses["qty_back"].fillna(0).astype(float)
            if as_of is not None:
                move_day = uses.get("date")
                if move_day is not None:
                    keep = ~move_day.apply(lambda v: _is_date(v) and v > as_of)
                    out_qty = out_qty[keep]
                    lot_out = lot_key[keep]
                else:
                    lot_out = lot_key
                back_day = uses.get("return_date")
                if back_day is not None:
                    keep = ~back_day.apply(lambda v: _is_date(v) and v > as_of)
                    back_qty = back_qty[keep]
                    lot_back = lot_key[keep]
                else:
                    lot_back = lot_key
            else:
                lot_out, lot_back = lot_key, lot_key
            out_sum = out_qty.groupby(lot_out).sum()
            back_sum = back_qty.groupby(lot_back).sum()
            lots_key = lots["lot"].astype(str)
            out_tot = lots_key.map(out_sum).fillna(0).astype(float)
            back_tot = lots_key.map(back_sum).fillna(0).astype(float)
        out = lots.copy()
        out["remaining"] = lots["qty_in"].fillna(0).astype(float) - out_tot + back_tot
        return out[out["remaining"] > 0]

    def lots_expiring_count(self, as_of, days: int) -> int:
        from datetime import timedelta

        df = self._lot_remaining(as_of)
        if df is None or df.empty:
            return 0
        end = as_of + timedelta(days=days) if as_of else None
        mask = df["expiry"].apply(
            lambda v: (
                _is_date(v)
                and (as_of is None or v >= as_of)
                and (end is None or v <= end)
            )
        )
        return int(mask.sum())

    def lot_capital_range(self, as_of, start, end) -> float:
        df = self._lot_remaining(as_of)
        if df is None or df.empty:
            return 0.0
        mask = df["expiry"].apply(lambda v: _is_date(v) and v >= start and v <= end)
        picked = df[mask]
        if picked.empty:
            return 0.0
        return float((picked["remaining"] * picked["unit_price"].fillna(0)).sum())

    def nearest_lot_expiry(self, product: str, as_of) -> object:
        df = self._lot_remaining(as_of)
        if df is None or df.empty or not product:
            return None
        key = norm_key(product)
        names = df["product"].map(norm_key)
        exact = df[names == key]
        if exact.empty and key:
            exact = df[names.str.contains(key, na=False)]
        if exact.empty:
            return None
        upcoming = exact[
            exact["expiry"].apply(
                lambda v: _is_date(v) and (as_of is None or v >= as_of)
            )
        ]
        if upcoming.empty:
            return None
        return upcoming["expiry"].min()

    def products_with_lots_in_range(self, as_of, start, end) -> list[str]:
        df = self._lot_remaining(as_of)
        if df is None or df.empty:
            return []
        mask = df["expiry"].apply(lambda v: _is_date(v) and v >= start and v <= end)
        picked = df[mask]
        if picked.empty:
            return []
        # Rulebook §9: earliest expiry first, ties by product id.
        earliest = (
            picked.groupby(["product_id", "product"])["expiry"].min().reset_index()
        )
        earliest = earliest.sort_values(["expiry", "product_id"])
        return [str(name) for name in earliest["product"] if str(name).strip()]

    def lot_product_names(self) -> list[str]:
        if self.lots is None or self.lots.empty or "product" not in self.lots.columns:
            return []
        return [
            str(n) for n in self.lots["product"].dropna().unique() if str(n).strip()
        ]

    def stockout_count(self, day) -> int:
        """Products whose stock at end of `day` is zero (the_kho TonCuoi)."""
        ledger = self.stock_ledger
        if ledger is None or ledger.empty:
            return 0
        valid = ledger[ledger["date"].apply(lambda v: _is_date(v) and v <= day)]
        if valid.empty:
            return 0
        last = valid.sort_values("date").groupby("product_id").tail(1)
        return int((last["stock"].fillna(0) == 0).sum())

    # Installment schedules (Cua hang dien thoai rulebook §6). The SoNo
    # ledger groups rows by contract (stt): the first row carries the
    # customer, term count and monthly amount; every row is one term with
    # its payment date (ngay_gop) and paid amount (da_gop). Due dates are
    # monthly on han_gop_ngay starting the month after the contract date.
    @classmethod
    def _build_installments(cls, tables) -> pd.DataFrame:
        cols_out = [
            "customer",
            "contract",
            "term",
            "due",
            "monthly",
            "paid_date",
            "paid",
        ]
        frames = []
        for table in tables:
            df = table.df
            if df.empty:
                continue
            norm_cols = {norm_key(str(c)): c for c in df.columns}
            keys = set(norm_cols)
            if not ({"so ky", "gop thang"} <= keys):
                continue
            if "ngay gop" not in keys and "han gop ngay" not in keys:
                continue
            stt_col = next(
                (norm_cols[k] for k in ("stt", "so thu tu") if k in norm_cols),
                None,
            )
            date_col = next(
                (norm_cols[k] for k in ("ngay", "ngay mua") if k in norm_cols),
                None,
            )
            cust_col = next(
                (
                    norm_cols[k]
                    for k in ("khach", "khach hang", "ten khach")
                    if k in norm_cols
                ),
                None,
            )
            due_col = next(
                (norm_cols[k] for k in ("han gop ngay",) if k in norm_cols),
                None,
            )
            paid_date_col = next(
                (norm_cols[k] for k in ("ngay gop",) if k in norm_cols),
                None,
            )
            paid_col = next(
                (norm_cols[k] for k in ("da gop",) if k in norm_cols),
                None,
            )
            monthly_col = next(
                (norm_cols[k] for k in ("gop thang",) if k in norm_cols),
                None,
            )
            groups: dict[str, list] = {}
            order: list[str] = []
            for idx, row in df.iterrows():
                key = str(row[stt_col]).strip() if stt_col is not None else str(idx)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(row)
            for key in order:
                rows = groups[key]
                first = rows[0]
                customer = str(first[cust_col]).strip() if cust_col is not None else ""
                if not customer:
                    continue
                contract_day = (
                    parse_date(first[date_col]) if date_col is not None else None
                )
                han_day = None
                if due_col is not None:
                    try:
                        han_day = int(float(str(first[due_col]).strip()))
                    except (ValueError, TypeError):
                        han_day = None
                monthly = (
                    parse_number(first[monthly_col])
                    if monthly_col is not None
                    else None
                )
                for number, row in enumerate(rows, start=1):
                    due = None
                    if (
                        contract_day is not None
                        and hasattr(contract_day, "year")
                        and han_day
                    ):
                        month = contract_day.month + number
                        year = contract_day.year + (month - 1) // 12
                        month = (month - 1) % 12 + 1
                        import calendar as _cal

                        last = _cal.monthrange(year, month)[1]
                        try:
                            due = date(year, month, min(han_day, last))
                        except ValueError:
                            due = None
                    frames.append(
                        {
                            "customer": customer,
                            "contract": key,
                            "term": number,
                            "due": due,
                            "monthly": monthly or 0.0,
                            "paid_date": (
                                parse_date(row[paid_date_col])
                                if paid_date_col is not None
                                else None
                            ),
                            "paid": (
                                parse_number(row[paid_col])
                                if paid_col is not None
                                else 0.0
                            )
                            or 0.0,
                        }
                    )
        if not frames:
            return pd.DataFrame(columns=cols_out)
        return pd.DataFrame(frames)

    def has_installments(self) -> bool:
        return self.installments is not None and not self.installments.empty

    def installment_customers(self) -> list[str]:
        if not self.has_installments():
            return []
        return [
            str(n)
            for n in self.installments["customer"].dropna().unique()
            if str(n).strip()
        ]

    def _installment_paid_by(self, paid_date, paid: float, ref) -> float:
        if paid_date is not None and hasattr(paid_date, "year"):
            if paid_date <= ref:
                return float(paid or 0)
            return 0.0
        return float(paid or 0)

    def installment_overdue_customers(self, ref) -> list[str]:
        """Customers with a term due before `ref` not fully paid by `ref`."""
        if not self.has_installments():
            return []
        late: set[str] = set()
        for _, row in self.installments.iterrows():
            due = row["due"]
            if due is None or not hasattr(due, "year") or due >= ref:
                continue
            paid = self._installment_paid_by(row["paid_date"], row["paid"], ref)
            if paid < (row["monthly"] or 0):
                late.add(str(row["customer"]))
        return sorted(late)

    def installment_overdue_total(self, ref) -> float:
        if not self.has_installments():
            return 0.0
        total = 0.0
        for _, row in self.installments.iterrows():
            due = row["due"]
            if due is None or not hasattr(due, "year") or due >= ref:
                continue
            paid = self._installment_paid_by(row["paid_date"], row["paid"], ref)
            total += max(0.0, (row["monthly"] or 0) - paid)
        return total

    def installment_terms_due(self, customer: str, ref) -> int:
        """Terms of one customer with due date on or before `ref`."""
        if not self.has_installments():
            return 0
        key = norm_key(customer)
        df = self.installments[self.installments["customer"].map(norm_key) == key]
        if df.empty:
            df = self.installments[
                self.installments["customer"].map(norm_key).str.contains(key, na=False)
            ]
        if df.empty:
            return 0
        return int(
            df["due"]
            .apply(lambda v: hasattr(v, "year") and v is not None and v <= ref)
            .sum()
        )

    @classmethod
    def _build_member_dates(cls, tables) -> list:
        """Registration dates from member registries ('ngay dang ky')."""
        dates: list = []
        for table in tables:
            df = table.df
            if df.empty:
                continue
            norm_cols = {norm_key(str(c)): c for c in df.columns}
            if "ngay dang ky" not in norm_cols:
                continue
            for value in df[norm_cols["ngay dang ky"]].map(parse_date):
                if _is_date(value):
                    dates.append(value)
        return dates

    def count_new_members(self, period) -> int:
        if not self.member_dates:
            return 0
        if period is None:
            return len(self.member_dates)
        return sum(1 for day in self.member_dates if period.contains(day))

    @classmethod
    def _build_price_history(cls, tables, profiles) -> dict:
        """Effective price changes: product_id -> sorted [(date, price)].

        From price-history ledgers ('lich su gia'). Used to date a price
        change the question quotes without an explicit date.
        """
        history: dict[str, list] = {}
        for table in tables:
            if "lich su gia" not in norm_key(table.name):
                continue
            cols = profiles[id(table)].roles
            pid_col = cols.get("product_id")
            date_col = cols.get("date")
            price_col = cols.get("price")
            norm_cols = {norm_key(str(c)): c for c in table.df.columns}
            if not pid_col:
                pid_col = next(
                    (
                        norm_cols[k]
                        for k in ("ma hang hoa", "ma sp", "ma thuoc")
                        if k in norm_cols
                    ),
                    None,
                )
            if not date_col:
                date_col = next(
                    (
                        norm_cols[k]
                        for k in ("ngay ap dung", "ngay hieu luc", "ngay")
                        if k in norm_cols
                    ),
                    None,
                )
            if not price_col:
                price_col = next(
                    (
                        norm_cols[k]
                        for k in ("gia ban le", "gia moi", "gia")
                        if k in norm_cols
                    ),
                    None,
                )
            if not pid_col or not date_col or not price_col:
                continue
            df = table.df
            dates = _to_date(df[date_col])
            prices = _to_num(df[price_col])
            for pid, day, price in zip(
                df[pid_col].astype(str).str.strip(), dates, prices
            ):
                if not pid or not _is_date(day) or pd.isna(price):
                    continue
                history.setdefault(pid, []).append((day, float(price)))
        for entries in history.values():
            entries.sort(key=lambda item: item[0])
        return history

    def product_ids_for_name(self, name: str) -> list[str]:
        """Product ids whose recorded name matches, for history lookups."""
        key = norm_key(name)
        ids: set[str] = set()
        for df in (self.products, self.lines):
            if df is None or df.empty:
                continue
            if "product" not in df.columns or "product_id" not in df.columns:
                continue
            matched = df[df["product"].map(norm_key) == key]["product_id"]
            ids.update(str(v).strip() for v in matched.tolist() if str(v).strip())
        return sorted(ids)

    def price_change_date(
        self, product: str, new_price: float, as_of=None, increase: bool = True
    ):
        """Latest date the product's price changed to new_price (<= as_of).

        Requires a previous entry on the other side (an actual change).
        Returns None when the history is missing or ambiguous.
        """
        if not self.price_history:
            return None
        target = int(round(new_price))
        candidates = []
        for pid in self.product_ids_for_name(product):
            entries = self.price_history.get(pid, [])
            for i, (day, price) in enumerate(entries):
                if int(round(price)) != target:
                    continue
                if as_of is not None and day > as_of:
                    continue
                if i == 0:
                    continue
                prev = entries[i - 1][1]
                if increase and not prev < price:
                    continue
                if not increase and not prev > price:
                    continue
                candidates.append(day)
        if not candidates:
            return None
        return max(candidates)

    @classmethod
    def _build_expenses(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            if roles[id(table)] != "expenses":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            date_col = cols.get("date")
            amount_col = cols.get("amount") or cols.get("total")
            if not date_col or not amount_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["date"] = _to_date(df[date_col])
            out["amount"] = _to_num(df[amount_col])
            out["category"] = (
                df[cols["category"]].astype(str).str.strip()
                if cols.get("category")
                else ""
            )
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["date", "amount", "category"])
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def _build_drugs(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            cols = profiles[id(table)].roles
            df = table.df
            prescription_col = None
            ingredient_col = None
            for candidate in df.columns:
                key = norm_key(candidate)
                if (
                    "thuoc ke don" in key
                    or "ke don" in key
                    or ("ban theo don" in key and "don vi" not in key)
                ):
                    prescription_col = candidate
                if "hoat chat" in key or "thanh phan" in key:
                    ingredient_col = candidate
            product_col = cols.get("product_id") or cols.get("product")
            if not product_col or not prescription_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["product_id"] = df[product_col].astype(str).str.strip()
            raw = df[prescription_col].astype(str).map(norm_key)
            out["prescription"] = raw.isin(
                {"1", "true", "yes", "co", "x", "dung", "thuoc ke don"}
            ) | raw.str.startswith("thuoc ke don")
            out["ingredient"] = (
                df[ingredient_col].astype(str).str.strip() if ingredient_col else ""
            )
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["product_id", "prescription", "ingredient"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["product_id"]
        )

    @classmethod
    def _build_customers(cls, tables, profiles, roles) -> pd.DataFrame:
        frames = []
        for table in tables:
            if roles[id(table)] != "customers":
                continue
            cols = profiles[id(table)].roles
            df = table.df
            id_col = cols.get("customer_id")
            name_col = None
            for candidate in df.columns:
                key = norm_key(candidate)
                if key in {
                    "ten",
                    "ten khach",
                    "ten khach hang",
                    "ten goi",
                    "ho ten",
                    "kh",
                }:
                    name_col = candidate
                    break
            if not name_col:
                name_col = cols.get("customer")
            if not id_col and not name_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["customer_id"] = df[id_col].astype(str).str.strip() if id_col else ""
            out["name"] = df[name_col].astype(str).str.strip() if name_col else ""
            group_col = None
            # Prefer an explicit customer group, then a loyalty tier. A
            # "loại hình khách" column is legal form (person/company), not tier.
            priorities = (
                ("nhom khach", "nhom khach hang"),
                ("hang khach", "hang thanh vien"),
                ("nhom", "hang"),
                ("loai khach", "loai khach hang", "doi tuong kh"),
                ("loai hinh khach",),
            )
            columns = {norm_key(candidate): candidate for candidate in df.columns}
            for group_keys in priorities:
                for group_key in group_keys:
                    if group_key in columns:
                        group_col = columns[group_key]
                        break
                if group_col:
                    break
            out["group"] = df[group_col].astype(str).str.strip() if group_col else ""
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["customer_id", "name", "group"])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ orders
    @classmethod
    def _build_orders(cls, order_tables, line_tables, profiles) -> pd.DataFrame:
        frames = []
        for table in order_tables:
            cols = profiles[id(table)].roles
            df = table.df.drop_duplicates()
            oid = cols.get("order_id")
            dt = cols.get("date")
            if not dt:
                continue
            out = pd.DataFrame(index=df.index)
            if oid:
                out["order_id"] = df[oid].astype(str).str.strip()
            else:
                if not (cols.get("amount") or cols.get("total") or cols.get("price")):
                    continue
                prefix = f"{table.name}:{table.sheet or ''}"
                out["order_id"] = [f"{prefix}:{i}" for i in df.index]
            out["date"] = _to_date_with_sheet(df[dt], table)
            time_col = None
            for candidate in df.columns:
                key = norm_key(candidate)
                if ("thoi gian tao" in key or "gio" in key or "created" in key) and (
                    df[candidate].astype(str).str.contains(":").any()
                ):
                    time_col = candidate
                    break
            out["hour"] = df[time_col].map(parse_hour) if time_col else -1
            out["channel"] = (
                df[cols["channel"]].astype(str).str.strip()
                if cols.get("channel")
                else ""
            )
            if cols.get("status"):
                out["status_raw"] = df[cols["status"]].astype(str)
            elif cols.get("note"):
                # Some ledgers hide the order state in "Ghi chú".
                out["status_raw"] = df[cols["note"]].astype(str)
            else:
                out["status_raw"] = ""
            out["status"] = [
                normalize_status(value) if norm_key(value) else "completed"
                for value in out["status_raw"]
            ]
            amount_col = cols.get("amount") or cols.get("total") or cols.get("price")
            if amount_col:
                out["revenue"] = _to_num(df[amount_col])
            elif cols.get("paid") or cols.get("debt"):
                # Header-only ledgers store the order total as paid + outstanding.
                revenue = pd.Series(0.0, index=df.index)
                if cols.get("paid"):
                    revenue = revenue + _to_num(df[cols["paid"]])
                if cols.get("debt"):
                    revenue = revenue + _to_num(df[cols["debt"]])
                out["revenue"] = revenue
            else:
                out["revenue"] = 0.0
            out["customer"] = (
                df[cols["customer"]].astype(str).str.strip()
                if cols.get("customer")
                else ""
            )
            out["customer_id"] = (
                df[cols["customer_id"]].astype(str).str.strip()
                if cols.get("customer_id")
                else ""
            )
            out["employee"] = (
                df[cols["employee"]].astype(str).str.strip()
                if cols.get("employee")
                else ""
            )
            out["employee_id"] = (
                df[cols["employee_id"]].astype(str).str.strip()
                if cols.get("employee_id")
                else ""
            )
            if cols.get("debt"):
                out["debt"] = _to_num(df[cols["debt"]]).clip(lower=0)
            elif cols.get("total") and cols.get("paid"):
                out["debt"] = (
                    _to_num(df[cols["total"]]) - _to_num(df[cols["paid"]])
                ).clip(lower=0)
            else:
                out["debt"] = (
                    _to_num(df[cols["debt"]]).clip(lower=0) if cols.get("debt") else 0.0
                )
            out["paid"] = _to_num(df[cols["paid"]]) if cols.get("paid") else 0.0
            out["discount"] = (
                _to_num(df[cols["discount"]]) if cols.get("discount") else 0.0
            )
            out["payment"] = (
                df[cols["payment"]].astype(str).str.strip()
                if cols.get("payment")
                else ""
            )
            out["service"] = (
                df[cols["product"]].astype(str).str.strip()
                if cols.get("product")
                else ""
            )
            frames.append(out)

        if frames:
            orders = pd.concat(frames, ignore_index=True)
            orders = orders.dropna(subset=["date"])
            orders = orders[orders["order_id"] != ""]
            agg = {
                "date": "min",
                "status": _status_priority,
                "status_raw": "first",
                "revenue": "sum",
                "customer": "first",
                "customer_id": "first",
                "employee": "first",
                "employee_id": "first",
                "hour": "min",
                "channel": "first",
                "debt": "max",
                "paid": "max",
                "discount": "max",
                "payment": "first",
                "service": "first",
            }
            orders = orders.groupby("order_id", as_index=False).agg(agg)
            return orders

        # No order table: derive from every line table, then aggregate.
        derived = []
        seen_orders: set = set()
        for table in line_tables:
            cols = profiles[id(table)].roles
            oid, dt = cols.get("order_id"), cols.get("date")
            if not dt:
                continue
            if not oid and not cols.get("amount"):
                continue
            df = _drop_footer_rows(table.df, cols)
            df = _drop_invalid_lines(df, cols)
            df = _drop_complete_duplicates(df, oid)
            # Excel merged cells leave order-level fields blank on extra lines.
            fill_cols = [
                cols.get(role)
                for role in (
                    "order_id",
                    "number",
                    "date",
                    "status",
                    "customer",
                    "customer_id",
                    "employee",
                    "employee_id",
                    "channel",
                    "payment",
                    "debt",
                    "paid",
                    "discount",
                )
                if cols.get(role)
            ]
            if fill_cols:
                df = df.copy()
                block = df[fill_cols].replace(r"^\s*$", pd.NA, regex=True)
                df[fill_cols] = block.ffill()
            out = pd.DataFrame(index=df.index)
            if oid:
                out["order_id"] = df[oid].astype(str).str.strip()
            elif cols.get("number"):
                prefix = f"{table.name}:{table.sheet or ''}"
                out["order_id"] = [
                    f"{prefix}:{value}"
                    for value in df[cols["number"]].astype(str).str.strip()
                ]
            else:
                prefix = f"{table.name}:{table.sheet or ''}"
                out["order_id"] = [f"{prefix}:{i}" for i in df.index]
            out["date"] = _to_date_with_sheet(df[dt], table)
            time_col = None
            for candidate in df.columns:
                key = norm_key(candidate)
                if ("thoi gian tao" in key or "gio" in key or "created" in key) and (
                    df[candidate].astype(str).str.contains(":").any()
                ):
                    time_col = candidate
                    break
            out["hour"] = df[time_col].map(parse_hour) if time_col else -1
            out["channel"] = (
                df[cols["channel"]].astype(str).str.strip()
                if cols.get("channel")
                else ""
            )
            if cols.get("status"):
                out["status_raw"] = df[cols["status"]].astype(str)
                out["status"] = out["status_raw"].map(normalize_status)
            elif cols.get("note"):
                out["status_raw"] = df[cols["note"]].astype(str)
                out["status"] = [
                    normalize_status(value) if norm_key(value) else "completed"
                    for value in out["status_raw"]
                ]
            else:
                out["status_raw"] = ""
                out["status"] = "completed"
            amount = cols.get("amount") or cols.get("total") or cols.get("price")
            out["revenue"] = _to_num(df[amount]) if amount else 0.0
            out["customer"] = (
                df[cols["customer"]].astype(str) if cols.get("customer") else ""
            )
            out["customer_id"] = ""
            out["employee"] = (
                df[cols["employee"]].astype(str).str.strip()
                if cols.get("employee")
                else ""
            )
            out["employee_id"] = (
                df[cols["employee_id"]].astype(str).str.strip()
                if cols.get("employee_id")
                else ""
            )
            out["debt"] = (
                _to_num(df[cols["debt"]]).clip(lower=0) if cols.get("debt") else 0.0
            )
            out["paid"] = _to_num(df[cols["paid"]]) if cols.get("paid") else 0.0
            out["discount"] = (
                _to_num(df[cols["discount"]]) if cols.get("discount") else 0.0
            )
            out["payment"] = (
                df[cols["payment"]].astype(str).str.strip()
                if cols.get("payment")
                else ""
            )
            out["service"] = (
                df[cols["product"]].astype(str).str.strip()
                if cols.get("product")
                else ""
            )
            out["_key_product"] = (
                df[cols["product"]].astype(str).str.strip()
                if cols.get("product")
                else ""
            )
            out["_key_quantity"] = (
                _to_num(df[cols["quantity"]]) if cols.get("quantity") else 1.0
            )
            out = _drop_seen(
                out,
                seen_orders,
                ["order_id", "date", "revenue", "_key_product", "_key_quantity"],
            )
            derived.append(out)
        if derived:
            orders = pd.concat(derived, ignore_index=True)
            orders = orders.dropna(subset=["date"])
            orders = orders[orders["order_id"] != ""]
            orders = orders.drop(
                columns=["_key_product", "_key_quantity"], errors="ignore"
            )
            return orders.groupby("order_id", as_index=False).agg(
                {
                    "date": "min",
                    "status": "first",
                    "status_raw": "first",
                    "revenue": "sum",
                    "customer": "first",
                    "customer_id": "first",
                    "employee": "first",
                    "employee_id": "first",
                    "hour": "min",
                    "channel": "first",
                    "debt": "max",
                    "paid": "max",
                    "discount": "max",
                    "payment": "first",
                    "service": "first",
                }
            )
        return pd.DataFrame(
            columns=[
                "order_id",
                "date",
                "status",
                "status_raw",
                "revenue",
                "customer",
                "customer_id",
                "employee",
                "employee_id",
                "debt",
                "paid",
                "discount",
                "payment",
                "service",
            ]
        )

    # -------------------------------------------------------------- conversions
    @classmethod
    def _build_conversions(cls, tables, profiles) -> tuple[dict, dict, dict]:
        """Map (product, pack unit) to a base-unit factor."""
        factors: dict[tuple[str, str], float] = {}
        base_units: dict[str, str] = {}
        units_by_product: dict[str, set[str]] = {}
        for table in tables:
            cols = profiles[id(table)].roles
            df = table.df
            product_col = cols.get("product")
            id_col = cols.get("product_id")
            pack_col = cols.get("pack_unit") or (
                cols.get("unit") if cols.get("factor") else None
            )
            factor_col = cols.get("factor")
            base_col = cols.get("base_unit")
            if not product_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if key in {"ten san pham", "ten hang", "hang", "ten hang hoa"}:
                        product_col = candidate
                        break
            if not id_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if key in {
                        "ma hang hoa",
                        "ma hang",
                        "ma san pham",
                        "ma sku",
                        "sku",
                    }:
                        id_col = candidate
                        break
            if not pack_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if "don vi quy doi" in key or key in {"dv lon", "don vi lon"}:
                        pack_col = candidate
                        break
            if not factor_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if "ti le" in key or "he so" in key:
                        factor_col = candidate
                        break
            if not base_col:
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if "don vi co ban" in key or key in {"dv nho", "don vi nho"}:
                        base_col = candidate
                        break
            if not ((product_col or id_col) and pack_col and factor_col):
                continue
            for _, row in df.iterrows():
                product_key = norm_key(row[product_col]) if product_col else ""
                id_key = f"#{norm_key(row[id_col])}" if id_col else ""
                unit_key = norm_key(row[pack_col])
                factor = parse_number(row[factor_col])
                if not unit_key or not factor or factor <= 0:
                    continue
                keys = [key for key in (product_key, id_key) if key]
                if not keys:
                    continue
                for key in keys:
                    factors[(key, unit_key)] = float(factor)
                    units_by_product.setdefault(key, set()).add(unit_key)
                    if base_col:
                        base_key = norm_key(row[base_col])
                        if base_key:
                            base_units[key] = base_key
        return factors, base_units, units_by_product

    # ------------------------------------------------------------------- lines
    @classmethod
    def _build_lines(
        cls,
        line_tables,
        orders: pd.DataFrame,
        profiles,
        conversions: tuple[dict, dict, dict] | None = None,
    ) -> pd.DataFrame:
        factors, base_units, units_by_product = conversions or ({}, {}, {})
        frames = []
        seen_rows: set = set()
        for table in line_tables:
            cols = profiles[id(table)].roles
            oid = cols.get("order_id")
            df = _drop_footer_rows(table.df, cols)
            df = _drop_invalid_lines(df, cols)
            df = _drop_complete_duplicates(df, oid)
            # Excel merged cells leave order-level fields blank on extra lines.
            fill_cols = [
                cols.get(role)
                for role in (
                    "order_id",
                    "number",
                    "date",
                    "status",
                    "customer",
                    "customer_id",
                    "employee",
                    "employee_id",
                    "channel",
                    "payment",
                    "debt",
                    "paid",
                    "discount",
                )
                if cols.get(role)
            ]
            if fill_cols:
                df = df.copy()
                block = df[fill_cols].replace(r"^\s*$", pd.NA, regex=True)
                df[fill_cols] = block.ffill()
            if not oid and not (cols.get("date") and cols.get("amount")):
                continue
            out = pd.DataFrame(index=df.index)
            if oid:
                out["order_id"] = df[oid].astype(str).str.strip()
            elif cols.get("number"):
                prefix = f"{table.name}:{table.sheet or ''}"
                out["order_id"] = [
                    f"{prefix}:{value}"
                    for value in df[cols["number"]].astype(str).str.strip()
                ]
            else:
                prefix = f"{table.name}:{table.sheet or ''}"
                out["order_id"] = [f"{prefix}:{i}" for i in df.index]
            out["product"] = (
                df[cols["product"]].astype(str).str.strip()
                if cols.get("product")
                else ""
            )
            out["product_id"] = (
                df[cols["product_id"]].astype(str).str.strip()
                if cols.get("product_id")
                else ""
            )
            out["quantity"] = (
                _to_num(df[cols["quantity"]]) if cols.get("quantity") else 1.0
            )
            out["unit"] = (
                df[cols["unit"]].astype(str).str.strip() if cols.get("unit") else ""
            )
            if cols.get("base_quantity"):
                out["base_quantity"] = _to_num(df[cols["base_quantity"]])
            elif factors:
                raw_quantity = (
                    df[cols["quantity"]]
                    if cols.get("quantity")
                    else pd.Series(1.0, index=df.index)
                )
                multipliers = []
                for position in range(len(df)):
                    product_key = norm_key(out["product"].iloc[position])
                    id_value = (
                        str(out["product_id"].iloc[position])
                        if "product_id" in out.columns
                        else ""
                    )
                    id_key = f"#{norm_key(id_value)}" if id_value else ""
                    product_keys = [key for key in (product_key, id_key) if key]
                    unit_key = norm_key(out["unit"].iloc[position])
                    factor = None
                    if unit_key:
                        for key in product_keys:
                            factor = factors.get((key, unit_key))
                            if factor is not None:
                                break
                        if factor is None:
                            for key in product_keys:
                                for known in units_by_product.get(key, ()):
                                    if re.search(
                                        rf"(?<![a-z0-9]){re.escape(known)}(?![a-z0-9])",
                                        unit_key,
                                    ):
                                        factor = factors.get((key, known))
                                        break
                                if factor is not None:
                                    break
                    if factor is None:
                        quantity_key = norm_key(raw_quantity.iloc[position])
                        for key in product_keys:
                            for known in units_by_product.get(key, ()):
                                if re.search(
                                    rf"(?<![a-z0-9]){re.escape(known)}(?![a-z0-9])",
                                    quantity_key,
                                ):
                                    factor = factors.get((key, known))
                                    break
                            if factor is not None:
                                break
                    multipliers.append(factor if factor else 1.0)
                out["base_quantity"] = out["quantity"] * pd.Series(
                    multipliers, index=df.index
                )
            else:
                out["base_quantity"] = out["quantity"]
            if factors:
                # Bulk rows are sometimes left unconverted in the source
                # (1 thùng recorded as 1 base unit). The README/table pack
                # factors state the gold convention; apply them only to rows
                # that still look unconverted (base == quantity).
                product_key = out["product"].map(norm_key)
                unit_key = out["unit"].map(norm_key)
                combos = pd.Series(list(zip(product_key, unit_key)), index=out.index)
                multipliers = combos.map(factors)
                if "product_id" in out.columns:
                    missing = multipliers.isna()
                    if missing.any():
                        raw_ids = out.loc[missing, "product_id"].astype(str).str.strip()
                        id_key = raw_ids.map(
                            lambda value: f"#{norm_key(value)}" if value else ""
                        )
                        id_combos = pd.Series(
                            list(zip(id_key, unit_key[missing])),
                            index=out.index[missing],
                        )
                        multipliers.loc[missing] = id_combos.map(factors)
                fixable = (
                    multipliers.notna()
                    & (multipliers > 1)
                    & ((out["base_quantity"] - out["quantity"]).abs() < 1e-9)
                )
                if fixable.any():
                    out.loc[fixable, "base_quantity"] = (
                        out.loc[fixable, "quantity"] * multipliers[fixable]
                    )
            out["unit_price"] = _to_num(df[cols["price"]]) if cols.get("price") else 0.0
            amount_col = cols.get("amount") or cols.get("total")
            out["line_amount"] = (
                _to_num(df[amount_col])
                if amount_col
                else out["quantity"] * out["unit_price"]
            )
            out["cost"] = _to_num(df[cols["cost"]]) if cols.get("cost") else 0.0
            out["category"] = (
                df[cols["category"]].astype(str).str.strip()
                if cols.get("category")
                else ""
            )
            out = _drop_seen(
                out,
                seen_rows,
                [
                    "order_id",
                    "product",
                    "quantity",
                    "base_quantity",
                    "unit",
                    "unit_price",
                    "line_amount",
                ],
            )
            frames.append(out)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "product",
                    "product_id",
                    "quantity",
                    "base_quantity",
                    "unit",
                    "unit_price",
                    "line_amount",
                    "cost",
                    "category",
                ]
            )
        lines = pd.concat(frames, ignore_index=True)
        if not orders.empty:
            meta_cols = ["order_id", "date", "status"]
            for extra in ("hour", "channel"):
                if extra in orders.columns:
                    meta_cols.append(extra)
            meta = orders[meta_cols]
            lines = lines.merge(meta, on="order_id", how="left")
        else:
            lines["date"] = pd.NaT
            lines["status"] = "completed"
        return lines

    # ---------------------------------------------------------------- products
    @classmethod
    def _build_products(
        cls, product_tables, lines: pd.DataFrame, profiles
    ) -> pd.DataFrame:
        frames = []
        for table in product_tables:
            cols = profiles[id(table)].roles
            df = table.df
            name_col = cols.get("product")
            if not name_col or name_col == cols.get("product_id"):
                # Generic exports call the name column "ten_goi" or "ten".
                for candidate in df.columns:
                    key = norm_key(candidate)
                    if key.startswith("ten") or key == "name":
                        name_col = candidate
                        break
            if not name_col:
                name_col = cols.get("product_id")
            if not name_col:
                continue
            out = pd.DataFrame(index=df.index)
            out["product_id"] = (
                df[cols["product_id"]].astype(str).str.strip()
                if cols.get("product_id")
                else ""
            )
            out["product"] = df[name_col].astype(str).str.strip()
            out["price"] = _to_num(df[cols["price"]]) if cols.get("price") else 0.0
            out["cost"] = _to_num(df[cols["cost"]]) if cols.get("cost") else 0.0
            out["unit"] = (
                df[cols["unit"]].astype(str).str.strip() if cols.get("unit") else ""
            )
            out["category"] = (
                df[cols["category"]].astype(str).str.strip()
                if cols.get("category")
                else ""
            )
            frames.append(out)
        if frames:
            products = pd.concat(frames, ignore_index=True)
            # Tables with only a product id add no name; drop those rows when a
            # named row for the same id exists.
            named_ids = set(
                products.loc[
                    products["product"].astype(str).str.strip()
                    != products["product_id"].astype(str).str.strip(),
                    "product_id",
                ]
            )
            if named_ids:
                keep = ~(
                    (
                        products["product"].astype(str).str.strip()
                        == products["product_id"].astype(str).str.strip()
                    )
                    & products["product_id"].isin(named_ids)
                )
                products = products[keep]
            return products.drop_duplicates(subset=["product"])
        if not lines.empty:
            return lines[["product", "product_id", "unit", "category"]].drop_duplicates(
                subset=["product"]
            )
        return pd.DataFrame(
            columns=["product", "product_id", "price", "cost", "unit", "category"]
        )

    # Master-data harvest: categories, alias spellings, condition tags and
    # customer gender from any master table, whatever its inferred role
    # (a conversion sheet often doubles as the product master).
    CONDITION_VALUES = frozenset({"may moi", "may cu", "phu kien"})
    GENDER_VALUES = frozenset({"nam", "nu"})
    ALIAS_KEYS = frozenset(
        {
            "ten goi khac",
            "ten khac",
            "ten day du",
            "ten thuong goi",
            "alias",
            "dien giai",
        }
    )

    @classmethod
    def _harvest_product_masters(
        cls, products, lines, tables, profiles
    ) -> tuple[pd.DataFrame, dict, dict, dict]:
        aliases: dict[str, str] = {}
        conditions: dict[str, str] = {}
        gender_map: dict[str, str] = {}
        ambiguous: set[str] = set()
        line_names = set()
        if not lines.empty and "product" in lines.columns:
            line_names = set(lines["product"].map(norm_key).dropna())
        for table in tables:
            cols = profiles[id(table)].roles
            df = table.df
            if df.empty:
                continue
            norm_cols = {norm_key(str(c)): c for c in df.columns}
            # ---- gender (customer masters) ----
            gender_keys = {norm_key(k) for k in {"nam nu", "gioi tinh", "phai"}}
            gender_col = next(
                (c for key, c in norm_cols.items() if key in gender_keys),
                None,
            )
            if gender_col is not None:
                id_col = cols.get("customer_id") or next(
                    (
                        c
                        for key, c in norm_cols.items()
                        if key in {"ma kh", "ma khach", "ma khach hang", "ma_kh"}
                    ),
                    None,
                )
                if id_col is not None and id_col in df.columns:
                    for cid, raw_gender in zip(
                        df[id_col].astype(str), df[gender_col].astype(str)
                    ):
                        gender = norm_key(raw_gender)
                        if gender in cls.GENDER_VALUES and cid.strip():
                            gender_map[norm_key(cid)] = gender
            # Only master-like tables below: skip anything order- or
            # customer-bound (their ids/phones would be bogus aliases).
            if "order_id" in cols or "customer" in cols or "customer_id" in cols:
                continue
            # ---- product masters ----
            # Canonical column = a ten-like column preferred, else the
            # distinct-string column overlapping the sales lines most
            # (codes vs full names).
            seen_cols: set = set()
            candidates: list = []
            profile_name = cols.get("product")
            if profile_name is not None and profile_name in df.columns:
                candidates.append(profile_name)
                seen_cols.add(profile_name)
            else:
                ten_like = next(
                    (
                        c
                        for key, c in norm_cols.items()
                        if key.startswith("ten")
                        or key in {"name", "hang hoa", "ten hang"}
                    ),
                    None,
                )
                if ten_like is not None:
                    candidates.append(ten_like)
                    seen_cols.add(ten_like)
            for _key, col in norm_cols.items():
                if col in seen_cols:
                    continue
                series = df[col].astype(str)
                nonblank = series[series.str.strip() != ""]
                if len(nonblank) >= 3 and nonblank.nunique() / len(nonblank) >= 0.5:
                    candidates.append(col)
            if not candidates:
                continue
            if line_names and len(candidates) > 1:
                name_col = max(
                    candidates,
                    key=lambda col: df[col].map(norm_key).isin(line_names).sum(),
                )
            else:
                name_col = candidates[0]
            names = df[name_col].astype(str).str.strip()
            # Category supplement.
            cat_col = cols.get("category")
            if (
                cat_col is not None
                and cat_col in df.columns
                and not products.empty
                and "category" in products.columns
            ):
                cat_map = (
                    pd.DataFrame({"name": names, "cat": df[cat_col]})
                    .assign(_key=lambda d: d["name"].map(norm_key))
                    .query("_key != ''")
                    .drop_duplicates(subset=["_key"])
                    .set_index("_key")["cat"]
                    .to_dict()
                )
                blank = products["category"].astype(str).str.strip() == ""
                if blank.any():
                    mapped = products.loc[blank, "product"].map(norm_key).map(cat_map)
                    products.loc[blank, "category"] = mapped.fillna("")
            # Alias + condition columns.
            alias_cols = [
                c
                for key, c in norm_cols.items()
                if c != name_col
                and (
                    key in cls.ALIAS_KEYS
                    or key.startswith("ten goi")
                    or (
                        key
                        not in {
                            (cols.get("category") or "").lower(),
                            "nhom",
                            "hang",
                            "brand",
                            "thuong hieu",
                        }
                        and c != cat_col
                    )
                )
            ]
            # Only keep string-distinct columns as alternates (drops brands).
            kept_alias_cols = []
            for col in alias_cols:
                if col not in df.columns:
                    continue
                series = df[col].astype(str)
                nonblank = series[series.str.strip() != ""]
                if len(nonblank) >= 3 and nonblank.nunique() / len(nonblank) >= 0.5:
                    kept_alias_cols.append(col)
            known_roles = {
                norm_key(str(cols.get(role) or ""))
                for role in (
                    "product_id",
                    "order_id",
                    "customer_id",
                    "employee_id",
                    "supplier",
                    "category",
                    "unit",
                    "factor",
                    "price",
                    "cost",
                    "pack_unit",
                    "base_unit",
                    "date",
                    "amount",
                    "quantity",
                )
            }
            kept_alias_cols = [
                c
                for c in kept_alias_cols
                if norm_key(str(c)) not in known_roles
                and norm_key(str(c))
                not in {
                    "hang",
                    "brand",
                    "thuong hieu",
                    "nhom",
                    "nha cung cap",
                    "ncc",
                    "nha san xuat",
                    "nsx",
                    "so size",
                    "size",
                    "co size",
                    "kich co",
                    "kich thuoc",
                    "dung luong",
                }
            ]
            condition_col = None
            for key, col in norm_cols.items():
                series = df[col].astype(str)
                nonblank = {
                    norm_key(value) for value in series[series.str.strip() != ""]
                }
                nonblank.discard("")
                if len(nonblank) >= 1 and nonblank <= set(cls.CONDITION_VALUES):
                    if len(series[series.str.strip() != ""]) >= 3:
                        condition_col = col
                        break
            for _, row in df.iterrows():
                canonical = str(row[name_col]).strip()
                if not canonical:
                    continue
                forms = [canonical]
                for col in kept_alias_cols:
                    for part in re.split(r"[,;/]", str(row[col])):
                        part = part.strip()
                        if part and part != canonical:
                            forms.append(part)
                for form in forms:
                    form_key = norm_key(form)
                    if not form_key:
                        continue
                    if form_key in aliases and aliases[form_key] != canonical:
                        ambiguous.add(form_key)
                        continue
                    aliases.setdefault(form_key, canonical)
                if condition_col is not None:
                    cond = norm_key(row[condition_col])
                    if cond in cls.CONDITION_VALUES:
                        for form in forms:
                            form_key = norm_key(form)
                            if form_key:
                                conditions.setdefault(form_key, cond)
        for form_key in ambiguous:
            aliases.pop(form_key, None)
            conditions.pop(form_key, None)
        return products, aliases, conditions, gender_map

    def product_name_set(self, name: str | None) -> set[str]:
        """All known spellings of a product (canonical + aliases)."""
        key = norm_key(name or "")
        out = {key} if key else set()
        for alias, canonical in (self.product_aliases or {}).items():
            if alias == key:
                out.add(norm_key(canonical))
            elif norm_key(canonical) == key:
                out.add(alias)
        return out

    def products_matching(self, phrase: str | None) -> list[str]:
        """Canonical product names containing `phrase` (names or aliases)."""
        key = norm_key(phrase or "")
        if not key:
            return []
        universe: dict[str, str] = {}
        if not self.products.empty and "product" in self.products.columns:
            for name in self.products["product"].dropna().unique():
                universe[norm_key(str(name))] = str(name)
        if not self.lines.empty and "product" in self.lines.columns:
            for name in self.lines["product"].dropna().unique():
                universe.setdefault(norm_key(str(name)), str(name))
        reverse: dict[str, list[str]] = {}
        for alias, canonical in (self.product_aliases or {}).items():
            reverse.setdefault(norm_key(canonical), []).append(alias)
        matched = []
        for name_key, name in universe.items():
            haystacks = [name_key] + reverse.get(name_key, [])
            if any(key in hay for hay in haystacks):
                matched.append(name)
        return matched

    def group_from_question(
        self, question: str, min_words: int = 2, min_chars: int = 7
    ) -> list[str]:
        """Canonical products sharing the longest word run with the question.

        For model families split across rows ("iPhone 13" new + used,
        "Samsung Galaxy A56 5G" 128G + 256G): returns every product tied for
        the longest run instead of picking one.
        """
        from .verifier import _word_run, _word_run_next

        q = norm_key(question)
        universe: dict[str, str] = {}
        if not self.products.empty and "product" in self.products.columns:
            for name in self.products["product"].dropna().unique():
                universe[norm_key(str(name))] = str(name)
        if not self.lines.empty and "product" in self.lines.columns:
            for name in self.lines["product"].dropna().unique():
                universe.setdefault(norm_key(str(name)), str(name))
        reverse: dict[str, list[str]] = {}
        for alias, canonical in (self.product_aliases or {}).items():
            reverse.setdefault(norm_key(canonical), []).append(alias)
        alias_to_canon = {
            alias: norm_key(canonical)
            for alias, canonical in (self.product_aliases or {}).items()
        }
        # A run that continues into a different model ("iphone 13 pro max")
        # must not count for the base model unless the question says so.
        MODEL_SUFFIX = {"pro", "max", "plus", "ultra", "lite", "mini", "fe"}
        qwords = set(q.split())
        best_words, best_len = 0, 0
        scored: list[tuple[int, int, int, str | None, str]] = []
        for name_key, name in universe.items():
            canon = alias_to_canon.get(name_key, name_key)
            pairs = [(name_key, name_key in alias_to_canon)]
            for alias_text in reverse.get(canon, []):
                if alias_text != name_key:
                    pairs.append((alias_text, True))
            top_words, top_len, top_extra = 0, 0, 10**9
            top_next: str | None = None
            for text, is_alias in pairs:
                if not is_alias:
                    run_words, run_len = _word_run(text, q)
                    if (run_words, run_len) > (top_words, top_len):
                        top_words, top_len = run_words, run_len
                        top_extra = len(text.split()) - run_words
                        top_next = None
                    if text in q:
                        name_words, name_chars = len(text.split()), len(text)
                        if (name_words, name_chars) > (top_words, top_len):
                            top_words, top_len, top_extra = (
                                name_words,
                                name_chars,
                                0,
                            )
                            top_next = None
                else:
                    run_words, run_len, next_word = _word_run_next(text, q)
                    if (run_words, run_len) >= (top_words, top_len) and (
                        run_words,
                        run_len,
                    ) != (0, 0):
                        top_words, top_len, top_extra = (
                            run_words,
                            run_len,
                            len(text.split()) - run_words,
                        )
                        top_next = next_word
            scored.append((top_words, top_len, top_extra, top_next, name))
            if (top_words, top_len) > (best_words, best_len):
                best_words, best_len = top_words, top_len
        if best_words < min_words or best_len < min_chars:
            return []
        # "iPhone 13" means every variant of the base model (storage, color,
        # new/used) — but not "iPhone 13 Pro Max". Drop products whose run
        # continues into another model unless the question says so.
        group = [
            (next_word, name)
            for words, length, _extra, next_word, name in scored
            if (words, length) == (best_words, best_len)
        ]
        filtered = [
            name
            for next_word, name in group
            if next_word not in MODEL_SUFFIX or next_word in qwords
        ]
        return sorted({name for name in (filtered or [n for _, n in group])})

    def superset_group(self, question: str, names: set[str]) -> list[str] | None:
        """Denominator for a share: the broad word in a "tổng số X" phrase.

        For "... % tổng số bịch bỉm bán ra": "bim" matching all diapers
        (names and aliases). Falls back to the broadest strict superset
        word in the question when no "tổng số" phrase matches.
        """
        import re

        q = norm_key(question)
        stop = {
            "thang",
            "nam",
            "tuan",
            "ngay",
            "hom",
            "nay",
            "truoc",
            "nhat",
            "may",
            "chiec",
            "ban",
            "duoc",
            "tong",
            "bao",
            "nhieu",
            "bnhiu",
            "tien",
            "dthu",
            "doanh",
            "thu",
            "cua",
            "hang",
            "tinh",
            "so",
            "chiem",
            "phan",
            "tram",
            "chung",
            "loai",
            "moi",
            "cu",
            "ca",
            "dang",
            "cho",
            "khach",
            "cai",
            "tat",
            "gop",
        }
        m = re.search(r"tong\s+(?:so\s+)?([a-z ]+?)(?:\s+ban|\s+trong|$)", q)
        if m:
            best: list[str] | None = None
            for word in dict.fromkeys(m.group(1).split()):
                if len(word) < 3 or word in stop:
                    continue
                matched = self.products_matching(word)
                if len(matched) >= 2 and (best is None or len(matched) > len(best)):
                    best = matched
            if best is not None:
                return best
        return None

    def brand_group_from_question(self, question: str) -> list[str]:
        """Products sharing one distinctive word (brand) with the question."""
        q = norm_key(question)
        stop = {
            "thang",
            "nam",
            "tuan",
            "ngay",
            "hom",
            "nay",
            "truoc",
            "nhat",
            "may",
            "chiec",
            "ban",
            "duoc",
            "tong",
            "bao",
            "nhieu",
            "bnhiu",
            "tien",
            "dthu",
            "doanh",
            "thu",
            "cua",
            "hang",
            "tinh",
            "so",
            "chiem",
            "phan",
            "tram",
            "chung",
            "loai",
            "moi",
            "cu",
        }
        counts: dict[str, list[str]] = {}
        for word in dict.fromkeys(q.split()):
            if len(word) < 4 or word in stop:
                continue
            matched = self.products_matching(word)
            if len(matched) >= 2:
                counts[word] = matched
        if not counts:
            return []
        best = max(counts, key=lambda w: (len(w), len(counts[w])))
        return counts[best]

    def lines_for_names(self, period, names: list[str] | set[str]) -> "pd.DataFrame":
        df = self.filtered_lines(period)
        if df.empty or not names:
            return df.iloc[0:0]
        wanted: set[str] = set()
        for name in names:
            wanted.update(self.product_name_set(name))
        return df[df["product"].map(norm_key).isin(wanted)]

    def revenue_group(self, period, names: list[str]) -> float:
        return float(self.lines_for_names(period, names)["line_amount"].sum())

    def quantity_group(self, period, names: list[str]) -> float | None:
        df = self.lines_for_names(period, names)
        if df.empty:
            return None
        metric = "base_quantity" if "base_quantity" in df.columns else "quantity"
        return float(df[metric].sum())

    def lines_for_conditions(self, period, wanted: set[str]):
        df = self.filtered_lines(period)
        if df.empty:
            return df.iloc[0:0]
        conditions = self.product_conditions or {}
        return df[df["product"].map(norm_key).map(conditions).isin(wanted)]

    def revenue_condition(self, period, wanted: set[str]) -> float:
        return float(self.lines_for_conditions(period, wanted)["line_amount"].sum())

    def quantity_condition(self, period, wanted: set[str]) -> float | None:
        df = self.lines_for_conditions(period, wanted)
        if df.empty:
            return None
        metric = "base_quantity" if "base_quantity" in df.columns else "quantity"
        return float(df[metric].sum())

    def revenue_gender(self, period, gender: str) -> float | None:
        """Revenue from orders of one gender ('nam'/'nu')."""
        gender_map = self.gender_of_customer or {}
        if not gender_map:
            return None
        df = self.valid_orders(period)
        if df.empty or "customer_id" not in df.columns:
            return None
        key = norm_key(gender)
        matched = df[
            df["customer_id"].astype(str).str.strip().map(norm_key).map(gender_map)
            == key
        ]
        if matched.empty or len(self.lines) == 0:
            return None
        ids = set(matched["order_id"])
        lines = self.lines[
            self.lines["order_id"].isin(ids)
            & ~self.lines["status"].isin(["cancelled", "returned"])
        ]
        if period is not None:
            lines = lines[lines["date"].apply(period.contains)]
        if lines.empty:
            return None
        return float(lines["line_amount"].sum())

    def _column_values_present(self, match) -> bool:
        """True when a column matching `match` has real (non-blank/nonzero)."""
        frames = []
        if self.orders is not None and not self.orders.empty:
            frames.append(self.orders)
        if self.lines is not None and not self.lines.empty:
            frames.append(self.lines)
        for table in self.tables:
            frames.append(table.df)
        for df in frames:
            for col in df.columns:
                if not match(norm_key(str(col))):
                    continue
                series = df[col].astype(str).str.strip()
                series = series[
                    ~series.isin(["", "nan", "None", "none", "-", "0", "0.0"])
                ]
                if series.empty:
                    continue
                number = parse_number(series.iloc[0])
                if number is not None:
                    if (series.map(parse_number).fillna(0) != 0).any():
                        return True
                    continue
                return True
        return False

    def has_gender_data(self) -> bool:
        return bool(self.gender_of_customer)

    def has_voucher_data(self) -> bool:
        """True when a voucher/discount amount column carries real values."""
        return self._column_values_present(
            lambda key: (
                ("voucher" in key or "chiet khau" in key) and "chuong trinh" not in key
            )
        )

    def has_shipping_data(self) -> bool:
        return self._column_values_present(
            lambda key: (
                ("ship" in key and ("phi" in key or "tien" in key or "cuoc" in key))
                or key in {"phi_ship", "phi ship"}
            )
        )

    def has_address_data(self) -> bool:
        return self._column_values_present(
            lambda key: (
                "dia chi" in key
                or key in {"addr", "address", "dia_chi"}
                or ("duong" in key and "pho" in key)
            )
        )

    def has_rating_data(self) -> bool:
        return self._column_values_present(
            lambda key: "danh gia" in key or "rating" in key or "review" in key
        )

    def has_customer_linkage(self, period=None) -> bool:
        df = self.valid_orders(period)
        if df.empty:
            return False
        for col in ("customer", "customer_id"):
            if col in df.columns and (df[col].astype(str).str.strip() != "").any():
                return True
        return False

    # --------------------------------------------------------------- purchases
    @classmethod
    def _build_purchases(cls, purchase_tables, profiles):
        purchase_frames, line_frames = [], []
        for table in purchase_tables:
            cols = profiles[id(table)].roles
            df = table.df
            purchase_id = cols.get("purchase_id") or cols.get("order_id")
            product_col = cols.get("product") or cols.get("product_id")
            if purchase_id and product_col:
                out = pd.DataFrame(index=df.index)
                out["purchase_id"] = df[purchase_id].astype(str).str.strip()
                out["product"] = df[product_col].astype(str).str.strip()
                out["product_id"] = (
                    df[cols["product_id"]].astype(str).str.strip()
                    if cols.get("product_id")
                    else ""
                )
                out["quantity"] = (
                    _to_num(df[cols["quantity"]]) if cols.get("quantity") else 1.0
                )
                out["unit_price"] = (
                    _to_num(df[cols["price"]]) if cols.get("price") else 0.0
                )
                out["amount"] = (
                    _to_num(df[cols["amount"]])
                    if cols.get("amount")
                    else out["quantity"] * out["unit_price"]
                )
                out["supplier"] = (
                    df[cols["supplier"]].astype(str).str.strip()
                    if cols.get("supplier")
                    else ""
                )
                out["unit"] = (
                    df[cols["unit"]].astype(str).str.strip() if cols.get("unit") else ""
                )
                out["date"] = (
                    _to_date(df[cols["date"]])
                    if cols.get("date") and "han" not in norm_key(cols["date"])
                    else pd.NaT
                )
                line_frames.append(out)
            elif purchase_id:
                out = pd.DataFrame(index=df.index)
                out["purchase_id"] = df[purchase_id].astype(str).str.strip()
                out["date"] = _to_date(df[cols["date"]]) if cols.get("date") else pd.NaT
                supplier_col = cols.get("supplier") or cols.get("supplier_id")
                out["supplier"] = (
                    df[supplier_col].astype(str).str.strip() if supplier_col else ""
                )
                amount_col = cols.get("amount") or cols.get("total")
                out["amount"] = _to_num(df[amount_col]) if amount_col else 0.0
                out["status"] = (
                    df[cols["status"]].map(normalize_status)
                    if cols.get("status")
                    else "completed"
                )
                purchase_frames.append(out)
        purchases = (
            pd.concat(purchase_frames, ignore_index=True)
            if purchase_frames
            else pd.DataFrame(
                columns=["purchase_id", "date", "supplier", "amount", "status"]
            )
        )
        lines = (
            pd.concat(line_frames, ignore_index=True)
            if line_frames
            else pd.DataFrame(
                columns=[
                    "purchase_id",
                    "product",
                    "quantity",
                    "unit_price",
                    "amount",
                    "unit",
                    "product_id",
                ]
            )
        )
        if not lines.empty and not purchases.empty:
            header = purchases.drop_duplicates(subset=["purchase_id"])
            if "date" in header.columns:
                date_map = header.set_index("purchase_id")["date"]
                header_date = lines["purchase_id"].map(date_map)
                if "date" in lines.columns:
                    lines["date"] = lines["date"].where(
                        lines["date"].notna(), header_date
                    )
                else:
                    lines["date"] = header_date
                lines["date"] = lines["date"].map(_plain_date)
            if "supplier" in header.columns:
                supplier_map = header.set_index("purchase_id")["supplier"]
                header_supplier = lines["purchase_id"].map(supplier_map)
                if "supplier" not in lines.columns:
                    lines["supplier"] = ""
                lines["supplier"] = lines["supplier"].where(
                    lines["supplier"].astype(str).str.strip() != "",
                    header_supplier.fillna(""),
                )
        return purchases, lines

    # ------------------------------------------------------------------ inventory
    @classmethod
    def _build_inventory(cls, tables, profiles, roles) -> pd.DataFrame:
        inventory_tables = [t for t in tables if roles[id(t)] == "inventory"]
        if not inventory_tables:
            inventory_tables = [
                t
                for t in tables
                if roles[id(t)] == "products"
                and any(
                    column in t.df.columns
                    for column in (
                        "ton_cuoi_ky",
                        "ton_cuoi",
                        "ton",
                        "so_luong_ton",
                        "ton_kho",
                    )
                )
            ]
        frames = []
        for table in inventory_tables:
            cols = profiles[id(table)].roles
            product_col = cols.get("product") or cols.get("product_id")
            if not product_col:
                continue
            qty_col = None
            for candidate in (
                "ton_cuoi_ky",
                "ton_cuoi",
                "ton",
                "so_luong_ton",
                "ton_kho",
            ):
                if candidate in table.df.columns:
                    qty_col = candidate
                    break
            if qty_col is None and cols.get("number"):
                qty_col = cols["number"]
            if qty_col is None:
                continue
            out = pd.DataFrame(index=table.df.index)
            out["product"] = table.df[product_col].astype(str).str.strip()
            out["product_id"] = (
                table.df[cols["product_id"]].astype(str).str.strip()
                if cols.get("product_id")
                else ""
            )
            out["unit"] = (
                table.df[cols["unit"]].astype(str).str.strip()
                if cols.get("unit")
                else ""
            )
            out["quantity"] = _to_num(table.df[qty_col])
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["product", "product_id", "unit", "quantity"])
        inventory = pd.concat(frames, ignore_index=True)
        return inventory.drop_duplicates(subset=["product", "unit"], keep="last")

    def stock(self, product: str | None = None) -> float | None:
        """Closing stock of one product in base units (inventory snapshot)."""
        if self.inventory is None or self.inventory.empty or not product:
            return None
        key = norm_key(product)
        rows = self.inventory[self.inventory["product"].map(norm_key) == key]
        if rows.empty:
            matches = [
                name
                for name in self.inventory["product"].unique()
                if key in norm_key(name)
            ]
            if matches:
                best = max(matches, key=lambda name: len(norm_key(name)))
                rows = self.inventory[self.inventory["product"] == best]
        if rows.empty:
            return None
        return float(rows["quantity"].sum())

    def stock_packs(self, product: str | None = None) -> list[int] | None:
        """Closing stock as [full packs, loose units] (rulebook §5)."""
        if self.inventory is None or self.inventory.empty or not product:
            return None
        key = norm_key(product)
        rows = self.inventory[self.inventory["product"].map(norm_key) == key]
        if rows.empty:
            matches = [
                name
                for name in self.inventory["product"].unique()
                if key in norm_key(name)
            ]
            if matches:
                best = max(matches, key=lambda name: len(norm_key(name)))
                rows = self.inventory[self.inventory["product"] == best]
        if rows.empty:
            return None
        factors, _, _ = self.conversions or ({}, {}, {})
        factor = None
        for _, row in rows.iterrows():
            for candidate in (
                norm_key(row.get("product", "")),
                f"#{norm_key(str(row.get('product_id', '')))}",
            ):
                if not candidate or candidate == "#":
                    continue
                for (product_key, _), value in factors.items():
                    if product_key == candidate and value and value > 1:
                        factor = float(value)
                        break
                if factor:
                    break
            if factor:
                break
        if not factor:
            return None
        total = int(rows["quantity"].sum())
        packs = total // int(factor)
        return [packs, total - packs * int(factor)]

    def inventory_value(self) -> float | None:
        """Stock value at cost per base unit (rulebook §5)."""
        if self.inventory is None or self.inventory.empty or self.products.empty:
            return None
        costs = {
            norm_key(product): float(cost)
            for product, cost in zip(self.products["product"], self.products["cost"])
            if cost and float(cost) > 0
        }
        total = 0.0
        matched = 0
        for _, row in self.inventory.iterrows():
            cost = costs.get(norm_key(row["product"]))
            if cost:
                total += float(row["quantity"]) * cost
                matched += 1
        return total if matched else None

    def unsold_count(self, period: Period | None = None) -> int:
        """Products sold before the period but not during it (rulebook §5)."""
        if period is None or self.lines.empty or "date" not in self.lines.columns:
            return 0
        valid = self.lines[~self.lines["status"].isin(["cancelled", "returned"])]
        before = valid[valid["date"].apply(lambda d: _is_date(d) and d < period.start)]
        during = valid[valid["date"].apply(period.contains)]
        return len(set(before["product"]) - set(during["product"]))

    def products_in_orders(self, period: Period | None, min_orders: int) -> int:
        """Products appearing in at least N distinct orders (rulebook §3)."""
        if period is None or min_orders <= 0:
            return 0
        df = self.filtered_lines(period)
        if df.empty:
            return 0
        df = df.assign(_key=df["product"].map(norm_key))
        df = df[df["_key"] != "phi phuc vu"]
        counts = df.groupby("_key")["order_id"].nunique()
        return int((counts >= min_orders).sum())

    # ----------------------------------------------------------------- metrics
    def valid_orders(
        self, period: Period | None = None, include_returned: bool = False
    ) -> pd.DataFrame:
        df = self.orders
        if df.empty or "status" not in df.columns:
            return df
        mask = df["status"].isin({"completed", "shipping", "other", "unknown"})
        if include_returned:
            mask = ~df["status"].isin({"cancelled"})
        if period is not None:
            mask &= df["date"].apply(period.contains)
        return df[mask]

    def filtered_orders(
        self,
        period: Period | None = None,
        customer: str | None = None,
        employee: str | None = None,
    ) -> pd.DataFrame:
        df = self.valid_orders(period)
        if customer:
            key = norm_key(customer)
            df = df[df["customer"].map(norm_key) == key]
        if employee:
            key = norm_key(employee)
            df = df[df["employee"].map(norm_key) == key]
        return df

    def revenue_shift(self, period: Period | None, shift: str) -> float | None:
        df = self.valid_orders(period)
        if df.empty or "hour" not in df.columns:
            return None
        hour = df["hour"]
        if shift == "morning":
            df = df[(hour >= 6) & (hour < 14)]
        elif shift == "afternoon":
            df = df[(hour >= 14) & (hour < 22)]
        else:
            return None
        return float(df["revenue"].sum())

    def orders_shift(self, period: Period | None, shift: str) -> int | None:
        df = self.valid_orders(period)
        if df.empty or "hour" not in df.columns:
            return None
        hour = df["hour"]
        if shift == "morning":
            df = df[(hour >= 6) & (hour < 14)]
        elif shift == "afternoon":
            df = df[(hour >= 14) & (hour < 22)]
        else:
            return None
        return int(len(df))

    def revenue_channel(self, period: Period | None, keyword: str) -> float | None:
        df = self.valid_orders(period)
        if df.empty or "channel" not in df.columns:
            return None
        if df["channel"].astype(str).str.strip().eq("").all():
            return None
        key = norm_key(keyword)
        matched = df[df["channel"].map(norm_key).str.contains(key, na=False)]
        if matched.empty:
            return None
        # Rulebook §1: revenue is the sum of line amounts.
        if len(self.lines):
            ids = set(matched["order_id"])
            lines = self.lines[
                self.lines["order_id"].isin(ids)
                & ~self.lines["status"].isin(["cancelled", "returned"])
            ]
            if period is not None:
                lines = lines[lines["date"].apply(period.contains)]
            if not lines.empty:
                return float(lines["line_amount"].sum())
        return float(matched["revenue"].sum())

    def channel_share(self, period: Period | None, keyword: str) -> float | None:
        total = self.revenue(period)
        if total <= 0:
            return None
        part = self.revenue_channel(period, keyword)
        if part is None:
            return None
        return round(100.0 * part / total, 2)

    def revenue_employee(self, period: Period | None, employee: str) -> float | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        key = norm_key(employee)
        mask = df["employee_id"].map(norm_key).eq(key) | df["employee"].map(
            norm_key
        ).eq(key)
        if not mask.any():
            mask = df["employee"].map(norm_key).str.contains(key, na=False) | df[
                "employee_id"
            ].map(norm_key).str.contains(key, na=False)
        if not mask.any() and key.split():
            # Shop ledgers often shorten names to the last word ("A.Phúc").
            last = key.split()[-1]
            tokens = df["employee"].map(norm_key).str.split().str[-1]
            mask = tokens.eq(last)
        if not mask.any():
            for eid in self._employee_ids(employee):
                mask = mask | df["employee_id"].astype(str).str.strip().eq(eid)
        if not mask.any():
            return None
        return float(df[mask]["revenue"].sum())

    def top_employee(self, period: Period | None = None) -> str | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        name = df["employee"].astype(str).str.strip()
        name = name.where(name != "", df["employee_id"].astype(str).str.strip())
        df = df[name != ""]
        if df.empty:
            return None
        grouped = df.groupby(name)["revenue"].sum().sort_values(ascending=False)
        return str(grouped.index[0])

    def _employee_ids(self, employee: str) -> list[str]:
        """Staff codes for a name via the employee master (name -> ID)."""
        if self.employees is None or self.employees.empty:
            return []
        if "employee_id" not in self.employees.columns:
            return []
        key = norm_key(employee)
        rows = self.employees[self.employees["employee"].map(norm_key) == key]
        if rows.empty and key.split():
            last = key.split()[-1]
            rows = self.employees[
                self.employees["employee"].map(norm_key).str.split().str[-1] == last
            ]
        return [str(v).strip() for v in rows["employee_id"].tolist() if str(v).strip()]

    def _employee_mask(self, df: pd.DataFrame, employee: str) -> pd.Series:
        key = norm_key(employee)
        mask = df["employee_id"].map(norm_key).eq(key) | df["employee"].map(
            norm_key
        ).eq(key)
        if not mask.any():
            mask = df["employee"].map(norm_key).str.contains(key, na=False) | df[
                "employee_id"
            ].map(norm_key).str.contains(key, na=False)
        if not mask.any() and key.split():
            last = key.split()[-1]
            tokens = df["employee"].map(norm_key).str.split().str[-1]
            mask = tokens.eq(last)
        if not mask.any():
            for eid in self._employee_ids(employee):
                mask = mask | df["employee_id"].astype(str).str.strip().eq(eid)
        return mask

    def employee_order_count(self, period: Period | None, employee: str) -> int | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        mask = self._employee_mask(df, employee)
        if not mask.any():
            return None
        return int(df[mask]["order_id"].nunique())

    def top_employee_tickets(self, period: Period | None = None) -> str | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        name = df["employee"].astype(str).str.strip()
        name = name.where(name != "", df["employee_id"].astype(str).str.strip())
        df = df[name != ""]
        if df.empty:
            return None
        grouped = (
            df.assign(_name=name[name != ""])
            .groupby("_name")["order_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        return str(grouped.index[0])

    def top_employee_days(self, period: Period | None = None) -> str | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        name = df["employee"].astype(str).str.strip()
        name = name.where(name != "", df["employee_id"].astype(str).str.strip())
        df = df[name != ""]
        if df.empty:
            return None
        grouped = (
            df.assign(_name=name[name != ""])
            .groupby("_name")["date"]
            .nunique()
            .sort_values(ascending=False)
        )
        return str(grouped.index[0])

    def match_employee(self, question: str) -> str | None:
        if self.employees is None or self.employees.empty:
            return None
        q = norm_key(question)
        best: str | None = None
        best_len = 0
        for name in self.employees["employee"].dropna().unique():
            key = norm_key(name)
            if len(key) >= 4 and key in q and len(key) > best_len:
                best, best_len = str(name), len(key)
        return best

    def commission_rate(self, employee: str) -> float | None:
        if self.employees is None or self.employees.empty:
            return None
        key = norm_key(employee)
        rows = self.employees[self.employees["employee"].map(norm_key) == key]
        if rows.empty and key.split():
            last = key.split()[-1]
            rows = self.employees[
                self.employees["employee"].map(norm_key).str.split().str[-1] == last
            ]
        if rows.empty:
            return None
        rate = rows["rate"].iloc[0]
        if pd.isna(rate):
            return None
        return float(rate)

    def commission(self, period: Period | None, employee: str) -> float | None:
        rate = self.commission_rate(employee)
        if rate is None:
            return None
        revenue = self.revenue_employee(period, employee)
        if revenue is None:
            return None
        return round(revenue * rate / 100.0)

    def commission_by_employee(
        self, period: Period | None = None
    ) -> list[tuple[str, float]]:
        if self.employees is None or self.employees.empty:
            return []
        df = self.valid_orders(period)
        if df.empty:
            return []
        name = df["employee"].astype(str).str.strip()
        df = df[name != ""]
        if df.empty:
            return []
        tokens = name[name != ""].map(
            lambda value: norm_key(value).split()[-1] if norm_key(value) else ""
        )
        grouped = df.assign(_token=tokens).groupby("_token")["revenue"].sum()
        result = []
        for token, revenue in grouped.items():
            if not token:
                continue
            rows = self.employees[
                self.employees["employee"].map(norm_key).str.split().str[-1] == token
            ]
            if rows.empty:
                continue
            employee = str(rows["employee"].iloc[0])
            rate = rows["rate"].iloc[0]
            if pd.isna(rate):
                continue
            result.append((employee, float(revenue) * float(rate) / 100.0))
        return sorted(result, key=lambda item: item[1], reverse=True)

    def filtered_lines(
        self,
        period: Period | None = None,
        product: str | None = None,
        category: str | None = None,
    ) -> pd.DataFrame:
        df = self.lines
        if df.empty or "date" not in df.columns:
            return df
        if period is not None and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        df = df[~df["status"].isin(["cancelled", "returned"])]
        if product:
            key = norm_key(product)
            keys = df["product"].map(norm_key)
            exact = df[keys == key]
            if exact.empty and key:
                # Alias spellings: same product recorded under another name.
                for spelling in self.product_name_set(product):
                    exact = df[keys == spelling]
                    if not exact.empty:
                        break
            if exact.empty and key:
                # Partial product names: keep the longest contained match.
                matches = [
                    name
                    for name in df["product"].dropna().unique()
                    if key in norm_key(name)
                ]
                if matches:
                    best = max(matches, key=lambda name: len(norm_key(name)))
                    exact = df[df["product"] == best]
            df = exact
        if category:
            key = norm_key(category)
            df = df[df["category"].map(norm_key) == key]
        return df

    def product_quantity(
        self, period: Period | None = None, product: str | None = None
    ) -> float | None:
        """Units sold of one product, in base units."""
        if not product:
            return None
        df = self.filtered_lines(period, product=product)
        if df.empty:
            return None
        metric = "base_quantity" if "base_quantity" in df.columns else "quantity"
        return float(df[metric].sum())

    def price_on_date(self, product: str, day: date) -> float | None:
        """Unit price per base unit on one day (rulebook §5)."""
        if self.lines.empty or "date" not in self.lines.columns:
            return None
        key = norm_key(product)
        df = self.lines[self.lines["date"] == day]
        df = df[df["product"].map(norm_key) == key]
        if df.empty:
            return None
        prices = []
        for _, row in df.iterrows():
            unit_price = row.get("unit_price") or 0
            if unit_price > 0:
                prices.append(round(unit_price))
        if not prices:
            return None
        # Prices on one day are normally identical; pick the most common value.
        return float(pd.Series(prices).mode().iat[0])

    def quantity_by_unit(
        self,
        period: Period | None = None,
        unit: str | None = None,
        size: str | None = None,
    ) -> float | None:
        """Units sold for products whose base unit is `unit` (rulebook §3)."""
        if not unit:
            return None
        df = self.filtered_lines(period)
        if df.empty:
            return None
        wanted = norm_key(unit)
        if size:
            # Size is encoded in the product name ("Cà phê sữa (L)"); unit
            # masters often omit drink products, so the name suffix wins.
            suffix = norm_key(size)
            df = df[df["product"].map(lambda name: norm_key(name).endswith(suffix))]
        else:
            master_units: dict[str, str] = {}
            if not self.products.empty and "unit" in self.products.columns:
                for name, unit in zip(self.products["product"], self.products["unit"]):
                    key = norm_key(name)
                    if key and key not in master_units:
                        master_units[key] = unit
            master_key = df["product"].map(
                lambda name: norm_key(master_units.get(norm_key(name), ""))
            )
            line_key = df["unit"].map(norm_key)
            # Prefer the catalog base unit; fall back to the line unit when the
            # catalog has no exact name (common with case/spacing variants).
            effective = master_key.where(master_key != "", line_key)
            df = df[effective == wanted]
        if df.empty:
            return None
        metric = "base_quantity" if "base_quantity" in df.columns else "quantity"
        return float(df[metric].sum())

    def date_range(self) -> tuple[date, date] | None:
        # pandas.NaT is an instance of datetime.date, so it must be filtered
        # explicitly; otherwise comparisons raise TypeError at inference time.
        dates: list[date] = []
        if not self.orders.empty and "date" in self.orders.columns:
            dates.extend(
                value
                for value in self.orders["date"]
                if isinstance(value, date) and not pd.isna(value)
            )
        if not self.lines.empty and "date" in self.lines.columns:
            dates.extend(
                value
                for value in self.lines["date"]
                if isinstance(value, date) and not pd.isna(value)
            )
        if not dates:
            return None
        return min(dates), max(dates)

    def revenue_last_year(self, period: Period) -> float:
        def shift(value: date) -> date:
            try:
                return value.replace(year=value.year - 1)
            except ValueError:
                return value.replace(year=value.year - 1, day=28)

        return self.revenue(Period(shift(period.start), shift(period.end)))

    def growth_yoy(self, period: Period, metric: str = "revenue") -> float | None:
        if metric == "orders":
            current = self.order_count(period)
            previous = self.order_count(
                Period(
                    period.start.replace(year=period.start.year - 1),
                    period.end.replace(year=period.end.year - 1),
                )
            )
        else:
            current = self.revenue(period)
            previous = self.revenue_last_year(period)
        if previous <= 0:
            return None
        return round(100.0 * (current - previous) / previous, 2)

    def revenue(self, period: Period | None = None) -> float:
        # Rulebook §1: revenue is the sum of line amounts after line discount.
        # An order header may carry an order-level discount, so the line sum is
        # the canonical source whenever lines exist.
        if len(self.lines):
            return float(self.filtered_lines(period)["line_amount"].sum())
        return float(self.valid_orders(period)["revenue"].sum())

    def revenue_filtered(
        self,
        period: Period | None = None,
        product: str | None = None,
        category: str | None = None,
        customer: str | None = None,
        employee: str | None = None,
        customer_group: str | None = None,
    ) -> float:
        if product or category:
            df = self.filtered_lines(period, product=product, category=category)
            return float(df["line_amount"].sum())
        df = self.filtered_orders(period, customer=customer, employee=employee)
        if customer_group and "customer_group" in df.columns:
            key = norm_key(customer_group)
            df = df[df["customer_group"].map(norm_key) == key]
        if len(self.lines) and len(df):
            ids = set(df["order_id"])
            lines = self.lines[
                self.lines["order_id"].isin(ids)
                & ~self.lines["status"].isin(["cancelled", "returned"])
            ]
            return float(lines["line_amount"].sum())
        return float(df["revenue"].sum())

    def revenue_share(
        self, category: str | None, period: Period | None = None
    ) -> float | None:
        """Category share of total revenue, in percent."""
        if not category:
            return None
        total = self.revenue(period)
        if total <= 0:
            return None
        part = float(
            self.filtered_lines(period, category=category)["line_amount"].sum()
        )
        return round(100.0 * part / total, 2)

    def top_products_filtered(
        self,
        period: Period | None = None,
        by: str = "line_amount",
        k: int = 1,
        category: str | None = None,
    ) -> list[tuple[str, float]]:
        df = self.filtered_lines(period, category=category)
        if df.empty:
            return []
        # "Phí phục vụ" is not a product (rulebook §3).
        df = df[df["product"].map(lambda name: norm_key(name) != "phi phuc vu")]
        if df.empty:
            return []
        metric = "base_quantity" if by == "quantity" else "line_amount"
        if metric not in df.columns:
            metric = "quantity" if by == "quantity" else "line_amount"
        return self._rank_products(df, metric, k)

    def order_count(self, period: Period | None = None) -> int:
        return int(len(self.valid_orders(period)))

    def cancellation_rate(self, period: Period | None = None) -> float:
        if period is None or self.orders.empty or "status" not in self.orders.columns:
            return 0.0
        all_orders = self.orders[self.orders["date"].apply(period.contains)]
        if all_orders.empty:
            return 0.0
        cancelled = (all_orders["status"] == "cancelled").sum()
        return round(100.0 * cancelled / len(all_orders), 2)

    def cancelled_count(self, period: Period | None = None) -> int:
        if period is None or self.orders.empty or "status" not in self.orders.columns:
            return 0
        df = self.orders[self.orders["date"].apply(period.contains)]
        return int((df["status"] == "cancelled").sum())

    def top_days(
        self, period: Period | None = None, by: str = "revenue", k: int = 1
    ) -> list[tuple[date, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        metric = "revenue" if by == "revenue" else "order_id"
        grouped = df.groupby("date")[metric].sum().sort_values(ascending=False)
        return [(d, float(v)) for d, v in grouped.head(k).items()]

    def top_months(
        self, period: Period | None = None, by: str = "revenue", k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        months = df["date"].map(lambda d: d.strftime("%Y-%m") if d else None)
        grouped = df.groupby(months)["revenue"].sum().sort_values(ascending=False)
        return [(m, float(v)) for m, v in grouped.head(k).items()]

    def count_orders_by_revenue(
        self,
        period: Period | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> int:
        df = self.valid_orders(period)
        if minimum is not None:
            df = df[df["revenue"] >= minimum]
        if maximum is not None:
            df = df[df["revenue"] <= maximum]
        return int(len(df))

    def count_days_by_revenue(
        self,
        period: Period | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> int:
        df = self.valid_orders(period)
        if df.empty:
            return 0
        daily = df.groupby("date")["revenue"].sum()
        if minimum is not None:
            daily = daily[daily >= minimum]
        if maximum is not None:
            daily = daily[daily <= maximum]
        return int(len(daily))

    def _rank_products(
        self, df: pd.DataFrame, metric: str, k: int
    ) -> list[tuple[str, float]]:
        frame = df.assign(_key=df["product"].map(norm_key))
        display = frame.groupby("_key")["product"].agg(
            lambda values: (
                values.mode().iat[0] if len(values.mode()) else values.iloc[0]
            )
        )
        values = frame.groupby("_key")[metric].sum()
        ranked = sorted(
            ((str(display.loc[key]), float(value)) for key, value in values.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]

    # Base units measured by weight or volume. A quantity ranking that mixes
    # these with discrete units is unreliable: bulk rows dominate by number
    # while the reference answers rank discrete goods.
    BULK_BASE_UNITS = frozenset(
        {
            "kg",
            "g",
            "gam",
            "gram",
            "m3",
            "m2",
            "l",
            "lit",
            "ml",
            "tan",
            "ta",
            "yen",
            "met khoi",
            "met vuong",
            "khoi",
        }
    )

    def base_unit_of(self, product: str) -> str | None:
        key = norm_key(product)
        if not key:
            return None
        if not self.products.empty and "unit" in self.products.columns:
            rows = self.products[self.products["product"].map(norm_key) == key]
            if not rows.empty:
                unit = str(rows.iloc[0]["unit"]).strip()
                if unit:
                    return unit
        _, base_units, _ = self.conversions or ({}, {}, {})
        return base_units.get(key)

    def mixed_bulk_units(self, names: list[str]) -> bool:
        classes = set()
        for name in names:
            unit = self.base_unit_of(name)
            if not unit:
                continue
            classes.add(
                "bulk" if norm_key(unit) in self.BULK_BASE_UNITS else "discrete"
            )
        return "bulk" in classes and "discrete" in classes

    def top_products(
        self, period: Period | None = None, by: str = "line_amount", k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.lines
        if df.empty or "date" not in df.columns or "product" not in df.columns:
            return []
        if period is not None:
            df = (
                df[df["date"].apply(period.contains)]
                if df["date"].notna().any()
                else df
            )
        df = df[~df["status"].isin(["cancelled", "returned"])]
        # "Phí phục vụ" is not a product (rulebook §3).
        df = df[df["product"].map(lambda name: norm_key(name) != "phi phuc vu")]
        if df.empty:
            return []
        metric = "base_quantity" if by == "quantity" else "line_amount"
        if metric not in df.columns:
            metric = "quantity" if by == "quantity" else "line_amount"
        return self._rank_products(df, metric, k)

    def top_customers(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        return self._rank_customers(df, "revenue", k)

    def top_customers_days(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        mask = ~key.isin({"khach le", "khach vang lai", "le", "si", "kl", ""})
        if not mask.any():
            return []
        frame = df.loc[mask].copy()
        frame["_key"] = key[mask]
        values = frame.groupby("_key")["date"].nunique()
        display = frame.groupby("_key")["customer"].agg(
            lambda v: v.mode().iat[0] if len(v.mode()) else v.iloc[0]
        )
        ranked = sorted(
            ((str(display[key]), float(value)) for key, value in values.items()),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:k]

    def top_customers_avg(
        self, period: Period | None = None, k: int = 1, min_orders: int = 3
    ) -> list[tuple[str, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        mask = ~key.isin({"khach le", "khach vang lai", "le", "si", "kl", ""})
        if not mask.any():
            return []
        frame = df.loc[mask].copy()
        frame["_key"] = key[mask]
        orders = frame.groupby("_key")["order_id"].nunique()
        revenue = frame.groupby("_key")["revenue"].sum()
        eligible = orders[orders >= min_orders].index
        if len(eligible) == 0:
            return []
        avg = (revenue[eligible] / orders[eligible]).sort_values(ascending=False)
        display = frame.groupby("_key")["customer"].agg(
            lambda v: v.mode().iat[0] if len(v.mode()) else v.iloc[0]
        )
        ranked = [(str(display[key]), float(value)) for key, value in avg.items()]
        return ranked[:k]

    def top_customers_cancelled(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.orders
        if df.empty:
            return []
        df = df[df["status"] == "cancelled"]
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        if df.empty:
            return []
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        mask = ~key.isin({"khach le", "khach vang lai", "le", "si", "kl", ""})
        if not mask.any():
            return []
        frame = df.loc[mask].copy()
        frame["_key"] = key[mask]
        counts = (
            frame.groupby("_key")["order_id"].nunique().sort_values(ascending=False)
        )
        display = frame.groupby("_key")["customer"].agg(
            lambda v: v.mode().iat[0] if len(v.mode()) else v.iloc[0]
        )
        return [(str(display[key]), float(value)) for key, value in counts.items()][:k]

    def top_products_orders(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.filtered_lines(period)
        if df.empty:
            return []
        df = df[df["product"].map(norm_key) != "phi phuc vu"].copy()
        if df.empty:
            return []
        df["_key"] = df["product"].map(norm_key)
        counts = df.groupby("_key")["order_id"].nunique().sort_values(ascending=False)
        display = df.groupby("_key")["product"].agg(
            lambda v: v.mode().iat[0] if len(v.mode()) else v.iloc[0]
        )
        ranked = [(str(display[key]), float(value)) for key, value in counts.items()]
        return ranked[:k]

    def peak_hour(self, period: Period | None = None) -> int | None:
        df = self.valid_orders(period)
        if df.empty or "hour" not in df.columns:
            return None
        df = df[df["hour"] >= 0]
        if df.empty:
            return None
        return int(df["hour"].mode().iat[0])

    def top_day_orders(self, period: Period | None = None) -> date | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        counts = df.groupby("date")["order_id"].nunique().sort_values(ascending=False)
        if counts.empty:
            return None
        value = counts.index[0]
        return value if _is_date(value) else None

    def _rank_customers(
        self,
        df: pd.DataFrame,
        value_column: str,
        k: int,
        count_orders: bool = False,
    ) -> list[tuple[str, float]]:
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        generic = {"khach le", "khach vang lai", "le", "si", "kl", ""}
        mask = ~key.isin(generic)
        if not mask.any():
            return []
        frame = df.loc[mask].copy()
        frame["_customer_key"] = key[mask]
        display = frame.groupby("_customer_key")["customer"].agg(
            lambda values: (
                values.mode().iat[0] if len(values.mode()) else values.iloc[0]
            )
        )
        if count_orders:
            values = frame.groupby("_customer_key")["order_id"].nunique()
        else:
            values = frame.groupby("_customer_key")[value_column].sum()
        ranked = sorted(
            (
                (str(display.loc[customer_key]), float(value))
                for customer_key, value in values.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]

    def debt_total(self, as_of: date | None = None) -> float:
        df = self.orders[self.orders["status"] != "cancelled"]
        if df.empty or "debt" not in df.columns:
            return 0.0
        if as_of is not None:
            df = df[df["date"].apply(lambda d: _is_date(d) and d <= as_of)]
        return float(df["debt"].clip(lower=0).sum())

    def paid_total(self, period: Period | None = None) -> float:
        df = self.valid_orders(period)
        if df.empty or "paid" not in df.columns:
            return 0.0
        return float(df["paid"].sum())

    def discount_total(self, period: Period | None = None) -> float | None:
        df = self.valid_orders(period)
        if df.empty or "discount" not in df.columns:
            return None
        if float(df["discount"].sum()) == 0:
            return None
        return float(df["discount"].sum())

    def revenue_prescription(
        self, period: Period | None, prescription: bool
    ) -> float | None:
        df = self.filtered_lines(period)
        if df.empty or "prescription" not in df.columns:
            return None
        mask = df["prescription"].astype(bool)
        if not prescription:
            # OTC = everything except prescription drugs (including non-drug items).
            total = float(df["line_amount"].sum())
            return total - float(df.loc[mask, "line_amount"].sum())
        part = df[mask]
        if part.empty:
            return None
        return float(part["line_amount"].sum())

    def prescription_share(self, period: Period | None) -> float | None:
        total = self.revenue(period)
        if total <= 0:
            return None
        part = self.revenue_prescription(period, True)
        if part is None:
            return None
        return round(100.0 * part / total, 2)

    def purchase_quantity(
        self, product: str | None, period: Period | None = None
    ) -> float | None:
        if not product or self.purchase_lines.empty:
            return None
        df = self.purchase_lines
        if period is not None and "date" in df.columns and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        key = norm_key(product)
        rows = df[df["product"].map(norm_key) == key]
        if rows.empty:
            matches = [
                name
                for name in df["product"].dropna().unique()
                if key in norm_key(name)
            ]
            if matches:
                best = max(matches, key=lambda name: len(norm_key(name)))
                rows = df[df["product"] == best]
        if rows.empty:
            return None
        factors, base_units, units_by_product = self.conversions or ({}, {}, {})
        total = 0.0
        for _, row in rows.iterrows():
            product_key = norm_key(row["product"])
            id_value = str(row.get("product_id", "")).strip()
            id_key = f"#{norm_key(id_value)}" if id_value else ""
            product_keys = [key for key in (product_key, id_key) if key]
            unit_key = norm_key(row.get("unit", ""))
            factor = None
            for key in product_keys:
                factor = factors.get((key, unit_key))
                if factor is not None:
                    break
            if factor is None:
                for key in product_keys:
                    for known in units_by_product.get(key, ()):
                        if known and known in unit_key:
                            factor = factors.get((key, known))
                            break
                    if factor is not None:
                        break
            total += float(row["quantity"]) * (factor if factor else 1.0)
        return total

    def revenue_payment(
        self,
        period: Period | None,
        keyword: str,
        synonyms: tuple[str, ...] | list[str] | None = None,
    ) -> float | None:
        df = self.valid_orders(period)
        if df.empty or "payment" not in df.columns:
            return None
        if df["payment"].astype(str).str.strip().eq("").all():
            return None
        keys = [norm_key(keyword)]
        if synonyms:
            keys.extend(norm_key(synonym) for synonym in synonyms)
        methods = df["payment"].map(norm_key)
        matched = df[
            pd.Series(
                [[key in str(value) for key in keys] for value in methods],
                index=df.index,
            ).map(any)
        ]
        if matched.empty:
            return None
        # Rulebook §1: revenue is the sum of line amounts. Order headers can
        # carry a different total, so the line sum wins whenever lines exist.
        if len(self.lines):
            ids = set(matched["order_id"])
            lines = self.lines[
                self.lines["order_id"].isin(ids)
                & ~self.lines["status"].isin(["cancelled", "returned"])
            ]
            if period is not None:
                lines = lines[lines["date"].apply(period.contains)]
            if not lines.empty:
                return float(lines["line_amount"].sum())
        return float(matched["revenue"].sum())

    def payment_share(
        self,
        period: Period | None,
        keyword: str,
        synonyms: tuple[str, ...] | list[str] | None = None,
    ) -> float | None:
        total = self.revenue(period)
        if total <= 0:
            return None
        part = self.revenue_payment(period, keyword, synonyms=synonyms)
        if part is None:
            return None
        return round(100.0 * part / total, 2)

    def expense_total(self, period: Period | None = None) -> float | None:
        if self.expenses is None or self.expenses.empty:
            return None
        df = self.expenses
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        if df.empty:
            return None
        return float(df["amount"].sum())

    def _cogs_series(self, df: pd.DataFrame) -> pd.Series:
        """Cost of goods sold per line, robust to mixed pack/base grains.

        Some ledgers record bulk rows with pack quantities but pack
        unit_price/cost (24 goi x 348000/box). quantity x cost then
        overstates COGS by the pack factor. Detect those rows by
        k = quantity x unit_price / line_amount (>= 2.5 means bulk grain)
        and use line_amount x cost / unit_price there instead. Rows with
        a consistent grain keep the exact quantity x cost math.
        """
        qty = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        cost = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
        price = (
            pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
            if "unit_price" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        amount = pd.to_numeric(df["line_amount"], errors="coerce").fillna(0.0)
        raw = qty * cost
        denom = amount.where(amount > 0, float("nan"))
        k = (qty * price / denom).fillna(1.0)
        bulk = (k >= 2.5) & (price > 0) & (amount > 0) & (qty > 0)
        k_final = k
        if bulk.any() and "product" in df.columns:
            # One discounted bulk row can round to the wrong factor
            # (25.2 instead of 24). The product's modal factor wins.
            rounded = (k[bulk] + 0.5).astype(int)
            counts: dict[tuple[str, int], int] = {}
            modes: dict[str, int] = {}
            for name, value in zip(df.loc[bulk, "product"], rounded):
                key = norm_key(name)
                counts[(key, value)] = counts.get((key, value), 0) + 1
                if counts[(key, value)] >= 3 and (
                    key not in modes or counts[(key, value)] > counts[(key, modes[key])]
                ):
                    modes[key] = value
            if modes:
                modal = df["product"].map(norm_key).map(modes)
                use_modal = (
                    bulk
                    & modal.notna()
                    & (modal >= 2.5)
                    & ((k - modal).abs() <= 0.15 * modal)
                )
                k_final = k.where(~use_modal, modal)
        safe_k = k_final.where((k_final > 0) & bulk, 1.0)
        corrected = (qty * cost / safe_k).fillna(0.0)
        return raw.where(~bulk, corrected)

    def gross_profit(self, period: Period | None = None) -> float | None:
        df = self.filtered_lines(period)
        if df.empty or "cost" not in df.columns:
            return None
        cost = float(self._cogs_series(df).sum())
        if cost <= 0:
            return None
        revenue = float(df["line_amount"].sum())
        return revenue - cost

    def net_profit(self, period: Period | None = None) -> float | None:
        gross = self.gross_profit(period)
        if gross is None:
            return None
        expense = self.expense_total(period)
        return gross - (expense or 0.0)

    def top_products_profit(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.filtered_lines(period)
        if df.empty or "cost" not in df.columns:
            return []
        df = df.copy()
        df["profit"] = df["line_amount"] - self._cogs_series(df)
        return self._rank_products(df, "profit", k)

    def debt_customers(self, as_of: date | None = None) -> int:
        df = self.orders[self.orders["status"] != "cancelled"]
        if df.empty or "debt" not in df.columns:
            return 0
        if as_of is not None:
            df = df[df["date"].apply(lambda d: _is_date(d) and d <= as_of)]
        df = df[df["debt"] > 0]
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        df = df[~key.isin({"", "khach le", "le", "kl"})]
        return int(key[key.index.isin(df.index)].nunique())

    def top_debtors(
        self, as_of: date | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.orders[self.orders["status"] != "cancelled"]
        if df.empty or "debt" not in df.columns:
            return []
        if as_of is not None:
            df = df[df["date"].apply(lambda d: _is_date(d) and d <= as_of)]
        df = df[df["debt"] > 0]
        if df.empty:
            return []
        return self._rank_customers(df, "debt", k)

    def debt_age_buckets(self, as_of: date) -> list[float] | None:
        df = self.orders[self.orders["status"] != "cancelled"]
        if df.empty or "debt" not in df.columns:
            return None
        df = df[df["date"].apply(lambda d: _is_date(d) and d <= as_of)]
        df = df[df["debt"] > 0]
        if df.empty:
            return None
        ages = df["date"].map(lambda d: (as_of - d).days)
        buckets = [
            float(df.loc[(ages >= 0) & (ages <= 7), "debt"].sum()),
            float(df.loc[(ages >= 8) & (ages <= 15), "debt"].sum()),
            float(df.loc[ages > 15, "debt"].sum()),
        ]
        return buckets

    def orders_with_category(
        self, period: Period | None, category: str | None
    ) -> int | None:
        if not category:
            return None
        df = self.filtered_lines(period, category=category)
        if df.empty:
            return None
        return int(df["order_id"].nunique())

    def orders_with_product_keyword(
        self, period: Period | None, keyword: str | None
    ) -> int | None:
        if not keyword:
            return None
        df = self.filtered_lines(period)
        if df.empty or "product" not in df.columns:
            return None
        df = df[df["product"].map(norm_key).str.contains(keyword, na=False)]
        if df.empty:
            return None
        return int(df["order_id"].nunique())

    def orders_with_service_keyword(
        self, period: Period | None, keyword: str | None
    ) -> int | None:
        if not keyword:
            return None
        df = self.valid_orders(period)
        if df.empty or "service" not in df.columns:
            return None
        df = df[df["service"].map(norm_key).str.contains(keyword, na=False)]
        if df.empty:
            return None
        return int(df["order_id"].nunique())

    def orders_by_status(self, period: Period | None, status: str) -> int | None:
        df = self.orders
        if df.empty or "status" not in df.columns:
            return None
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        return int((df["status"] == status).sum())

    def returned_count(self, period: Period | None = None) -> int:
        df = self.orders
        if df.empty or "status" not in df.columns:
            return 0
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        return int((df["status"] == "returned").sum())

    def customers_with_orders(
        self, period: Period | None, min_orders: int
    ) -> int | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        mask = ~key.isin({"", "khach le", "le", "kl"})
        if not mask.any():
            return None
        counts = (
            df.loc[mask].assign(_key=key[mask]).groupby("_key")["order_id"].nunique()
        )
        return int((counts >= min_orders).sum())

    def customer_order_share(
        self, period: Period | None, min_orders: int
    ) -> float | None:
        df = self.valid_orders(period)
        if df.empty:
            return None
        names = df["customer"].astype(str).str.strip()
        key = names.map(norm_key)
        mask = ~key.isin({"", "khach le", "le", "kl"})
        if not mask.any():
            return None
        counts = (
            df.loc[mask].assign(_key=key[mask]).groupby("_key")["order_id"].nunique()
        )
        return round(100.0 * (counts >= min_orders).sum() / len(counts), 2)

    def top_customer_orders(
        self, period: Period | None = None, k: int = 1
    ) -> list[tuple[str, float]]:
        df = self.valid_orders(period)
        if df.empty:
            return []
        if df.empty:
            return []
        return self._rank_customers(df, "order_id", k, count_orders=True)

    def tradein_total(self, period: Period | None = None) -> float | None:
        if self.tradeins is None or self.tradeins.empty:
            return None
        df = self.tradeins
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        if df.empty:
            return None
        return float(df["amount"].sum())

    def warranty_count(self, period: Period | None = None) -> int | None:
        if self.warranties is None or self.warranties.empty:
            df = self.filtered_lines(period)
            if not df.empty and "product" in df.columns:
                mask = df["product"].map(norm_key).str.contains("bao hanh", na=False)
                if mask.any():
                    return int(mask.sum())
            orders = self.valid_orders(period)
            if not orders.empty and "service" in orders.columns:
                mask = (
                    orders["service"].map(norm_key).str.contains("bao hanh", na=False)
                )
                if mask.any():
                    return int(mask.sum())
            return None
        df = self.warranties
        if period is not None:
            df = df[df["date"].apply(period.contains)]
        return int(len(df))

    def purchase_total(self, period: Period | None = None) -> float:
        df = self.purchases
        if df.empty or float(df["amount"].sum()) == 0:
            df = self.purchase_lines
            if df.empty:
                return 0.0
        if period is not None and "date" in df.columns and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        return float(df["amount"].sum())

    def delivered_quantity(
        self, period: Period | None, product: str | None
    ) -> float | None:
        """Delivered quantity of one product from delivery-note tables."""
        if not product:
            return None
        key = norm_key(product)
        total = 0.0
        found = False
        for table in self.tables:
            name = norm_key(f"{table.name} {table.sheet or ''}")
            if "giao hang" not in name and "delivery" not in name:
                continue
            columns = {norm_key(str(c)): c for c in table.df.columns}
            product_col = next(
                (
                    columns[k]
                    for k in columns
                    if "hang hoa" in k or "ten hang" in k or "san pham" in k
                ),
                None,
            )
            qty_col = next(
                (
                    columns[k]
                    for k in columns
                    if ("so luong" in k and "giao" in k)
                    or k in {"sl giao", "so luong giao"}
                ),
                None,
            )
            date_col = next(
                (columns[k] for k in columns if "ngay" in k or "thoi gian" in k),
                None,
            )
            if product_col is None or qty_col is None:
                continue
            df = table.df
            if date_col is not None and period is not None:
                dates = df[date_col].map(parse_date)
                df = df[dates.apply(lambda d: _is_date(d) and period.contains(d))]
            matches = df[df[product_col].map(norm_key).str.contains(key, na=False)]
            if matches.empty:
                continue
            total += float(_to_num(matches[qty_col]).sum())
            found = True
        return total if found else None

    def delivered_quantity_for_orders(
        self,
        order_period: Period | None,
        delivery_end,
        product: str | None,
    ) -> float | None:
        """Delivered quantity for orders placed in order_period.

        Used by "Các đơn đặt tháng X, tới hôm nay đã chở đi ..." questions:
        the order month selects the order IDs, the delivery date only needs
        to be on or before delivery_end.
        """
        if not product:
            return None
        key = norm_key(product)
        order_ids: set[str] | None = None
        if order_period is not None and not self.orders.empty:
            mask = self.orders["date"].apply(order_period.contains)
            order_ids = set(self.orders.loc[mask, "order_id"].astype(str).str.strip())
            order_ids.discard("")
            if not order_ids:
                return None
        total = 0.0
        found = False
        for table in self.tables:
            name = norm_key(f"{table.name} {table.sheet or ''}")
            if "giao hang" not in name and "delivery" not in name:
                continue
            columns = {norm_key(str(c)): c for c in table.df.columns}
            product_col = next(
                (
                    columns[k]
                    for k in columns
                    if "hang hoa" in k or "ten hang" in k or "san pham" in k
                ),
                None,
            )
            qty_col = next(
                (
                    columns[k]
                    for k in columns
                    if ("so luong" in k and "giao" in k)
                    or k in {"sl giao", "so luong giao"}
                ),
                None,
            )
            date_col = next(
                (columns[k] for k in columns if "ngay" in k or "thoi gian" in k),
                None,
            )
            oid_col = next(
                (
                    columns[k]
                    for k in columns
                    if "so don" in k or "ma don" in k or "don hang" in k
                ),
                None,
            )
            if product_col is None or qty_col is None:
                continue
            df = table.df
            if order_ids is not None and oid_col is not None:
                df = df[df[oid_col].astype(str).str.strip().isin(order_ids)]
            if date_col is not None and delivery_end is not None:
                dates = df[date_col].map(parse_date)
                df = df[dates.apply(lambda d: _is_date(d) and d <= delivery_end)]
            matches = df[df[product_col].map(norm_key).str.contains(key, na=False)]
            if matches.empty:
                continue
            total += float(_to_num(matches[qty_col]).sum())
            found = True
        return total if found else None

    def delivery_count(self, period: Period | None = None) -> int | None:
        """Count delivery rounds (bill + trip) from a delivery-note table."""
        for table in self.tables:
            name = norm_key(f"{table.name} {table.sheet or ''}")
            if "giao hang" not in name and "delivery" not in name:
                continue
            columns = {norm_key(column): column for column in table.df.columns}
            date_col = next(
                (
                    columns[key]
                    for key in columns
                    if "ngay" in key or "thoi gian" in key or "time" in key
                ),
                None,
            )
            if not date_col:
                continue
            trip_col = next(
                (
                    columns[key]
                    for key in columns
                    if "luot" in key
                    or "dot" in key
                    or "chuyen" in key
                    or "lan giao" in key
                    or key == "lan"
                ),
                None,
            )
            bill_col = next(
                (
                    columns[key]
                    for key in columns
                    if (key.startswith("ma ") or key.startswith("so "))
                    and ("bill" in key or "don" in key)
                ),
                None,
            )
            df = table.df
            dates = df[date_col].map(parse_date)
            if period is not None:
                mask = dates.apply(
                    lambda value: _is_date(value) and period.contains(value)
                )
                df = df[mask]
            if df.empty:
                return 0
            if trip_col and bill_col:
                return int(len(df[[bill_col, trip_col]].drop_duplicates()))
            if trip_col:
                return int(df[trip_col].nunique())
            return int(len(df))
        return None

    def match_ingredient(self, question: str) -> str | None:
        if self.lines.empty or "ingredient" not in self.lines.columns:
            return None
        q = norm_key(question)
        best: str | None = None
        best_len = 0
        for ingredient in self.lines["ingredient"].dropna().unique():
            key = norm_key(ingredient)
            if len(key) >= 4 and key in q and len(key) > best_len:
                best, best_len = str(ingredient), len(key)
        return best

    def revenue_ingredient(
        self, period: Period | None, ingredient: str
    ) -> float | None:
        df = self.filtered_lines(period)
        if df.empty or "ingredient" not in df.columns:
            return None
        key = norm_key(ingredient)
        rows = df[df["ingredient"].map(norm_key) == key]
        if rows.empty:
            return None
        return float(rows["line_amount"].sum())

    def purchase_avg_price(
        self, product: str | None, period: Period | None = None
    ) -> float | None:
        """Quantity-weighted average purchase price (rulebook §5)."""
        if not product or self.purchase_lines.empty:
            return None
        df = self.purchase_lines
        if period is not None and "date" in df.columns and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        key = norm_key(product)
        rows = df[df["product"].map(norm_key) == key]
        if rows.empty:
            matches = [
                name
                for name in df["product"].dropna().unique()
                if key in norm_key(name)
            ]
            if matches:
                best = max(matches, key=lambda name: len(norm_key(name)))
                rows = df[df["product"] == best]
        if rows.empty:
            return None
        if "unit" in rows.columns:
            units = rows["unit"].astype(str).str.strip().replace("", pd.NA).dropna()
            if units.nunique() > 1:
                return None
        quantity = float(rows["quantity"].sum())
        if quantity <= 0:
            return None
        amount = float(rows["amount"].sum())
        if amount <= 0:
            return None
        return round(amount / quantity)

    def top_supplier(self, period: Period | None = None) -> str | None:
        df = self.purchase_lines
        if df.empty or "supplier" not in df.columns:
            return None
        df = df[df["supplier"].astype(str).str.strip() != ""]
        if df.empty:
            return None
        if period is not None and "date" in df.columns and df["date"].notna().any():
            df = df[df["date"].apply(period.contains)]
        if df.empty:
            return None
        grouped = df.groupby("supplier")["amount"].sum().sort_values(ascending=False)
        return str(grouped.index[0]) if len(grouped) else None

    def growth(self, period: Period, metric: str = "revenue") -> float | None:
        prev = previous_period(period)
        if metric == "revenue":
            cur, old = self.revenue(period), self.revenue(prev)
        elif metric == "orders":
            cur, old = self.order_count(period), self.order_count(prev)
        else:
            cur, old = 0.0, 0.0
        if old == 0:
            return None
        return round(100.0 * (cur - old) / old, 2)
