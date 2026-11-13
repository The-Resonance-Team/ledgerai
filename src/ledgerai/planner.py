"""LLM planner: question -> structured plan -> deterministic execution."""

from __future__ import annotations

import json
import re
from datetime import date

from .dataset import Dataset
from .normalize import norm_key
from .periods import Period, resolve_period
from .pipeline import Answer


_SYSTEM = """Bạn là chuyên gia phân tích dữ liệu cho hộ kinh doanh Việt Nam.
Nhiệm vụ: chọn một phép tính trên dữ liệu đã cho. Chỉ trả về JSON, không giải thích.
Nếu dữ liệu không đủ trả lời, trả về {"metric": "abstain", "confidence": 0}.

Schema JSON:
{
  "metric": "revenue|order_count|cancelled_count|cancellation_rate|top_products|top_customers|avg_order|product_quantity|quantity_by_unit|abstain",
  "period_start": "YYYY-MM-DD" | null,
  "period_end": "YYYY-MM-DD" | null,
  "filters": {"product": string|null, "category": string|null, "customer": string|null, "employee": string|null},
  "k": integer,
  "by": "quantity|revenue",
  "answer_type": "number|percent|string|list|date|boolean",
  "confidence": number
}

Luật thời gian (dùng đúng "Hôm nay" trong câu hỏi):
- "hôm nay" = hôm nay; "hôm qua", "bữa qua", "bữa hôm qua" = hôm nay trừ 1 ngày.
- "hôm kia" = hôm nay trừ 2 ngày.
- "tháng này" = ngày 1 của tháng đến hết hôm nay; "tháng trước" = cả tháng liền trước.
- "tuần này" = thứ Hai đến hôm nay; "tuần trước" = thứ Hai đến Chủ nhật tuần trước.
- "N ngày gần nhất" = N ngày liền nhau, ngày cuối là hôm nay.
- "tháng M/YYYY" = cả tháng đó; "quý Q/YYYY" = 3 tháng của quý.
- "từ đầu năm" = 01/01 năm nay đến hôm nay.
- "từ X đến Y" = gồm cả X và Y.

Luật số liệu:
- Doanh thu = tổng thành tiền các dòng, bỏ đơn huỷ và đơn trả hàng; đơn đang giao vẫn tính.
- Số đơn chỉ tính đơn không huỷ, không trả.
- "Đếm số đơn huỷ", "số đơn bị huỷ" -> metric = cancelled_count.
- Tỉ lệ huỷ = đơn huỷ / mọi đơn đặt trong kỳ.
- "Bán được bao nhiêu <đơn vị> X", "mấy <đơn vị> X", "số <đơn vị> X" -> metric = product_quantity, filters.product = X, đáp án là số lượng quy về đơn vị cơ bản.
- "Bán được bao nhiêu <đơn vị>" không nêu mặt hàng -> metric = quantity_by_unit, filters.unit = <đơn vị>; thêm filters.size nếu câu hỏi nêu "size M/L".
- Bán chạy nhất xếp theo số lượng quy về đơn vị cơ bản.
- "Phí phục vụ" không phải mặt hàng: không xếp hạng, không đếm.
- Top k trả về list, k=1 trả về string.
- Tiền làm tròn về đồng; phần trăm làm tròn 2 chữ số.

Luật lọc:
- "mặt hàng X", "sản phẩm X", "hàng X" -> filters.product = X.
- "nhóm hàng X", "loại X" -> filters.category = X.
- "khách X", "khách hàng X" -> filters.customer = X.
- "nhân viên X" -> filters.employee = X.
- Nếu câu hỏi lọc theo thứ không có trong schema (ví dụ "kê đơn", "kênh bán", "hình thức thanh toán", "ca sáng/chiều"), trả về {"metric": "abstain", "confidence": 0}.
"""


def _profile_text(ds: Dataset, max_tables: int = 8) -> str:
    from .profile import profile_table

    lines = [f"Business: {ds.business_id}"]
    for table in ds.tables[:max_tables]:
        cols = [c for c in table.df.columns if not c.startswith("unnamed")]
        roles = profile_table(table).roles
        role_text = ", ".join(f"{k}={v}" for k, v in roles.items())
        lines.append(f"- {table.name} ({len(table.df)} dòng): {', '.join(cols[:14])}")
        if role_text:
            lines.append(f"  vai trò: {role_text}")
        for col in cols[:4]:
            values = table.df[col].replace("", None).dropna().head(3).tolist()
            if values:
                lines.append(f"    {col}: {values}")
    if len(ds.orders):
        dates = ds.orders["date"].dropna()
        if len(dates):
            lines.append(f"- Khoảng ngày đơn: {dates.min()} .. {dates.max()}")
    return "\n".join(lines)


def build_prompt(question: str, as_of: date, ds: Dataset) -> list[dict]:
    user = (
        f"Hôm nay: {as_of.isoformat()}\n"
        f"Câu hỏi: {question}\n\n"
        f"Dữ liệu:\n{_profile_text(ds)}\n\n"
        "Trả về JSON."
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse_plan(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return plan if isinstance(plan, dict) else None


def _period_from_plan(plan: dict, as_of: date) -> Period | None:
    start = plan.get("period_start")
    end = plan.get("period_end")
    if start and end:
        try:
            return Period(date.fromisoformat(start), date.fromisoformat(end), "plan")
        except ValueError:
            return None
    if start:
        try:
            d = date.fromisoformat(start)
            return Period(d, d, "plan")
        except ValueError:
            return None
    expression = plan.get("period") or plan.get("time")
    return resolve_period(expression, as_of) if expression else None


def _filters(plan: dict) -> dict:
    raw = plan.get("filters") or {}
    return {
        key: raw.get(key)
        for key in ("product", "category", "customer", "employee", "unit", "size")
        if isinstance(raw.get(key), str) and raw.get(key).strip()
    }


def execute_plan(plan: dict, as_of: date, ds: Dataset) -> Answer:
    metric = str(plan.get("metric") or "abstain").lower()
    confidence = float(plan.get("confidence") or 0.0)
    if metric == "abstain":
        return Answer("abstain", confidence=confidence, plan="llm_abstain")

    period = _period_from_plan(plan, as_of)
    filters = _filters(plan)
    k = int(plan.get("k") or 1)
    by = str(plan.get("by") or "quantity").lower()

    try:
        if metric == "revenue":
            value = ds.revenue_filtered(
                period,
                product=filters.get("product"),
                category=filters.get("category"),
                customer=filters.get("customer"),
                employee=filters.get("employee"),
            )
            return Answer("number", int(round(value)), confidence, "llm_revenue")
        if metric == "order_count":
            df = ds.filtered_orders(
                period,
                customer=filters.get("customer"),
                employee=filters.get("employee"),
            )
            return Answer("number", int(len(df)), confidence, "llm_order_count")
        if metric == "cancelled_count":
            return Answer(
                "number", ds.cancelled_count(period), confidence, "llm_cancelled_count"
            )
        if metric == "product_quantity":
            product = filters.get("product")
            if not product:
                return Answer(
                    "abstain", confidence=confidence, plan="llm_product_quantity"
                )
            value = ds.product_quantity(period, product=product)
            if value is None:
                return Answer(
                    "abstain", confidence=confidence, plan="llm_product_quantity"
                )
            return Answer(
                "number", int(round(value)), confidence, "llm_product_quantity"
            )
        if metric == "quantity_by_unit":
            value = ds.quantity_by_unit(
                period, unit=filters.get("unit"), size=filters.get("size")
            )
            if value is None:
                return Answer(
                    "abstain", confidence=confidence, plan="llm_quantity_by_unit"
                )
            return Answer(
                "number", int(round(value)), confidence, "llm_quantity_by_unit"
            )
        if metric == "cancellation_rate":
            if period is None:
                return Answer("abstain", confidence=confidence, plan="llm_cancel")
            return Answer(
                "percent", ds.cancellation_rate(period), confidence, "llm_cancel_rate"
            )
        if metric == "top_products":
            top = ds.top_products_filtered(
                period, by=by, k=k, category=filters.get("category")
            )
            if not top:
                return Answer("abstain", confidence=confidence, plan="llm_top_products")
            if k > 1:
                return Answer(
                    "list", [n for n, _ in top], confidence, "llm_top_products"
                )
            return Answer("string", top[0][0], confidence, "llm_top_product")
        if metric == "top_customers":
            top = ds.top_customers(period, k=k)
            if not top:
                return Answer(
                    "abstain", confidence=confidence, plan="llm_top_customers"
                )
            if k > 1:
                return Answer(
                    "list", [n for n, _ in top], confidence, "llm_top_customers"
                )
            return Answer("string", top[0][0], confidence, "llm_top_customer")
        if metric == "avg_order":
            df = ds.filtered_orders(period, customer=filters.get("customer"))
            if df.empty:
                return Answer("abstain", confidence=confidence, plan="llm_avg")
            value = float(df["revenue"].sum()) / len(df)
            return Answer("number", int(round(value)), confidence, "llm_avg_order")
    except Exception:
        return Answer("abstain", confidence=0.0, plan="llm_error")

    return Answer("abstain", confidence=confidence, plan="llm_unsupported")
