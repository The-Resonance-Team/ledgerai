# Timeline — LedgerAI

Múi giờ: GMT+7.

## Mốc cuộc thi

| Ngày | Việc |
|---|---|
| 26/10/2026 | Cuộc thi bắt đầu; data + dev kit |
| **15/11/2026 23:59** | Test Solution đóng |
| **16/11/2026 00:00** | Khóa sửa team |
| **16/11/2026 23:59** | Nộp Docker + code + model |
| 20/11/2026 09:00 | Private test release |
| 23/11/2026 23:59 | Nộp kết quả private test |
| 02/12/2026 | Công bố top 2 mỗi track |
| 12/12/2026 | Công bố kết quả cuối |

## Tiến độ thực tế (12/11)

- ✅ Profiler: CSV/Excel/SQLite, header detection, role inference.
- ✅ Metric library: doanh thu, đơn, huỷ, top sản phẩm, top khách, trung bình đơn.
- ✅ Eval harness: holdout theo business, PA, coverage, precision, per-type.
- ✅ Baseline holdout: **PA 0.192**, coverage 43.0%, precision 72.3%.
- ✅ LLM planner + verifier (80 câu): **PA 0.3125**, coverage 73.8%, precision 71.2%.
- ✅ Dockerfile linux/amd64 build thành công trên bvpt3 (QEMU), 49.4GB.
- ⏳ Test Docker native trên A6000 (15/11).
- ⏳ Cải thiện: debt, inventory, purchase, filtered revenue, multi-step.

**Mục tiêu điều chỉnh:** PA holdout > 0.4 (14/11), > 0.5 (15/11).

## Kế hoạch 4 ngày

### 12/11 — Setup + baseline

- Tạo repo, data, QA pack.
- Chạy Gemma 4 12B trên `bvpt3`.
- Smoke test 20 câu.
- Profiler skeleton: đọc CSV/XLSX/SQLite, schema card.
- Eval harness: PA, type accuracy, format error.

**Milestone:** 1 câu chạy end-to-end; biết model load được.

### 13/11 — Metric library + pipeline

- Metric core: doanh thu, đơn, huỷ, top sản phẩm, công nợ, tồn kho, nhập hàng.
- Planner DSL + executor.
- Chạy 300 câu holdout.
- Đo PA, lỗi format, latency.

**Milestone:** PA holdout > 0.5.

### 14/11 — Verifier + long-tail

- Verifier: type/format/unit/date/ranking.
- Unanswerable/abstain.
- Error analysis theo intent family.
- Test trên business holdout.
- Fallback code generation + repair.

**Milestone:** PA holdout > 0.7; format error < 2%.

### 15/11 — Freeze + Docker

- Freeze feature.
- Chạy full train/public.
- Build Docker linux/amd64.
- Test trên A6000 48GB.
- Logs, determinism, 2 lần chạy.
- Nộp public test (5 submission/ngày).
- Test Solution trước 23:59.

**Milestone:** Docker chạy hết public test; submission.jsonl đủ câu.

### 16/11 — Nộp cuối

- Kiểm tra Docker lần cuối.
- `docker save` + gzip + MD5.
- Upload Google Drive; kiểm tra link từ máy khác.
- Nộp Docker trước 23:59.
- Không sửa code sau freeze.

**Milestone:** Docker + MD5 + link Drive đã gửi.

## Nhịp ngày

| Giờ | Việc |
|---|---|
| 09:00 | Standup 15 phút |
| 13:00 | Eval run + báo cáo |
| 18:00 | Error review |
| 22:00 | Freeze commit, cập nhật decision log |
