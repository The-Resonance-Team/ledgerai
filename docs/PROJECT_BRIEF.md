# Project Brief — LedgerAI

## 1. Tổng quan

**LedgerAI** là một AI agent trả lời câu hỏi tiếng Việt trực tiếp trên dữ liệu bán hàng, kho, công nợ của hộ kinh doanh nhỏ. Dự án tham gia **Zalo AI Challenge 2026 — DataMate track**.

Tên dự án: **LedgerAI**. Tagline: **“Ask your ledger anything.”**

Điểm khác biệt: agent không chỉ viết SQL/pandas. Nó có một **metric library** mã hoá rulebook của cuộc thi, một **planner** chọn metric/filter/time range, và một **verifier** kiểm tra type/format/đơn vị trước khi trả lời. Khi không chắc, agent abstain thay vì đoán.

## 2. Mục tiêu

- **Mục tiêu chính:** top 1 track DataMate.
- **Mục tiêu thực tế:** điểm private test cao, top 5; một entry hợp lệ, reproducible.
- **Điểm số:** penalized accuracy. Đúng +1, sai −1, abstain 0.
- **Chiến lược:** coverage cao + verifier; không đoán bừa.

## 3. Luật cuộc thi quan trọng

- Track: DataMate only. Không làm LifeMate.
- Model: tổng parameters ≤15B; weights release trước **26/10/2026**; license cho phép dùng.
- Inference offline trong Docker; không API, không download.
- Runtime: toàn bộ private test ≤6 giờ trên **2×RTX 5090**, 512GB RAM.
- Docker image: linux/amd64, uncompressed ≤100GB.
- Logs: mỗi câu có `id`, `answer`, `elapsed_ms`, `thinking`, `code`, `output`.
- Submission: tối đa 2 Docker submissions; organizer chấm bản cuối.
- Public test: 5 submission/ngày.
- Test Solution đóng **15/11/2026 23:59 GMT+7**.
- Team edit lock: **16/11/2026 00:00 GMT+7**.
- Docker deadline: **16/11/2026 23:59 GMT+7**.
- Private test: **20–23/11/2026**.
- Top 2: **02/12/2026**. Final: **12/12/2026**.

## 4. Dữ liệu

- **Train:** 3.000 câu, 30 business, có `answers.jsonl`.
- **Public test:** 1.000 câu, 14 business, không có đáp án.
- **Private test:** khoảng 500 câu, có ngành chưa xuất hiện trong train/public.
- Mỗi business: 12–30 tháng dữ liệu; vài trăm đến ~300.000 dòng.
- Format: CSV, XLSX, SQLite; README chất lượng không đều.
- Rulebook và calendar dùng chung cho mỗi split.
- Answer types: number, percent, date, boolean, string, list, table, unanswerable, abstain.

## 5. Model

**Chính: Gemma 4 12B**

- 11.96B parameters; ≤15B.
- Apache 2.0.
- 140+ ngôn ngữ; native function calling; agentic/coding.
- 256K context.
- Release: 23/05/2026 (trước cutoff 26/10/2026).
- vLLM: cần bản hỗ trợ Gemma 4 (v0.31.0 hoặc mới hơn).

**Fallback: Qwen3.5-9B**

- 9.65B; Apache 2.0; 201 ngôn ngữ; agentic/coding; 262K context.
- Release: 27/02/2026.
- Dùng nếu Gemma 4 yếu tiếng Việt hoặc lỗi vLLM.

**Không dùng:**

- Qwen3.8-27B: 28B > 15B cap.
- Qwen3.8-2.4T: quá lớn.
- Granite 4.2 8B / Ling-3.0-tiny / Nemotron Bolt: release H2 2026 nhưng không list tiếng Việt; chỉ dùng nếu bake-off chứng minh tốt hơn.

**Cap:**

- Gemma 4 12B + embedding nhỏ (<1B) vẫn ≤15B.
- Nếu dùng BM25 thì không cần embedding, tiết kiệm VRAM.

## 6. Kiến trúc D1 + verifier

1. **Profiler**: đọc file, tạo schema card (sheet, cột, kiểu, ngày, trạng thái, đơn vị, quy đổi).
2. **Question classifier**: intent family, answer type, time range, filter, group by, top k.
3. **Planner (Gemma 4 12B)**: xuất JSON plan theo DSL cố định; nếu không khớp metric nào → sinh Python code.
4. **Executor**: hàm deterministic cho metric; pandas/sqlite; timeout; bắt lỗi.
5. **Verifier**: type/format, đơn vị, khoảng ngày, thứ tự ranking, tie-break theo mã; nếu nghi ngờ → abstain.
6. **Formatter/Logger**: `submission.jsonl` + `logs/{id}.json`.
7. **Fallback**: code generation + repair 2–3 lần; vẫn lỗi → abstain.

## 7. Metric library ưu tiên

- Doanh thu, số đơn, tỉ lệ huỷ.
- Top sản phẩm, doanh thu theo nhóm.
- Công nợ, tuổi nợ, khách nợ nhiều nhất.
- Tồn kho, giá trị tồn, hàng hết hạn, hàng ế.
- Nhập hàng, giá nhập bình quân, giá vốn, lãi gộp, lãi ròng.
- Xếp hạng khách/nhân viên.
- Quy đổi đơn vị, lịch âm/Tết.

## 8. Team

- **SWE A — Data:** profiler, metric library, unit conversion, rulebook edge cases, SQLite/Excel.
- **SWE B — Agent/Infra:** vLLM, planner, executor, verifier, Docker, logs, determinism, submission.
- **Content manager:** taxonomy, QA review, test cases, manual QA, README/approach, checklist.

## 9. Hạ tầng

- **Dev:** `ssh bvpt3` — NVIDIA GB10, 20 CPU, 121GB RAM, Docker 29.2.1, PyTorch 2.13 + CUDA 13.0, ARM64.
- **Test image:** ThueGPU.vn — **RTX A6000 48GB** (Ampere), gần nhất với 5090 trong bảng giá.
- **Máy chấm:** 2×RTX 5090 (Blackwell), 512GB RAM.
- **Rủi ro đã chấp nhận:** A6000 không phải Blackwell; không test được sm_120. Giảm rủi ro bằng cách chạy 1×5090-compatible (model fit 32GB), dùng vLLM bản hỗ trợ Blackwell, test kỹ trên A6000.
- **ThueGPU:** A6000 48GB giá 40.000 VND/giờ. Đặt 4–6 giờ ngày 15/11.

## 10. QA

- `qa/train_review.csv`: **3.000 câu từ Zalo train**, không phải dữ liệu tự tạo.
- `qa/train_review_sample.md`: 523 câu stratified.
- `qa/taxonomy.md`: thống kê type/intent/business.
- Content manager review, đánh `approved/fixed/rejected`.
- Synthetic QA sinh sau khi có metric library; đáp án tính bằng code; review trước khi dùng.

## 11. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Rulebook nhiều luật | Cao | Encode thành hàm; test bằng train |
| Excel bẩn | Cao | Profiler + schema card + fallback |
| Ngành mới ở private | Cao | Holdout theo business; không hardcode |
| Gemma 4 lỗi vLLM/Blackwell | Cao | Test A6000; fallback Qwen3.5-9B |
| Thời gian 4 ngày | Cao | Freeze 15/11; ưu tiên metric phổ biến |
| Docker image quá lớn | Trung bình | Chỉ copy weights + code; nén |
| Team coordination | Trung bình | Standup 9:00; freeze commit 22:00 |

## 12. Open questions

- vLLM version chính xác cho Gemma 4 trên A6000/5090.
- Lịch thuê A6000 ngày 15/11.
- Ai review QA full 3.000 câu.
- Ai build/test Docker.
- Ngân sách thuê GPU.
