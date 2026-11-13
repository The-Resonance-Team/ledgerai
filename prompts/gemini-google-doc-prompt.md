# Prompt gửi Gemini — tạo Google Doc cho team

Copy toàn bộ nội dung trong khung dưới đây, gửi cho Gemini (Google Workspace). Nếu Gemini có công cụ tạo Google Docs, nó sẽ tạo trực tiếp; nếu không, nó sẽ trả về nội dung đầy đủ để dán vào Google Docs.

---

```text
Bạn là trợ lý tài liệu dự án cho team tham gia Zalo AI Challenge 2026.

NHIỆM VỤ
Tạo một Google Doc tiếng Việt có tiêu đề:
"LedgerAI — Kế hoạch & Quyết định"

Tài liệu dành cho toàn bộ thành viên team (2 kỹ sư phần mềm + 1 content manager), gồm cả người không chuyên kỹ thuật. Văn phong rõ ràng, ngắn gọn, có bảng. Không tự bịa thông tin; chỗ nào chưa chắc ghi "Cần xác nhận".

BỐI CẢNH DỰ ÁN
- Tên dự án: LedgerAI.
- Tagline: "Ask your ledger anything."
- Mô tả: trợ lý dữ liệu tiếng Việt cho hộ kinh doanh nhỏ; trả lời câu hỏi trên CSV/Excel/SQLite bẩn.
- Cuộc thi: Zalo AI Challenge 2026 — track DataMate.
- Mục tiêu: top 1; mục tiêu thực tế: điểm private test cao, top 5; một entry hợp lệ, reproducible.
- Điểm số: penalized accuracy. Đúng +1, sai −1, abstain 0.
- Chiến lược: coverage cao + verifier; không đoán bừa.

LUẬT CUỘC THI QUAN TRỌNG
- Chỉ chọn track DataMate; không làm LifeMate; không làm cả hai.
- Tổng parameters của mọi model inference ≤15B.
- Model weights phải release trước 26/10/2026; license cho phép dùng.
- Inference offline trong Docker; không API, không download.
- Toàn bộ private test ≤6 giờ trên 2×RTX 5090, 512GB RAM.
- Docker image linux/amd64, uncompressed ≤100GB.
- Logs mỗi câu: id, answer, elapsed_ms, thinking, code, output.
- Tối đa 2 Docker submissions; organizer chấm bản cuối.
- Public test: 5 submission/ngày.
- Test Solution đóng 15/11/2026 23:59 GMT+7.
- Team edit lock: 16/11/2026 00:00 GMT+7.
- Docker deadline: 16/11/2026 23:59 GMT+7.
- Private test: 20–23/11/2026.
- Công bố top 2: 02/12/2026. Kết quả cuối: 12/12/2026.

DỮ LIỆU
- Train: 3.000 câu, 30 business, có answers.jsonl.
- Public test: 1.000 câu, 14 business, không có đáp án.
- Private test: khoảng 500 câu; có ngành chưa xuất hiện trong train/public.
- Mỗi business: 12–30 tháng dữ liệu; vài trăm đến ~300.000 dòng.
- Format: CSV, XLSX, SQLite; README chất lượng không đều.
- Rulebook + calendar dùng chung cho mỗi split.
- Answer types: number, percent, date, boolean, string, list, table, unanswerable, abstain.
- Không dùng public/private test để train hoặc lấy đáp án.
- QA pack trên Google Drive: https://drive.google.com/drive/folders/1_ISBOoqWUD0hzZSt8YSjb7ck4LVdQICb?usp=sharing

QUYẾT ĐỊNH ĐÃ CHỐT
1. Track: DataMate only.
2. Kiến trúc: D1 + verifier. Pipeline: profiler → question classifier → planner → metric executor → verifier → formatter/logger; fallback code generation.
3. Model chính: Gemma 4 12B.
   - 11.96B parameters; Apache 2.0; 140+ ngôn ngữ; native function calling; 256K context.
   - Release 23/05/2026; trước cutoff 26/10/2026.
   - Cần vLLM bản hỗ trợ Gemma 4 (v0.31.0 hoặc mới hơn).
4. Fallback: Qwen3.5-9B.
   - 9.65B; Apache 2.0; 201 ngôn ngữ; agentic/coding; 262K context.
   - Release 27/02/2026.
5. Không dùng Qwen3.8-27B: 28B > 15B cap. Quantization không đổi số parameter.
6. Không fine-tune trong 48 giờ đầu.
7. Holdout theo business, không random theo câu, để giả lập private test có ngành mới.
8. QA pack lấy từ Zalo train (3.000 câu), không phải dữ liệu tự tạo.
9. Synthetic QA sinh sau khi có metric library; đáp án phải tính bằng code; content manager review trước khi dùng.
10. Dev trên ssh bvpt3: NVIDIA GB10, 20 CPU, 121GB RAM, Docker 29.2.1, PyTorch 2.13 + CUDA 13.0, ARM64.
11. Test image trên ThueGPU.vn: RTX A6000 48GB (Ampere), gần nhất với 5090 trong bảng giá. Không có 5090 trên ThueGPU.
12. Chấp nhận rủi ro không test Blackwell (sm_120); A6000 chỉ là proxy. Thiết kế chạy trên 1×5090 (32GB) để giảm phụ thuộc tensor parallel.
13. Freeze code 15/11; nộp Docker trước 16/11 23:59.
14. Tối đa 2 Docker submissions; bản cuối được chấm.

TEAM
- SWE A — Data: profiler, metric library, unit conversion, rulebook edge cases, SQLite/Excel.
- SWE B — Agent/Infra: vLLM, planner, executor, verifier, Docker, logs, determinism, submission.
- Content manager: taxonomy, QA review, test cases, manual QA, README/approach, checklist.
- Nhịp ngày: 09:00 standup; 13:00 eval; 18:00 error review; 22:00 freeze commit.

KẾ HOẠCH 4 NGÀY
- 12/11: setup repo, data, QA pack; chạy Gemma 4 12B trên bvpt3; smoke test 20 câu; profiler skeleton; eval harness.
- 13/11: metric library core (doanh thu, đơn, huỷ, top sản phẩm, công nợ, tồn kho, nhập hàng); pipeline end-to-end; PA holdout > 0.5.
- 14/11: verifier, type/format, unanswerable/abstain, error analysis, test business holdout; PA holdout > 0.7; format error < 2%.
- 15/11: freeze feature; full train/public; build Docker linux/amd64; test A6000 48GB; logs; determinism; nộp public; Test Solution trước 23:59.
- 16/11: kiểm tra cuối; docker save + gzip + MD5; upload Google Drive; kiểm tra link; nộp trước 23:59; không sửa code.

RỦI RO
- Rulebook nhiều luật → encode thành hàm, test bằng train.
- Excel bẩn → profiler + schema card + fallback.
- Ngành mới ở private → holdout theo business; không hardcode.
- Gemma 4 lỗi vLLM/Blackwell → test A6000; fallback Qwen3.5-9B.
- Thời gian 4 ngày → freeze 15/11; ưu tiên metric phổ biến.
- Docker image quá lớn → chỉ copy weights + code; nén.
- Team coordination → standup 9:00; freeze commit 22:00.

CẤU TRÚC TÀI LIỆU YÊU CẦU
1. Trang bìa: tên dự án, track, team, ngày, trạng thái.
2. Executive summary 1 trang: mục tiêu, cách làm, mốc chính, rủi ro lớn nhất.
3. Mục tiêu & tiêu chí thành công.
4. Track & luật cuộc thi (bảng mốc thời gian).
5. Quyết định model (bảng so sánh Gemma 4 12B, Qwen3.5-9B, Qwen3.8-27B, Nemotron Bolt, Granite 4.2).
6. Kiến trúc D1 + verifier (sơ đồ text + giải thích từng bước).
7. Dữ liệu & QA (train/public/private; QA pack; link Google Drive; synthetic QA).
8. Team & phân công (bảng owner/deliverable).
9. Timeline 4 ngày (bảng ngày → việc → milestone → owner).
10. Kế hoạch từng ngày (chi tiết).
11. Rủi ro & giảm thiểu (bảng mức độ/owner).
12. Checklist nộp bài (Docker, logs, MD5, Drive, deadline).
13. Decision log (bảng quyết định, lý do, trạng thái).
14. Việc cần chốt (open questions, deadline, owner).
15. Glossary cho thành viên không chuyên.

YÊU CẦU ĐỊNH DẠNG
- Tiếng Việt, giữ thuật ngữ kỹ thuật tiếng Anh khi cần và giải thích ngắn.
- Dùng heading, bảng, bullet; không viết đoạn quá dài.
- Mỗi bảng có cột Owner và Deadline nếu phù hợp.
- Đánh dấu rõ "Đã chốt" và "Cần chốt".
- Không đưa dữ liệu private/test, API key, thông tin cá nhân lên tài liệu.
- Ghi rõ: "Tài liệu nội bộ, không chia sẻ ra ngoài team."
- Trong mục Dữ liệu & QA, chèn link QA pack: https://drive.google.com/drive/folders/1_ISBOoqWUD0hzZSt8YSjb7ck4LVdQICb?usp=sharing

OUTPUT
Nếu bạn có công cụ tạo Google Docs: tạo trực tiếp Google Doc với tiêu đề trên và chia sẻ cho tôi.
Nếu không: trả về toàn bộ nội dung tài liệu dưới dạng Markdown, sẵn sàng để dán vào Google Docs.
```
