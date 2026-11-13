# Glossary

- **DataMate**: track của Zalo AI Challenge 2026; hỏi đáp tiếng Việt trên CSV/Excel/SQLite.
- **PA (Penalized Accuracy)**: đúng +1, sai −1, abstain 0; chia cho số câu.
- **Rulebook**: luật mặc định của split; quy định doanh thu, đơn huỷ, công nợ, tồn kho, giá vốn, thời gian, xếp hạng.
- **Profiler**: bước đọc file, tạo schema card (sheet, cột, kiểu, ngày, đơn vị).
- **Metric library**: các hàm tính toán deterministic (doanh thu, top sản phẩm, công nợ...).
- **Planner**: model chọn metric/filter/time range/group by/top k, xuất JSON plan.
- **Executor**: chạy plan/code, trả kết quả.
- **Verifier**: kiểm tra type, format, đơn vị, ngày, thứ tự ranking trước khi trả lời.
- **Abstain**: bỏ trống, 0 điểm; dùng khi không chắc.
- **Unanswerable**: dữ liệu không đủ để trả lời; type riêng.
- **Holdout**: tập test nội bộ, giữ nguyên business/ngành để giả lập private.
- **vLLM**: inference server cho LLM; cần bản hỗ trợ Gemma 4 và RTX 5090.
- **Docker image**: gói inference offline; chạy bằng `predict.sh`.
- **Blackwell**: kiến trúc GPU RTX 5090 (sm_120); A6000 là Ampere (sm_86).
- **Bake-off**: so sánh 2–3 model trên cùng tập câu hỏi nhỏ.
- **QA review**: content manager kiểm tra câu hỏi/đáp án trước khi vào eval set.
