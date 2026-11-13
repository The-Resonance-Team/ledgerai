# QA review pack — DataMate

**Nguồn dữ liệu:** 3.000 câu trong `train_review.csv` là **train split của Zalo AI Challenge 2026 — DataMate**, không phải dữ liệu do team tự tạo. Sample 523 câu, taxonomy và các cột heuristic do team tạo từ dữ liệu train.

Mục đích: kiểm duyệt câu hỏi và đáp án trước khi đưa vào eval set. Không dùng public/private test để train hoặc lấy đáp án.

## File

- `train_review.csv` — toàn bộ 3.000 câu train, kèm đáp án gold, type, business, `as_of_date`, alias, intent heuristic và cột review.
- `train_review_sample.md` — 523 câu stratified để review nhanh trước.
- `taxonomy.md` — thống kê answer type, intent family, business type.

## Cột review trong CSV

| Cột | Giá trị |
|---|---|
| `review_status` | `pending` / `approved` / `fixed` / `rejected` |
| `reviewer_note` | Ghi chú lỗi, cách sửa, lý do reject |
| `reviewed_by` | Tên người review |
| `reviewed_at` | Ngày review, dạng `YYYY-MM-DD` |

## Quy trình

1. Đọc `train_review_sample.md`. Đánh dấu các câu sai/khó hiểu.
2. Mở `train_review.csv`, lọc `review_status = pending`.
3. Kiểm tra:
   - Câu hỏi có rõ nghĩa không? Có nhiều cách hiểu không?
   - `answer_type` có đúng không?
   - `answer_json` có khớp dữ liệu và rulebook không?
   - `alias_json` có thiếu tên/mã hợp lệ không?
   - `business_type` và `intent_family` có đúng không? Nếu sai, sửa.
4. Đặt `review_status`:
   - `approved` — dùng được.
   - `fixed` — đã sửa câu hỏi/đáp án; ghi rõ trong `reviewer_note`.
   - `rejected` — không dùng; ghi lý do.
5. Không sửa trực tiếp file gốc trong `data/`. Chỉ sửa trong bản review.

## Synthetic QA

Sau khi có metric library, sẽ sinh thêm:

- Paraphrase: không dấu, telex, cách nói thân mật.
- Câu hỏi khó: nhiều bước, nhiều bảng, đơn vị quy đổi.
- Câu hỏi unanswerable: thiếu cột, thiếu dữ liệu, ngoài khoảng ngày.

Mỗi câu synthetic phải có đáp án tính bằng code, không do model tự đoán. Tất cả vào `qa/synthetic/` với cùng cột review. Chỉ câu `approved` mới vào eval set.
