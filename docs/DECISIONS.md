# Decision Log

| # | Quyết định | Lý do | Owner | Ngày | Trạng thái |
|---|---|---|---|---|---|
| D01 | Chọn track **DataMate**, không làm LifeMate | Không cần ASR/audio; 15B cap; 6 giờ; code execution; phù hợp 4 ngày | Team | 12/11 | Chốt |
| D02 | Không làm cả hai track | 4 ngày không đủ; split team là rủi ro | Team | 12/11 | Chốt |
| D03 | Mục tiêu top 1; chiến lược coverage + verifier | Top 1 cần coverage cao, nhưng sai bị −1 | Team | 12/11 | Chốt |
| D04 | Kiến trúc **D1 + verifier** | Rulebook-grounded, ít sai format, debug được | Team | 12/11 | Chốt |
| D05 | Model chính **Gemma 4 12B** | 11.96B ≤15B; Apache 2.0; 140+ ngôn ngữ; native function calling; release 23/05/2026 | Team | 12/11 | Chốt |
| D06 | Fallback **Qwen3.5-9B** | 9.65B; Apache 2.0; 201 ngôn ngữ; mạnh tiếng Việt | Team | 12/11 | Chốt |
| D07 | Không dùng **Qwen3.8-27B** | 28B > 15B cap | Team | 12/11 | Chốt |
| D08 | Không fine-tune trong 48h đầu | Ưu tiên pipeline; fine-tune chỉ khi còn thời gian | Team | 12/11 | Chốt |
| D09 | Holdout theo **business**, không random theo câu | Private test có ngành mới | Team | 12/11 | Chốt |
| D10 | QA pack lấy từ **Zalo train**, không phải dữ liệu tự tạo | Train có gold answer; được phép dùng | Team | 12/11 | Chốt |
| D11 | Synthetic QA sinh sau khi có metric library | Đáp án phải tính bằng code, không model đoán | Team | 12/11 | Chốt |
| D12 | Dev trên **bvpt3** (GB10, ARM64) | Có sẵn GPU, Docker, 121GB RAM | Team | 12/11 | Chốt |
| D13 | Test image trên **A6000 48GB** tại ThueGPU | Gần nhất với 5090 trong bảng giá; không có 5090 | Team | 12/11 | Chốt |
| D14 | Chấp nhận rủi ro không test Blackwell | User quyết định không cần 5090; A6000 là proxy | Team | 12/11 | Chốt |
| D15 | Thiết kế chạy trên **1×5090** (32GB) | Gemma 4 12B bf16 ~24GB; giảm phụ thuộc tensor parallel | Team | 12/11 | Chốt |
| D16 | Docker deadline 16/11 23:59; freeze code 15/11 | Cần thời gian build/test/upload | Team | 12/11 | Chốt |
| D17 | Tối đa 2 Docker submissions; chấm bản cuối | Luật cuộc thi | Team | 12/11 | Chốt |
| D18 | Tên dự án: **LedgerAI** | Tiếng Anh, dễ hiểu; không có team nào trên leaderboard Zalo dùng; chấp nhận trùng tên sản phẩm bên ngoài | Team | 12/11 | Chốt |
| D19 | Chỉ nộp **SLM** (≤15B, offline); không nộp LLM/API | Leaderboard ghi rõ: xếp hạng chỉ dựa trên SLM; LLM (>15B) và API chỉ để tham khảo; luật cấm API ngoài khi inference | Team | 12/11 | Chốt |
| D20 | Base image `vllm/vllm-openai:v0.33.0` | CUDA 13.0, vLLM 0.33.0, torch 2.14.0; hỗ trợ Blackwell; build linux/amd64 thành công, image 49.4GB | SWE B | 12/11 | Chốt |
| D21 | LLM planner + **verifier** trước khi trả lời | LLM tăng coverage nhưng precision giảm; verifier lọc period/type/filter | Team | 12/11 | Chốt |
| D22 | Bản nộp mặc định **bật LLM + verifier** (`LEDGERAI_USE_LLM=1`) | Router deterministic đạt PA 0.8915, precision 100%; LLM+verifier full 553 đạt **0.9078**, precision 100%; schema **0.8481**; tự fallback router nếu model lỗi | Team | 13/11 | Chốt |

## Leaderboard categories (12/11/2026)

Public leaderboard có 3 cột `Score/Model`:

| Cột | Nghĩa | Có tính điểm xếp hạng? |
|---|---|---|
| SLM | Model nhỏ, ≤15B | ✅ Có — xếp hạng chỉ dựa trên SLM |
| LLM | Model open-source lớn, >15B | ❌ Không — chỉ tham khảo |
| APIs/Others | Gọi API bên thứ ba (Claude, Gemini, OpenAI...) | ❌ Không — chỉ tham khảo |

Ghi chú chính thức trên portal:

> Public Leaderboard rankings are based solely on SLM scores. LLM and APIs/Others results are for reference only. The valid solution must satisfy the rules of each track and the competition.

Vì vậy:

- Các team hiện trên cột LLM/API là **nộp tham khảo**, không hợp lệ cho điểm cuối.
- Bài nộp cuối cùng (Docker) phải là SLM ≤15B, chạy offline, không gọi API.
- LedgerAI chỉ làm SLM. Không cần thêm LLM/API.

## Open decisions

| # | Câu hỏi | Deadline | Owner |
|---|---|---|---|
| O01 | vLLM version chính xác cho Gemma 4 trên A6000 (bvpt3 đã chạy vLLM 0.33.0 OK) | 12/11 | SWE B |
| O02 | Lịch thuê A6000 ngày 15/11 | 13/11 | Team |
| O03 | Ai review full 3.000 QA | 13/11 | Content |
| O04 | Ai build/test Docker | 14/11 | SWE B |
| O05 | Ngân sách thuê GPU | 13/11 | Team |
