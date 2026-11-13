# Model Choice

## Quyết định

**Model chính: Gemma 4 12B** (`google/gemma-4-12B-it`)

- Parameters: **11.96B** → ≤15B cap.
- License: **Apache 2.0**.
- Context: **256K**.
- Ngôn ngữ: 140+; 35+ ngôn ngữ out-of-the-box.
- Agentic: native function calling, coding, reasoning.
- Release: **23/05/2026** → trước cutoff 26/10/2026.
- vLLM: cần bản hỗ trợ Gemma 4 (v0.31.0 hoặc mới hơn).

**Fallback: Qwen3.5-9B** (`Qwen/Qwen3.5-9B`)

- 9.65B; Apache 2.0; 201 ngôn ngữ; agentic/coding; 262K.
- Release: 27/02/2026.
- Dùng nếu Gemma 4 yếu tiếng Việt, lỗi vLLM, hoặc VRAM không đủ.

## Vì sao không dùng các model khác

| Model | Lý do không chọn |
|---|---|
| Qwen3.8-27B | 28B > 15B cap. Quantization không đổi số parameter. |
| Qwen3.8-2.4T-A95B | Quá lớn. |
| Qwen3-14B | Cũ (2025); organizer đã test nhưng không còn là model mới nhất. |
| Nemotron 3.5 Bolt E4B | Release 05/10/2026 (H2), tool use mạnh, nhưng language list không có tiếng Việt; 4B effective. |
| Granite 4.2 8B | Release 25/08/2026, Apache 2.0, nhưng language list không có tiếng Việt. |
| Ling-3.0-tiny | Release 10/08/2026, MIT, agentic, nhưng không list tiếng Việt; phải build vLLM từ source. |
| AesCode-8B | Code/visual, không phải data analyst. |
| Fara1.5-4B | GUI agent, không phù hợp. |

## Cap và VRAM

- Gemma 4 12B bf16 ~24GB → fit 1×RTX 5090 32GB nếu context vừa phải.
- A6000 48GB (Ampere) là proxy trên ThueGPU; không phải Blackwell.
- Nếu dùng embedding nhỏ (<1B), tổng vẫn ≤15B.
- Nếu VRAM tight: giảm `max-model-len`, dùng bf16, không bật vision/audio.

## Validation

- Smoke test 20 câu train trên `bvpt3`.
- Đo: parse JSON, exec code, exact match, latency.
- Nếu Gemma 4 < Qwen3.5-9B quá xa: đổi fallback.
- Không fine-tune trong 48h đầu.

## Lưu ý license

- Gemma 4: Apache 2.0 (theo model card).
- Qwen3.5: Apache 2.0.
- Khai báo model, revision, license, parameter count trong `README.md` của Docker image.
