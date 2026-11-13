# Infrastructure

## Dev GPU — `bvpt3`

- Host: `ai-inf-03`
- GPU: NVIDIA GB10 (Blackwell, unified memory)
- CPU: 20 cores
- RAM: 121 GB
- Disk: 3.7 TB
- OS: Linux aarch64
- Docker: 29.2.1
- Python: 3.12.3
- PyTorch: 2.13.0 + CUDA 13.0 (sau cài vLLM: torch 2.14.0, transformers 5.20.2)
- vLLM: **0.33.0** (cài 12/11, dùng `pip install --break-system-packages vllm`)
- Ninja: `~/.local/bin/ninja` — **phải thêm vào PATH** trước khi load model
- Model: `~/models/gemma-4-12b-it` (23GB, tải bằng `hf download`)

**Smoke test (12/11):**
- Load Gemma 4 12B: **199s** (sau khi cache compile)
- Prompt qua `apply_chat_template` → trả lời tiếng Việt đúng
- Lưu ý: prompt thô không qua chat template cho output rác (`-1111...`)
- `gpu_memory_utilization=0.4` do GB10 chia sẻ unified memory với process khác
- Process `VLLM::EngineCore` đôi khi còn sót; kill bằng `sudo kill -9 <pid>` (password: xem ngoài repo)

**Dùng để:** code, profiler, metric library, chạy model, eval nội bộ.

**Lưu ý:** ARM64. Không build/test image linux/amd64 ở đây. Không phải máy chấm.

## Test image — ThueGPU.vn

Bảng giá hiện có:

| GPU | VRAM | Giá | Ghi chú |
|---|---:|---:|---|
| RTX A6000 | 48GB | 40.000 VND/h | Ampere, gần nhất với 5090 |
| RTX 3090 | 24GB | 16.000 VND/h | Ampere, VRAM tight |
| RTX A4000 | 16GB | 10.000 VND/h | Không đủ |
| V100 | 32GB | 25.000 VND/h | Volta, cũ |
| P40 | 24GB | 8.000 VND/h | Pascal, cũ |

**Không có RTX 5090.** A6000 48GB là proxy gần nhất (Ampere, sm_86). Không test được Blackwell sm_120.

**Kế hoạch:** thuê A6000 48GB 4–6 giờ ngày 15/11. Test Docker với:

```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -v /path/to/public_test:/data:ro \
  -v /path/to/output:/result \
  zaic2026_datamate_ledgerai:final
```

**Base image:** `vllm/vllm-openai:v0.33.0` (CUDA 13.0, vLLM 0.33.0,
torch 2.14.0+cu130). Hỗ trợ Blackwell; image ~35GB + model ~23GB = ~58GB.

**Lưu ý:** build linux/amd64 trên macOS không khả thi (thiếu dung lượng).
Build trên máy A6000 x86_64; copy repo + model qua rsync.

**Rủi ro còn lại:** kernel/FlashAttention cho Blackwell, driver 5090, tensor parallel. Giảm thiểu: thiết kế chạy 1 GPU; dùng vLLM bản hỗ trợ Blackwell; test kỹ model load.

## Sync code

```bash
rsync -av --exclude data/raw --exclude .git ./ bvpt3:~/ledgerai/
```

## Model

- Gemma 4 12B: cần vLLM ≥0.31.0 (hỗ trợ Gemma 4).
- Fallback Qwen3.5-9B: cần vLLM main/nightly hoặc bản stable hỗ trợ Qwen3.5.
- Không dùng base image `vllm/vllm-openai:v0.10.1.1` cho Gemma 4; quá cũ.
- Kiểm tra CUDA 12.8+ và RTX 5090 support trước khi build.

## Data sync

Dữ liệu nằm trong `data/raw/` (train + public_test). Không commit lên repo public. Nếu cần chia sẻ nội bộ, dùng link riêng và giới hạn quyền.
