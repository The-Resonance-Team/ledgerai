# Docker — LedgerAI

## Image name

`zaic2026_datamate_ledgerai:final`

## Final image status

- Architecture: **linux/amd64**.
- Uncompressed size: **49.6GB** (`49,566,600,938` bytes; limit 100GB).
- Built on `bvpt3` (ARM64 host, QEMU) with `docker/Dockerfile.rebuild`.
- The fallback build starts from the already validated `:test` image, so it
  reuses the vLLM 0.33.0/CUDA 13 base, the Gemma 4 12B weights and the train
  data. It needs **no registry access**. Docker Hub DNS was failing on the
  build host.
- Deterministic offline smoke test passed: 8/8 answers, 8/8 logs, submission
  format valid, answers match the local router.
- GPU + LLM + two-run determinism still must be validated natively on A6000.

## Prerequisites (build machine, x86_64 + NVIDIA)

1. Copy the repo and the model weights:

```bash
rsync -av --exclude .git --exclude .venv ./ a6000:~/ledgerai/
# Either copy the weights from the build cache:
rsync -av models/gemma-4-12b-it/ a6000:~/ledgerai/models/gemma-4-12b-it/
# or download once on the build machine (build-time network is allowed):
ssh a6000 'cd ~/ledgerai && mkdir -p models && hf download google/gemma-4-12B-it --local-dir models/gemma-4-12b-it'
```

2. Clean build (the base image is ~35GB, the model ~23GB, total ~58GB):

```bash
cd ~/ledgerai
docker build --platform linux/amd64 \
  -f docker/Dockerfile \
  -t zaic2026_datamate_ledgerai:final .
```

Local macOS builds are not supported: the image is linux/amd64 and needs ~60GB
free disk.

### Fallback build without registry access

If the validated `zaic2026_datamate_ledgerai:test` image is already present
(it contains the vLLM base and weights), layer the frozen code on top:

```bash
rsync -a --exclude models --exclude 'eval/run_*' --exclude 'eval/predict_*' \
  --exclude dist --exclude .venv ./ ~/ledgerai_rebuild/
cd ~/ledgerai_rebuild
docker build --platform linux/amd64 \
  -f docker/Dockerfile.rebuild \
  -t zaic2026_datamate_ledgerai:final .
```

The fallback image is slightly larger (new code layers on top of the old ones)
but stays far below 100GB. This is how the current `:final` image was built.

## Test (A6000 48GB proxy)

The image entrypoint is already `/code/predict.sh`; do not override it. The
default mode is the LLM planner + verifier (`LEDGERAI_USE_LLM=1`):

```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -v /path/to/public_test:/data:ro \
  -v /path/to/output:/result \
  zaic2026_datamate_ledgerai:final
```

Deterministic router only (fallback if the model cannot load):

```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -e LEDGERAI_USE_LLM=0 \
  -v /path/to/public_test:/data:ro \
  -v /path/to/output:/result \
  zaic2026_datamate_ledgerai:final
```

Run the full native check (build or load, offline run, format, two-run
determinism):

```bash
# Clean build on A6000:
RUNS=2 bash docker/a6000_test.sh

# Image already loaded from the saved tar:
SKIP_BUILD=1 RUNS=2 bash docker/a6000_test.sh

# Fallback build path on a host that has the :test image:
DOCKERFILE=docker/Dockerfile.rebuild RUNS=2 bash docker/a6000_test.sh
```

## Save + checksum

```bash
set -o pipefail
docker save zaic2026_datamate_ledgerai:final | gzip > zaic2026_datamate_ledgerai.tar.gz
gzip -dc zaic2026_datamate_ledgerai.tar.gz | wc -c
md5sum zaic2026_datamate_ledgerai.tar.gz > zaic2026_datamate_ledgerai.tar.gz.md5
```

## Lưu ý

- Image linux/amd64; uncompressed ≤100GB.
- Chạy offline: `--network none`.
- Test trên A6000 48GB (Ampere) là proxy; không phải Blackwell.
- `predict.sh` đọc `/data`, ghi `/result/submission.jsonl` và `/result/logs/`.
- `LEDGERAI_BUDGET_S` (mặc định `18000` = 5 giờ) là ngân sách LLM; hết
  ngân sách thì tự chuyển sang router deterministic để kịp giới hạn 6 giờ.
- Kết quả cuối: router 0.8915 (main) / 0.8354 (schema); LLM+verifier
  0.9078 / 0.8481.
- Không hardcode ID câu hỏi/business.
- `predict.py` ghi từng câu ngay khi có đáp án; nếu timeout, các câu đã ghi vẫn được chấm.
- Mỗi câu có log riêng trong `/result/logs/<id>.json` với `id`, `answer`,
  `elapsed_ms`, `thinking`, `code`, `output`.
