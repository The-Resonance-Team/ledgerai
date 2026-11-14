# Submission Checklist — DataMate

## Trước freeze (15/11)

- [x] Chốt model: Gemma 4 12B (`google/gemma-4-12B-it`, 11.96B, Apache 2.0).
- [x] Chạy full holdout; lưu report PA, format error, latency (`docs/EVAL.md`).
- [x] Kiểm tra `submission.jsonl` đủ mọi câu, đúng format (`scripts/check_submission.py`).
- [x] Kiểm tra logs: `id`, `answer`, `elapsed_ms`, `thinking`, `code`, `inputs`, `output`.
- [x] Không đọc `answers.jsonl` của public/private trong `predict.py`.
- [x] Không hardcode ID câu hỏi/business.
- [x] Greedy decoding, temperature 0.
- [ ] Chạy 2 lần trên A6000, so sánh submission; PA lệch ≤0.03.
- [x] Freeze code, tag commit (repo local chưa phải git; nén source kèm MD5).
  Archive: `dist/ledgerai_source_20261113.tar.gz`; checksum trong
  `dist/ledgerai_source_20261113.tar.gz.sha256`; file list + hash trong
  `dist/FREEZE.sha256`.

## Kết quả hiện tại

| Holdout | Router PA | LLM + verifier PA | Wrong | Coverage |
|---|---:|---:|---:|---:|
| Main (553 câu) | **0.9476** | n/a (LLM blocked on GB10) | 1 | 95.1% |
| Schema (711 câu) | **0.9156** | n/a (LLM blocked on GB10) | 2 | 92.1% |

Public test: 699/1000 câu trả lời, 0 lỗi format, 1 đáp án bất thường
(lãi ròng âm — hợp lệ nếu chi phí lớn hơn lãi gộp). Public test chỉ dùng để
kiểm tra format/coverage, không dùng để chọn đáp án.

## Docker

- [x] Image name: `zaic2026_datamate_ledgerai:final`.
- [x] Base image `vllm/vllm-openai:v0.33.0` (CUDA 13.0, hỗ trợ Blackwell).
- [x] `ENTRYPOINT []`, `CMD ["/code/predict.sh"]`.
- [x] `predict.sh` đọc `/data`, ghi `/result/submission.jsonl` và `/result/logs/`.
- [x] `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- [x] Model weights nằm trong image tại `/code/models/gemma-4-12b-it`; không download lúc chạy.
- [x] `README.md` trong image: approach, model, license, data, run instructions.
- [x] `training_code/` có code + data + manifest + checksums.
- [x] Image **linux/amd64**, uncompressed **49.6GB** (`49,566,600,938` bytes; limit 100GB).
  Built on `bvpt3` (QEMU) with `docker/Dockerfile.rebuild`, layered on the
  validated `:test` image, no registry access.
- [x] Test offline deterministic: `--network none`, `--shm-size=16g`,
  `-v /data:ro`, `-v /result`; 8/8 answers, 8/8 logs, format valid.
- [ ] Test đầy đủ trên A6000 48GB (GPU + LLM + 2 lần chạy) và ghi rõ rủi ro Blackwell.

### Lệnh build/test trên A6000

```bash
# 1. Copy repo + model weights sang máy A6000 (model ~23GB).
rsync -av --exclude eval/predict_public* --exclude eval/run_* \
  --exclude dist ./ user@a6000:~/ledgerai/

# 2a. Clean build native linux/amd64.
cd ~/ledgerai
docker build --platform linux/amd64 -f docker/Dockerfile \
  -t zaic2026_datamate_ledgerai:final .

# 2b. Hoặc dùng image đã lưu: docker load -i zaic2026_datamate_ledgerai_final.tar.gz

# 3. Test offline + determinism (2 lần chạy).
RUNS=2 bash docker/a6000_test.sh
# Nếu image đã load sẵn:
SKIP_BUILD=1 RUNS=2 bash docker/a6000_test.sh
# Nếu host có image :test và registry lỗi:
DOCKERFILE=docker/Dockerfile.rebuild RUNS=2 bash docker/a6000_test.sh
```

Script `docker/a6000_test.sh` kiểm tra: kiến trúc image, size ≤100GB, chạy
`--network none`, đủ `submission.jsonl` + `logs/`, format, và tỉ lệ đáp án
giống nhau giữa 2 lần chạy.

### Ghi chú build

- Docker Hub DNS trên `bvpt3` lỗi (`registry-1.docker.io` lookup fail), nên
  bản `:final` hiện tại dùng `Dockerfile.rebuild` với base là image `:test`
  đã validate. Image vẫn là linux/amd64 và chứa đúng weights.
- `docker save | gzip -1` đã xong trên `bvpt3`:
  `~/ledgerai/dist/zaic2026_datamate_ledgerai_final.tar.gz`
  (30,008,998,850 bytes; checksum ở mục **Gửi bài**).

## Gửi bài

- [x] `docker save` + gzip; đo uncompressed size.
- [x] `md5sum` file `.tar.gz`.
  File: `~/ledgerai/dist/zaic2026_datamate_ledgerai_final.tar.gz`
  - compressed: **30,008,998,850 bytes**
  - MD5: `7bd0684500b5b2f57eb544fae8e98217`
  - SHA256: `3b1f094f38e4319a18893ca4569578b0bb7152e224408c8ffe0a2981b66cf634`
  - uncompressed image: **49,566,600,938 bytes**
  - `gzip -t`: OK
- [ ] Upload Google Drive; kiểm tra link từ máy khác.
- [ ] Gửi link + MD5 + uncompressed size cho organizer.
- [ ] Tối đa 2 submissions; bản cuối được chấm.
- [ ] Trước 16/11/2026 23:59 GMT+7.

## Sau khi nộp

- [ ] Không sửa image/code/model.
- [ ] Theo dõi private test 20–23/11.
- [ ] Chạy frozen solution trên private test, nộp kết quả trước 23/11 23:59.
- [ ] Lưu log, screenshot, checksum.


## Latest update (commit 32de8f4)
- Voltaren quantity fix (tuyp unit gate) resolved 1 schema wrong.
- Deterministic tie-break in _rank_products.
- LLM planner validated end-to-end on GB10 via production endpoint (4/5 eligible verified).
- GitHub repo: https://github.com/The-Resonance-Team/ledgerai (main + improve-v53).
- Remaining blockers: A6000 LLM run, Google Drive upload of Docker tar.
