# LedgerAI

**Ask your ledger anything.**

Trợ lý dữ liệu tiếng Việt cho hộ kinh doanh nhỏ. Dự án tham gia **Zalo AI Challenge 2026 — DataMate track**.

## Mục tiêu

- Trả lời câu hỏi tiếng Việt trên dữ liệu CSV/Excel/SQLite bẩn.
- Trả về đúng type, đúng đơn vị, đúng rulebook.
- Biết nói `unanswerable` hoặc `abstain` khi dữ liệu không đủ.
- Mục tiêu: top 1; mục tiêu thực tế: điểm private test cao, top 5.

## Trạng thái

- Track: DataMate.
- Model chính: **Gemma 4 12B** (11.96B, Apache 2.0, 140+ ngôn ngữ, native function calling).
- Fallback: **Qwen3.5-9B** (9.65B, Apache 2.0, 201 ngôn ngữ).
- Kiến trúc: **profiler → semantic layer → metric library → router + LLM planner → verifier → formatter**.
- Deterministic holdout PA: **0.8915** (coverage 89.2%, precision 100%,
  0 câu sai); schema holdout **0.8354**.
- Router + LLM + verifier full 553: **PA 0.9078** (coverage 90.8%,
  precision 100%); schema **0.8481**. LLM mặc định **bật**, tự fallback router.
- Public test deterministic: **699/1000** câu trả lời, 0 lỗi format, 1000 logs.
- Hạn nộp Docker: **16/11/2026 23:59 GMT+7**.
- Private test: **20–23/11/2026**.
- Docker image `zaic2026_datamate_ledgerai:final`: **linux/amd64, 49.6GB**,
  đã build và test offline deterministic; còn test GPU/LLM + 2 lần chạy trên A6000.

## Cấu trúc

```
ledgerai/
├── data/
│   ├── raw/train/           # Zalo train split (3.000 câu, 30 business)
│   ├── raw/public_test/     # Zalo public test (1.000 câu, 14 business)
│   └── processed/           # cache, schema cards, feature files
├── qa/                      # QA review pack
├── docs/                    # brief, decisions, timeline, eval, infra
├── prompts/                 # prompt gửi Gemini tạo Google Doc
├── resources/               # regulations, Docker guide, model cards
├── src/ledgerai/            # inference code
│   ├── loaders.py           # CSV/Excel/SQLite loaders + header detection
│   ├── normalize.py         # Vietnamese number/date normalization
│   ├── profile.py           # table/column role inference
│   ├── periods.py           # Vietnamese time expressions
│   ├── dataset.py           # semantic layer + metric library
│   ├── pipeline.py          # conservative deterministic router
│   ├── planner.py           # LLM planner + plan executor
│   ├── llm.py               # vLLM/Gemma wrapper
│   └── scoring.py           # answer comparison + PA
├── eval/                    # eval harness, business holdout, reports
├── scripts/                 # data prep, planner smoke test
├── models/                  # model weights (not committed)
└── docker/                  # Dockerfile, predict.sh, image docs
```

## Chạy nhanh

```bash
# Cài môi trường
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas openpyxl python-dateutil pydantic

# Eval baseline trên holdout theo business
.venv/bin/python eval/run_eval.py --data data/raw/train --out eval/run_holdout

# Chạy predict offline (chỉ router, không LLM)
.venv/bin/python predict.py --data data/raw/train --result eval/predict_smoke --limit 20

# Chạy predict mặc định (LLM + verifier, cần GPU + model)
.venv/bin/python predict.py --data data/raw/train --result eval/predict_llm --use-llm
```

## Docker

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_ledgerai:final .
docker run --rm --gpus all --network none --shm-size=16g \
  -v /path/to/test:/data:ro -v /path/to/result:/result \
  zaic2026_datamate_ledgerai:final
```

Chi tiết: `docker/README.md`.

## Đọc trước

1. `docs/PROJECT_BRIEF.md` — tóm tắt dự án.
2. `docs/DECISIONS.md` — quyết định đã chốt.
3. `docs/EVAL.md` — kết quả eval và holdout.
4. `docs/TIMELINE.md` — kế hoạch 4 ngày.
5. `qa/README.md` — quy trình review QA.
6. `resources/` — luật cuộc thi và Docker guideline.

## Lưu ý

- Không dùng public/private test để train hoặc lấy đáp án.
- Không đưa dữ liệu test/private lên tài liệu chia sẻ.
- Tài liệu nội bộ, không chia sẻ ra ngoài team.
