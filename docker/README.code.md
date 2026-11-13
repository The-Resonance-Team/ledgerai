# LedgerAI — DataMate 2026

Offline Vietnamese data-analysis agent for the Zalo AI Challenge 2026 DataMate
track.

## Approach

1. **Profiler** (`src/ledgerai/profile.py`): loads every CSV/Excel/SQLite file,
   finds the real header row, infers column roles (order id, date, status,
   amount, product, customer, quantity, cost, debt) and table roles (orders,
   order lines, products, purchases, inventory).
2. **Semantic layer** (`src/ledgerai/dataset.py`): normalizes Vietnamese
   numbers (`1.234.567`, `25k`, `4,82 triệu`), dates (`dd/mm/yyyy`, Excel
   serials), and order statuses into a standard `orders` / `lines` model.
3. **Metric library** (`src/ledgerai/dataset.py`): revenue, order count,
   cancellation rate, top products by base-unit quantity, top customers,
   average order value, debt, inventory value, purchases.
4. **Router** (`src/ledgerai/pipeline.py`): answers only narrow,
   high-confidence families; everything else abstains (wrong = −1).
5. **LLM planner** (`src/ledgerai/planner.py`): for remaining questions, Gemma
   4 12B returns a JSON plan that the metric library executes deterministically.
6. **Scoring** (`src/ledgerai/scoring.py`): exact format-aware comparison and
   penalized accuracy.

## Model

- **google/gemma-4-12B-it**, 11.96B parameters, Apache 2.0, released
  2026-05-23. Public source: https://huggingface.co/google/gemma-4-12B-it.
- vLLM 0.33.0, deterministic sampling (`temperature=0`).
- No fine-tuning; no embedding or reranker models. Total inference parameters:
  11.96B (≤15B).

## Data

- Train split only (`data/raw/train/`). No public/private test data is used for
  training or answer collection.
- `data_manifest.json` lists local paths, sizes and SHA-256 checksums.

## Run

```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -v /path/to/test:/data:ro \
  -v /path/to/result:/result \
  zaic2026_datamate_ledgerai:final
```

`predict.sh` reads `/data/questions.jsonl`, writes `/result/submission.jsonl`
and one log record per question under `/result/logs/`.

Environment:
- `LEDGERAI_USE_LLM=1` (default): Gemma 4 planner + verifier for questions
  the router cannot answer. Full holdout PA 0.9078, coverage 90.8%,
  precision 100%; schema 0.8481, coverage 85.1%, precision 99.8%.
  List/name questions stay deterministic.
- `LEDGERAI_USE_LLM=0`: deterministic router only (main PA 0.8915,
  schema 0.8354, precision 100% / 99.8%). Use it as a fallback or for a
  fast smoke test.

## Reproduce the holdout

```bash
python eval/run_eval.py --data data/raw/train --out eval/run_holdout
```

The holdout is split by business: `biz_105`, `biz_110`, `biz_115`, `biz_120`,
`biz_125`, `biz_130`.
