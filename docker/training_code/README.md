# Training code and reproduction

LedgerAI does not fine-tune the base model. The solution uses:

- **Base model:** `google/gemma-4-12B-it` (11.96B parameters, Apache 2.0),
  released 23/05/2026. Public source:
  `https://huggingface.co/google/gemma-4-12B-it`.
- **Inference:** vLLM 0.33.0 with deterministic sampling (`temperature=0`).
- **Pipeline:** deterministic profiler + metric library + conservative router;
  the LLM planner is used only for questions the router cannot answer.

## Data

- Training data: the organizers' train split (`data/raw/train/`), used for
  evaluation and prompt development only.
- QA review pack: `qa/train_review.csv` (3,000 train questions with gold
  answers), derived from the train split.
- No public or private test data is used for training or answer collection.

## Reproduce

```bash
python eval/run_eval.py --data data/raw/train --out eval/run_holdout
```

The holdout is split by business (`biz_105`, `biz_110`, `biz_115`,
`biz_120`, `biz_125`, `biz_130`), never by question.

## Manifest

`/code/data_manifest.json` lists local paths, sizes and SHA-256 checksums.
