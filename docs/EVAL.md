# Evaluation — LedgerAI

## Setup

- Data: `data/raw/train` (3.000 questions, 30 businesses).
- Holdout: **business-level**, never question-level.
- Main holdout: `biz_105`, `biz_110`, `biz_115`, `biz_120`, `biz_125`, `biz_130` (553 questions).
- Schema holdout: `biz_102`, `biz_103`, `biz_108`, `biz_112`, `biz_117`, `biz_119` (711 questions; merged cells, conversion tables, hand-written ledgers).
- Metric: penalized accuracy (PA) = (correct − wrong) / N. Abstain = 0.
- Harness: `eval/run_eval.py`, `eval/run_eval_llm.py`.

## Final deterministic router

| Holdout | PA | Correct | Wrong | Abstain | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Main | **0.8915** | 493 | 0 | 60 | 89.2% | 100.0% |
| Schema | **0.8354** | 595 | 1 | 115 | 83.8% | 99.8% |

The main holdout has **zero wrong answers**. The schema run abstains on multi-item
rankings when a unit-conversion table exists, because mixed pack/base units make
the order unreliable; its only remaining wrong answer is one top-product ranking
where the gold label does not follow any quantity, revenue or order-count metric.

### Main holdout by answer type

| Type | N | Correct | Wrong | PA |
|---|---:|---:|---:|---:|
| date | 31 | 31 | 0 | **1.000** |
| boolean | 4 | 4 | 0 | **1.000** |
| number | 365 | 348 | 0 | **0.953** |
| string | 11 | 10 | 0 | **0.909** |
| list | 27 | 24 | 0 | **0.889** |
| percent | 102 | 76 | 0 | **0.745** |
| unanswerable | 13 | 0 | 0 | 0.000 |

## LLM planner + verifier

- Model: Gemma 4 12B, vLLM 0.33.0, temperature 0.
- LLM runs **only when the deterministic router abstains**; the verifier checks period, type and filters before execution.
- Full 553-question run: **PA 0.9078**, coverage 90.8%, precision 100%
  (router: 0.8915 / 89.2% / 100%). The LLM adds 9 correct answers; it introduces
  no wrong answers.
- Schema-gated run: **PA 0.8481**, coverage 85.1%, precision 99.8%
  (router: 0.8354 / 83.8% / 99.8%). The only wrong answer is the deterministic
  top-product ranking described above.
- The LLM is skipped for list/name questions ("liệt kê", "top N", "… nào"),
  where its ranking order is unreliable. This gate keeps 100% precision on the
  main holdout and removes the list errors on the schema holdout.
- Decision: **LLM enabled by default** (`LEDGERAI_USE_LLM=1`), with automatic
  fallback to the deterministic router if the model fails to load.

## Năng lực đã phủ

- Doanh thu theo ngày/tuần/tháng/quý/khoảng thời gian, theo mặt hàng, nhóm hàng.
- Kênh bán, phương thức thanh toán, ca sáng/chiều.
- Số đơn, đơn huỷ, tỉ lệ huỷ, doanh thu bình quân mỗi đơn.
- Ngày/tháng doanh thu cao nhất; top sản phẩm và khách hàng.
- Giá bán trong ngày, tồn kho cuối kỳ, giá trị tồn kho, mặt hàng ế.
- Nhập hàng, nhà cung cấp, giá nhập bình quân (abstain nếu không đủ chắc).
- Đếm đơn/ngày đạt ngưỡng, tăng trưởng cùng kỳ, xu hướng boolean.
- Thuốc kê đơn / không kê đơn, công nợ, chi phí, lãi gộp/lãi ròng, hoa hồng, bảo hành, thu máy cũ.
- Excel merged cells, bảng quy đổi đơn vị, footer "Tổng ngày/tháng", shorthand `1tr440`.

## Cách chạy lại

```bash
python eval/run_eval.py --out eval/run_holdout_main
LEDGERAI_HOLDOUT=biz_102,biz_103,biz_108,biz_112,biz_117,biz_119 \
  python eval/run_eval.py --out eval/run_holdout_schema
```

## Lịch sử tiến bộ

| Bản | Thay đổi chính | PA main |
|---|---|---:|
| baseline | router hẹp | 0.1917 |
| v7 | dedupe dòng trùng, sửa đơn vị | 0.4937 |
| v12 | tồn kho, giá bán, nhóm hàng | 0.5913 |
| v17 | ca sáng/chiều, kênh bán | 0.7342 |
| v22 | nhập hàng, NCC, growth | 0.7939 |
| v27 | merged cells, conversion tables | 0.8011 |
| v34 | kê đơn, khách hàng, đơn vị | 0.8101 |
| v38 | trạng thái, cùng kỳ, category share | 0.8246 |
| v46 | typo/status/unit fixes, customer & product keying, YoY | 0.8481 |
| v50 | mua bình quân, giao hàng, nhóm khách, hoạt chất | 0.8861 |
| **v52** | khách hàng `ten_kh`, đơn vị theo dòng, doanh thu sản phẩm, 30 ngày/YTD | **0.8915** |

Schema PA theo các bản gần nhất: v46 **0.7440** → v48 **0.7890** →
v50 **0.8129** → v52 **0.8354**. Bản LLM gated tương ứng: v50 **0.8228** →
v52 **0.8481**.
