# Team & Roles

## Thành viên

| Vai trò | Người | Phụ trách | Deliverable |
|---|---|---|---|
| **SWE A — Data** | TBD | Profiler, metric library, unit conversion, rulebook edge cases, SQLite/Excel | `src/data/`, `src/metrics/`, schema cards, unit tests |
| **SWE B — Agent/Infra** | TBD | vLLM, planner, executor, verifier, Docker, logs, determinism, submission | `src/agent/`, `docker/`, `predict.sh`, submission package |
| **Content manager** | TBD | Taxonomy, QA review, test cases, manual QA, README/approach, checklist | `qa/`, `docs/`, `README.md` inference |

## Nguyên tắc làm việc

- Mỗi người sở hữu một phần rõ ràng; không sửa file của người khác mà không báo.
- Mọi thay đổi lớn ghi vào `docs/DECISIONS.md`.
- Commit nhỏ, message rõ; freeze commit 22:00 mỗi ngày.
- Không đưa dữ liệu test/private lên chat, doc, hay repo public.
- Không hardcode ID câu hỏi/business.

## Nhịp ngày

| Giờ | Việc | Ai |
|---|---|---|
| 09:00 | Standup 15 phút: hôm qua, hôm nay, blocker | Cả team |
| 13:00 | Eval run + báo cáo PA/lỗi | SWE B + Content |
| 18:00 | Error review: câu sai, format lỗi, nguyên nhân | Cả team |
| 22:00 | Freeze commit + cập nhật decision log | SWE A/B |

## Kênh liên lạc

- Zalo group: trao đổi nhanh.
- Git repo: code, issue, PR.
- Google Doc: kế hoạch, quyết định, timeline.
- `docs/` trong repo: nguồn sự thật kỹ thuật.

## Escalation

- Blocker kỹ thuật > 30 phút: báo SWE B.
- Blocker dữ liệu/rulebook: báo SWE A + Content.
- Quyết định ảnh hưởng timeline: đưa vào standup, chốt trong ngày.
