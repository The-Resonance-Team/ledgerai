#!/usr/bin/env bash
# Build and validate the final image on an x86_64 GPU host (A6000 proxy).
# Run from the repository root:  bash docker/a6000_test.sh
set -euo pipefail

IMAGE="${IMAGE:-zaic2026_datamate_ledgerai:final}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PUBLIC="${PUBLIC:-data/raw/public_test}"
OUT="${OUT:-/tmp/ledgerai_result}"
LOG="${LOG:-/tmp/ledgerai_a6000.log}"

echo "== host =="
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

test -f "$PUBLIC/questions.jsonl" || {
  echo "ERROR: $PUBLIC/questions.jsonl is missing" >&2
  exit 1
}

if [ "$SKIP_BUILD" = "1" ]; then
  echo "== build skipped (SKIP_BUILD=1) =="
  docker image inspect "$IMAGE" >/dev/null
else
  echo "== build =="
  docker build --platform linux/amd64 -f "$DOCKERFILE" -t "$IMAGE" . \
    2>&1 | tee "$LOG"
fi

echo "== image =="
ARCH=$(docker image inspect "$IMAGE" --format '{{.Architecture}}')
OS=$(docker image inspect "$IMAGE" --format '{{.Os}}')
SIZE_BYTES=$(docker image inspect "$IMAGE" --format '{{.Size}}')
SIZE_GB=$(awk -v bytes="$SIZE_BYTES" 'BEGIN { printf "%.1f", bytes / 1000000000 }')
echo "arch=$ARCH os=$OS size_gb=$SIZE_GB bytes=$SIZE_BYTES"
if [ "$ARCH" != "amd64" ] || [ "$OS" != "linux" ]; then
  echo "ERROR: image must be linux/amd64" >&2
  exit 1
fi
if [ "$SIZE_BYTES" -gt 100000000000 ]; then
  echo "ERROR: uncompressed image is larger than 100GB" >&2
  exit 1
fi

echo "== offline run =="
rm -rf "$OUT"
mkdir -p "$OUT"
START=$(date +%s)
docker run --rm --gpus all --network none --shm-size=16g \
  -v "$(cd "$PUBLIC" && pwd)":/data:ro \
  -v "$(cd "$OUT" && pwd)":/result \
  "$IMAGE" 2>&1 | tee -a "$LOG"
END=$(date +%s)
echo "elapsed_s=$((END - START))"

if [ "${RUNS:-2}" -ge 2 ]; then
  echo "== determinism run =="
  OUT2="${OUT}_run2"
  rm -rf "$OUT2"
  mkdir -p "$OUT2"
  docker run --rm --gpus all --network none --shm-size=16g \
    -v "$(cd "$PUBLIC" && pwd)":/data:ro \
    -v "$(cd "$OUT2" && pwd)":/result \
    "$IMAGE" 2>&1 | tee -a "$LOG"
  python3 - "$OUT" "$OUT2" <<'PY'
import json
import sys
from pathlib import Path


def load(root):
    path = Path(root) / "submission.jsonl"
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


a, b = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(a) & set(b))
same = sum(
    1
    for qid in shared
    if a[qid].get("type") == b[qid].get("type")
    and a[qid].get("answer") == b[qid].get("answer")
)
print(f"determinism: identical={same}/{len(shared)} ({100.0 * same / max(len(shared), 1):.2f}%)")
PY
fi

echo "== artifacts =="
test -s "$OUT/submission.jsonl"
test -d "$OUT/logs"
wc -l "$OUT/submission.jsonl"
ls "$OUT/logs" | wc -l

python3 scripts/check_submission.py \
  --submission "$OUT/submission.jsonl" \
  --questions "$PUBLIC/questions.jsonl"
python3 - "$OUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "submission.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
print("types", Counter(row["type"] for row in rows))
ids = {row["id"] for row in rows}
print("answers", len(rows), "unique_ids", len(ids))
missing = [qid for qid in ids if not (root / "logs" / f"{qid}.json").exists()]
print("missing_logs", len(missing))
PY

echo "== done =="
