#!/usr/bin/env bash
set -euo pipefail

mkdir -p /result/logs

ARGS=(--data /data --result /result --model /code/models/gemma-4-12b-it)
if [ "${LEDGERAI_USE_LLM:-1}" = "1" ]; then
  ARGS+=(--use-llm)
fi

exec python3 /code/predict.py "${ARGS[@]}"
