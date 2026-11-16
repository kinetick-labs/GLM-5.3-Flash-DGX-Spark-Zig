#!/usr/bin/env bash
# Single- and two-stream decode bench against a running server (tools from the TensorFold checkout).
set -euo pipefail
TF_DIR="${TF_DIR:-$HOME/TensorFold-glm53}"; BASE="${BASE:-http://127.0.0.1:${PORT:-8094}}"; NAME="${NAME:-GLM-5.3-Flash}"
LEVELS="${LEVELS:-1,2}"
python3 "$TF_DIR/tools/bench_openai.py" "$BASE" "$NAME" --tokens 256 --reps 5 --temperatures 0
python3 "$TF_DIR/tools/bench_concurrent.py" "$BASE" "$NAME" --levels "$LEVELS" --tokens 256 --reps 3 \
  --temperatures 0 --alone
