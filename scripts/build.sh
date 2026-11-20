#!/usr/bin/env bash
# Build tensorfold-native (ReleaseFast, CUDA fatbins embedded) from kinetick-labs/TensorFold feat/glm53-rexl3 on a
# DGX Spark, and install it with its runtime libraries under a stable root ($ROOT, default ~/tf-glm53-serve).
set -euo pipefail
TF_DIR="${TF_DIR:-$HOME/TensorFold-glm53}"
TF_BRANCH="${TF_BRANCH:-rebase/1.3.1-glm53}"
TF_REF="${TF_REF:-4f08e84ea}"   # kinetick-labs/TensorFold rebase/1.3.1-glm53 (upstream ashhart/TensorFold v1.3.1 5e1e1912e + our 24 GLM-5.3 REXL3 commits)
ROOT="${ROOT:-$HOME/tf-glm53-serve}"
LIBS_FROM="${LIBS_FROM:-}"      # a directory with libtfgrammar.so, libturbojpeg.so*, libwebpdecoder.so* (see README)
ZIG="${ZIG:-$(command -v zig || echo "$HOME/zig-0.17.0/zig")}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
[[ -d "$TF_DIR/.git" ]] || git clone https://github.com/kinetick-labs/TensorFold "$TF_DIR"
cd "$TF_DIR"
git fetch origin "$TF_BRANCH"
git checkout --detach "$TF_REF"
echo "[build] $(git log -1 --format='%h %s')"
"$ZIG" version | grep -q '^0\.17\.' || { echo "[build] Zig 0.17.x required (.zig-version)" >&2; exit 2; }
"$ZIG" build native -Dnvcc="$NVCC" -Dsm=121 -Doptimize=ReleaseFast
mkdir -p "$ROOT/bin" "$ROOT/libs"
install -m755 zig-out/native/bin/tensorfold-native "$ROOT/bin/tensorfold-native"
echo "$(git rev-parse --short=9 HEAD) ReleaseFast $(date -u +%FT%TZ)" > "$ROOT/COMMIT"
if [[ -n "$LIBS_FROM" ]]; then cp -a "$LIBS_FROM"/libtfgrammar.so "$LIBS_FROM"/libturbojpeg.so* "$LIBS_FROM"/libwebpdecoder.so* "$ROOT/libs/"; fi
[[ -e "$ROOT/libs/libtfgrammar.so" ]] || echo "[build] warning: $ROOT/libs/libtfgrammar.so missing: serve.sh refuses to start without it (README, Grammar library)" >&2
echo "[build] -> $ROOT/bin/tensorfold-native ($(cat "$ROOT/COMMIT"))"
