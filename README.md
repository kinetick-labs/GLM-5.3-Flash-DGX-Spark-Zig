# GLM-5.3-Flash SAGE REXL3 2.48 bpw on one DGX Spark — TensorFold (kinetick-labs feat/glm53-rexl3)

Serves **GLM-5.3-Flash SAGE REXL3 2.48 bpw** ([vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw](https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw),
rev `7d0de246`, a quantized pack of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)) with
**DFlash2 drafts** ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), rev `6cc6c481`)
on **one NVIDIA GB10** with the native Zig engine (`tensorfold-native`) of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) branch
[**`feat/glm53-rexl3` @ `2f01cde2e`**](https://github.com/kinetick-labs/TensorFold/tree/feat/glm53-rexl3): our `main`
`082aee98a` (upstream [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `d6c4cce60` / v1.2.8 + our engine
work + Jay Leaton's compute lane) plus our GLM-5.3 REXL3 work (FP8 prompt experts, live-half kept states, conversation
reuse on the paged KV, boot retry). `scripts/build.sh` pins it.

Status: tested on our Spark on port 8094 (container `glm53`); it is **not** our prod (prod is Qwen3.8 Flash-Next,
[Qwen3.8-Flash-Next-DGX-Spark-Zig](https://github.com/kinetick-labs/Qwen3.8-Flash-Next-DGX-Spark-Zig)). The branch is
not merged into our `main`.

## Configuration

| | |
|---|---|
| Weights | `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` rev `7d0de24624020fc4f07923572be2cef591a4d1ed` (~100 GB) |
| Drafts | `incoai/GLM-5.3-Flash-DFlash2` rev `6cc6c481eee930a5139619ed919f78726371adbc` (~2.2 GB), `--drafter` |
| Streams / context | `--parallel 2` x `--context 262144`, `--max-tokens 8192` |
| KV cache | paged (`TF_KV_PAGES=on`), `TENSORFOLD_MEMORY_RESERVE_GIB=5`; the two streams share one page pool (554 pages of 512 rows, 3.45 GiB), enough for one full 262k window |
| Prompt prefill | EXL3 prompt experts on FP8 (build default; `TF_GLM_X3_FP8=0` gives f16 back), 2048-row passes |
| Conversation reuse | `TF_KV_SHARE=4` kept states, `TF_GLM_KV_LIVE_HALF=1` (live-half KDA snapshots, 302 MiB each instead of 438 MiB, warm outputs bit-identical to cold), `TF_GLM_HOST_CACHE=1` (snapshots in host RAM), `--prompt-cache-gib 1` |
| Scheduling | `TENSORFOLD_PREFILL_POLICY=fcfs`: a short prompt goes first between 2048-row passes of a long one |
| Think tags | `TF_THINK_CLOSE_ONCE=1` (build default): after a think block closed, a close tag the answer spells again is dropped |
| Vision | `--vision` (in-engine tower, ~1.3 GiB); JPEG / WebP through `libturbojpeg` / `libwebpdecoder`. Video is not served |
| Boot | a start refused with "budget fits no stream" (KV budget ~0.1 GiB short right after another server stops) is retried 3 times, 20 s apart |
| Memory | MemAvailable floor accepted **~3.4 GiB** (lowest measured 3528 MiB); keep a host watchdog (we stop the container under 1 GiB; it never tripped) |

Optional knobs on this branch: `TF_KV_SHARE_MIN_PROMPT=N` (prompts shorter than N keep no state, so short requests do
not evict a long conversation's kept states; default 0), `KV_SHARE=2` in serve.sh (~0.4 GiB more margin).

## Measured (Spark GB10, 16 Nov 2026, this config; details in `bench/results.md`)

- **Prefill** (FP8 experts): **968 tok/s** at 2.9k tokens, **1,009 tok/s** at 14k, ~1,000–1,026 tok/s on 120k–247k fills.
- **Decode** (DFlash2, single stream, temperature 0): prose **29.9**, code **49.1**, math **54.3**, structured/JSON
  **63.2**, French **33.6** tok/s. Two short streams: ~27 tok/s each.
- **Conversation reuse**: turn 2 of a 120k conversation resumes 118,784 cached tokens in **1.2 s** (turn 1 116.7 s);
  four ~28k conversations all resume on turn 2 (0.5–1.4 s vs 27.8–28.9 s cold), outputs identical to cold.
- **Memory**: MemAvailable low 3,971 MiB (4 x 28k conversations), **3,528 MiB** (a 120k turn pair then two concurrent
  247k fills). With `KV_SHARE=2`: low 4,360 MiB.

## 1. Checkpoint and drafter

```bash
hf download vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw --revision 7d0de24624020fc4f07923572be2cef591a4d1ed \
  --local-dir ~/models/GLM-5.3-Flash-SAGE-REXL3-2.48bpw
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 6cc6c481eee930a5139619ed919f78726371adbc \
  --local-dir ~/models/glm53-dflash2
```

## 2. Build and install

```bash
ROOT=$HOME/tf-glm53-serve LIBS_FROM=/path/to/libs scripts/build.sh
```

Builds `feat/glm53-rexl3` @ `2f01cde2e` ReleaseFast (Zig 0.17, CUDA nvcc, sm_121) and installs
`bin/tensorfold-native`, `libs/` and `COMMIT` under `$ROOT`. **Grammar library:** serve.sh expects `libtfgrammar.so`
(Linux aarch64) in `$ROOT/libs`: take it from a TensorFold release (pinned grammar runtime) or build it with
`zig build grammar -Dxgrammar=DIR`; `libturbojpeg` / `libwebpdecoder` there serve image input.

Runtime image: any aarch64 image with the CUDA 13 runtime libraries; we use our local `tensorfold-zig:exl3-572414d`
(the same runtime as our Qwen recipe). Set `IMAGE=` to yours.

## 3. Serve

```bash
scripts/serve.sh              # docker container glm53 on :8094 (OpenAI-compatible, model name GLM-5.3-Flash)
DRY_RUN=1 scripts/serve.sh    # print the command only
```

Checks after start: `/health` 200, a text and an image chat. `scripts/bench.sh` runs a quick decode bench.

GLM and another large model do not fit together on one Spark: stop the other server first (serve.sh retries the
boot while the page cache drains).

## Known limits

- One 262k window fits in the page pool: two concurrent ~247k conversations are served (the second prefill waits
  for pages) but neither is reused on turn 2; a 277k mix is refused (`context_length_exceeded`).
- Reuse resumes only at a 2048-row cold cut (that is what keeps warm == cold bit-exact): short agent prompts
  (4.5–6k) reuse at most the last 2048 mark. `TF_GLM_REQUEST_STARTS=1` is refused with `TF_KV_SHARE`.
- While a long prefill runs, the other stream decodes at ~1.2 tok/s until it finishes (short prompts still start
  within ~1 s with fcfs).
- FP8 prompt experts changed no answer on our 64-question bench but add 9–10 % perplexity on long prose and one known
  arithmetic drift in our coherence set; `TF_GLM_X3_FP8=0` restores f16 (~960 / ~980 tok/s).
- Memory margin is thin: ~3.5 GiB MemAvailable at worst with 4 kept states, and the boot margin is ~0.1 GiB.
- `TF_GLM_CHUNK_SOLO=4096` does not boot at 2 x 262,144 with vision and drafts; the adaptive draft depth
  (`TF_GLM_DRAFT_ADAPT`) and the KDA pair launch gave no measurable gain.
- Prefill is ~1,000 tok/s, below our 1,300 tok/s target.

## Licences

- This repository (scripts, docs): MIT.
- TensorFold: Apache-2.0 (ashhart and contributors; our fork keeps it).
- Weights are under their own licences, read the model cards before use: the REXL3 pack declares **MIT** (Z.AI's
  GLM-5.3-Flash licence, cite the base model). The DFlash2 card metadata declares **Apache-2.0**, but we have seen it
  referred to as CC BY-NC-ND elsewhere: treat the drafter's licence as unclear until incoai's card states it
  unambiguously, and check it before any commercial use.

## Credits

- **TensorFold**: ashhart and contributors (upstream ashhart/TensorFold, Apache-2.0): the native engine, the GLM
  family and EXL3 kernels this branch builds on.
- **vcruz305**: the SAGE REXL3 2.48 bpw pack. **Z.AI** (zai-org): GLM-5.3-Flash. **incoai** / z-lab: DFlash2 drafts.
- **Jay Leaton** (jayleaton): the compute lane and in-model runner in our `main` (jayleaton/qwen4-tensorfold-spark,
  Apache-2.0) and the upstream strict-tool / grammar work our grammar hooks follow.
- **MiaAI-Lab**: the mia engine and layer-sliced prefill in our `main` (used by our Qwen recipe, not by GLM). No
  MiaAI-Lab recipe is used as a base here.
- **Kinetick-labs** (tournierjc): GLM-5.3 REXL3 serving on one Spark (FP8 prompt experts, live-half kept states,
  paged-KV conversation reuse, think-close and short-prompt knobs, boot retry), the benches and this recipe.
