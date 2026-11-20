# GLM-5.3-Flash SAGE REXL3 2.48 bpw on one DGX Spark — TensorFold (kinetick-labs rebase/1.3.1-glm53)

Serves **GLM-5.3-Flash SAGE REXL3 2.48 bpw** ([vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw](https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw),
rev `7d0de246`, a quantized pack of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)) with
**DFlash2 drafts** ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), rev `6cc6c481`)
on **one NVIDIA GB10** with the native Zig engine (`tensorfold-native`) of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) branch
[**`rebase/1.3.1-glm53` @ `4f08e84ea`**](https://github.com/kinetick-labs/TensorFold/tree/rebase/1.3.1-glm53): upstream
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) **v1.3.1** (`5e1e1912e`) plus our GLM-5.3 REXL3 work
(FP8 prompt experts, live-half kept states, conversation reuse on the paged KV, boot retry). `scripts/build.sh` pins it.

The branch carries the 24 GLM units alone: it does **not** include our `main`'s 209 Flash-Next/compute-lane units
(the mia engine this rig never serves), so a GLM build stays independent of the `main` rebase. Before 2026-11-20 this
rig pinned `feat/glm53-rexl3` @ `2f01cde2e` on our `main` `082aee98a` = upstream v1.2.8 `d6c4cce60`.

Status: this is our **prod** model since 2026-11-16 — port `:8083`, container `glm53-prod`, user unit
`glm53-tensorfold.service`, copy of the serve script at `~/tf-prod-glm53/serve.sh`. The Qwen3.8 Flash-Next rig
([Qwen3.8-Flash-Next-DGX-Spark-Zig](https://github.com/kinetick-labs/Qwen3.8-Flash-Next-DGX-Spark-Zig)) is stopped and
switched back with `~/switch-to-qwen.sh`.

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

These figures were taken on the previous pin (`feat/glm53-rexl3` @ `72d1c64c2`/`c18970d85`, upstream v1.2.8): they are
**not re-measured on the v1.3.1 rebase** yet. Re-run `bench/` before quoting them against the new pin.

- **Prefill** (FP8 experts): **968 tok/s** at 2.9k tokens, **1,009 tok/s** at 14k, ~1,000–1,026 tok/s on 120k–247k fills.
- **Decode** (DFlash2, single stream, temperature 0): prose **29.9**, code **49.1**, math **54.3**, structured/JSON
  **63.2**, French **33.6** tok/s. Two short streams: ~27 tok/s each.
- **Conversation reuse**: turn 2 of a 120k conversation resumes 118,784 cached tokens in **1.2 s** (turn 1 116.7 s);
  four ~28k conversations all resume on turn 2 (0.5–1.4 s vs 27.8–28.9 s cold), outputs identical to cold.
- **Memory**: MemAvailable low 3,971 MiB (4 x 28k conversations), **3,528 MiB** (a 120k turn pair then two concurrent
  247k fills). With `KV_SHARE=2`: low 4,360 MiB.

### Concurrency (c1–c4)

The server runs `--parallel 2`, so a 3rd or 4th request waits for a free stream. Only measured numbers are shown;
"not measured" means no run exists for that cell.

| streams | prefill | decode (DFlash2, temperature 0) |
|---|---|---|
| c1 | 968 tok/s (2.9k), 1,009 (14k), 984 (64k), 1,026 (120k) | prose 29.9 (30.7 on a 25-token prompt), code 49.1, math 54.3, JSON 63.2, French 33.6 tok/s |
| c2 | 2 x ~130k: 971 / 484 tok/s a stream (the second prompt fills after the first, ~950 combined); 2 x ~247k: TTFT 246.8 / 495.2 s, **~1,000 tok/s combined** | 2 short prose streams: 27.0 / 27.6 tok/s; a stream decodes ~1.2 tok/s while the other fills a long prompt |
| c3 | not measured | not measured |
| c4 | not measured | not measured |

Builds: c1 prefill at 2.9k/14k/120k and the 2 x 247k run on `72d1c64c2` (FP8 experts, 2026-11-16, windows 29–30).
The 64k and 2 x 130k fills and the short-prompt decode ran on `c18970d85` (f16 experts, 2026-11-16, window 25). The
1.2 tok/s figure comes from windows 26–28.

### Agent jobs (Mercury, 2026-11-16)

In a test window, our Mercury bots ran against GLM in place of Qwen:

| job | GLM | Qwen (prod) |
|---|---|---|
| `veille-colombages` weekly run | 90.2 min, 2.14M tokens in / 45.8k out, 64 steps, 72 tool calls | 65 min (2026-11-13), 3.08M / 79k, 67 steps, 78 tool calls |
| `veille-dgx` daily run | 151.6 s, 24.4k / 1.4k, 4 steps, 3 tool calls | 125–127 s (2026-11-14 / 15) |

The 90 min run predates the fixes.

- **Cause of the slow run:** each step, Mercury's compaction rewrote an early tool result, so GLM resumed only at the
  4096-token system mark. On ~51k-token prompts, that cost ~50 s of TTFT a step.
- **Replay of that run** (30 steps + 12 vision requests):

  | config | mean TTFT a step | prompt tokens served from cache |
  |---|---|---|
  | before the fix | 26.5 s | 25 % |
  | `TF_KV_SHARE=4` alone | 26.2 s | 26 % |
  | with Mercury compaction hysteresis | **12.0 s** | **84 %** |

  The hysteresis moves the stub boundary only every 8 tool results; it is a Mercury fix, not a server change.
- **Status:** a full weekly run with the fix has not been done on GLM yet.
- **`veille-dgx`:** its 5–7k prompts were not reused in that run, because the test set `TF_KV_SHARE_MIN_PROMPT=8192`
  (the default is now 0). Reuse also resumes at 2048 marks only; see Known limits.

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

Builds `rebase/1.3.1-glm53` @ `4f08e84ea` ReleaseFast (Zig 0.17, CUDA nvcc, sm_121) and installs
`bin/tensorfold-native`, `libs/` and `COMMIT` under `$ROOT`. **Grammar library:** serve.sh expects `libtfgrammar.so`
(Linux aarch64) in `$ROOT/libs`: take it from a TensorFold release (pinned grammar runtime) or build it with
`zig build grammar -Dxgrammar=DIR`; `libturbojpeg` / `libwebpdecoder` there serve image input.


## 3. Runtime image

serve.sh runs `tensorfold-native` (step 2, mounted from `$ROOT`) inside a local runtime image. Build it once, no GPU
needed:

```bash
scripts/build-image.sh        # -> tensorfold-zig:runtime-pt2607 (image/Dockerfile); IMAGE=<tag> JOBS=<n> to override
```

The image (`image/Dockerfile`, multi-stage) is the same runtime as our Qwen recipe:

- **Base:** `nvcr.io/nvidia/pytorch:26.07-py3`, pinned by digest. It provides the CUDA 13 libraries the binary loads,
  and that is all GLM uses from the image. The 36.5 GB base is pulled once; after that the build takes under a minute
  on a Spark.
- **Qwen runtime parts:** the image also carries the Python package of the Qwen vision helper and the sm_121 Triton
  kernel set that Qwen's mia engine replays. GLM serves vision in-engine (`--vision`) and uses neither.
  - Python package: upstream ashhart/TensorFold `4af95036e` plus `image/python-overlay.patch`, with transformers
    5.17.0 and PyAV 19.0.1.
  - Kernel set: compiled without a GPU from the TensorFold `082aee98a` spec (our `main`'s mia-engine kernel spec, the
    one this image was built and verified against), every cubin checked against
    `image/sm121-kernels.txt`.

Checked on our Spark (2026-11-16) against the local image we served GLM with until now (`tensorfold-zig:exl3-572414d`):

- **Identical:** the kernel set (all 320 cubins and `aot.json`, byte for byte) and `pip freeze`. With a binary built from
  `72d1c64c2` (same engine code as the pin), `ldd` resolves the same libraries and none are missing.
- **Difference:** one comment line in the Python tree.

`IMAGE=` selects another tag.

## 4. Serve

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

Only what this recipe actually runs is listed; TensorFold's
[THIRD_PARTY_NOTICES.md](https://github.com/kinetick-labs/TensorFold/blob/feat/glm53-rexl3/THIRD_PARTY_NOTICES.md) has
the full attributions of the engine.

**Models**
- **Z.AI** ([zai-org](https://huggingface.co/zai-org)): the base model,
  [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
- **vcruz305** ([HF](https://huggingface.co/vcruz305), [GitHub](https://github.com/vcruz305)): the SAGE REXL3 2.48 bpw
  pack this recipe serves ([vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw](https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw)).
- **Inco AI** ([incoai](https://huggingface.co/incoai), [blog](https://inco.ai/blog/dflash2/)): the DFlash2 drafter
  ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)).
- **z-lab** (Jian Chen, Yesheng Liang, Zhijian Liu; [z-lab/dflash](https://github.com/z-lab/dflash), MIT): the DFlash
  block-diffusion drafter architecture TensorFold's native drafter implements.

**Engine and formats**
- **TensorFold**: [ashhart](https://github.com/ashhart) and [contributors](https://github.com/ashhart/TensorFold/graphs/contributors)
  ([ashhart/TensorFold](https://github.com/ashhart/TensorFold) v1.3.1 `5e1e1912e`, Apache-2.0): the native Zig/CUDA
  engine. The CUDA GLM family is mostly by ashhart, [Bizuayeu](https://github.com/Bizuayeu) and
  [BobClawblaw](https://github.com/BobClawblaw), with others (git history); the CUDA EXL3 linear and grouped expert
  kernels started with Justin Casey's work and were extended by ashhart, Anders Jensen-Waud and others.
- **turboderp** ([ExLlamaV3](https://github.com/turboderp-org/exllamav3), MIT): the EXL3 format (trellis layout,
  codebooks, bit widths) that REXL3 packs use; TensorFold's EXL3 decoders and kernels implement it, checked against
  ExLlamaV3.
- **MiaAI-Lab** ([GitHub](https://github.com/MiaAI-Lab)): upstream's GLM EXL3 routed-expert path (BobClawblaw,
  `9a9cda815`) was first written for MiaAI-Lab's 4-bit GLM-5.3-Flash EXL3 checkpoint; our REXL3 support extends that
  path to mixed widths. No MiaAI-Lab recipe is used as a base here, and their mia engine and layer-sliced prefill
  (in our `main`) are used by our Qwen recipe, not by GLM.
- **flash-linear-attention** ([fla-org](https://github.com/fla-org/flash-linear-attention), MIT): our KDA prompt
  chunks on tf32 tensor cores (`TF_GLM_KDA_CHUNK`) follow its `chunk_kda`.
- **mlx-vlm** (Prince Canuma, [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm), MIT): upstream's GLM image tower
  follows its `glm5_next` vision code.
- **Jay Leaton** ([jayleaton](https://github.com/jayleaton)): his compute lane and in-model runner
  ([jayleaton/qwen4-tensorfold-spark](https://github.com/jayleaton/qwen4-tensorfold-spark), Apache-2.0) are in this
  build through our `main` (off in this serve config), and his strict-tool work under `tool_choice: auto` is in the
  server.

**Runtime image** (shared with our Qwen recipe)
- **NVIDIA**: the [PyTorch 26.07 container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) (CUDA 13
  libraries; the only part of the image GLM uses). **Triton** ([triton-lang](https://github.com/triton-lang/triton),
  MIT) compiles the Qwen kernel set the image carries.
- `image/python-overlay.patch` (Qwen vision helper and FP8 KV kernels, not run by GLM) keeps MiaAI-Lab's attributions:
  the helper's request limits follow their single-Spark recipe's patches 0008 and 0009, and the FP8 KV rows follow
  their row format (patch 0038).
- Image input decodes in libjpeg-turbo (`libturbojpeg`) and libwebp (`libwebpdecoder`).

**This recipe**
- **Kinetick-labs** ([tournierjc](https://github.com/tournierjc)): GLM-5.3 REXL3 serving on one Spark (mixed-width
  REXL3 admission, FP8 prompt experts, KDA/MLA prompt kernels, live-half kept states, paged-KV conversation reuse,
  think-close and short-prompt knobs, boot retry), the image build, the benches and this recipe.
