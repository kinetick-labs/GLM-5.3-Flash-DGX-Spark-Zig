# Bench — GLM-5.3-Flash SAGE REXL3 2.48 bpw, kinetick-labs/TensorFold feat/glm53-rexl3 (Spark GB10, 16 Nov 2026)

Measured with our window harness on the Spark (temperature 0, DFlash2 on). Builds: `c18970d85` (window 25),
`45199be92` (windows 26–28), `72d1c64c2` (windows 29–30, FP8 default on; same engine code as the pin `2f01cde2e`,
which only adds serve-script defaults and docs). Needles were found in every long run.

## FP8 prompt experts and decode (windows 29–30)

| Workload | result |
|---|---|
| prefill 2.9k tokens | 968 / 968 / 966 tok/s (f16: ~960) |
| prefill 14k tokens | 1,009 / 1,005 tok/s (f16: ~980) |
| prefill 120k–247k fills | ~1,000–1,026 tok/s |
| decode prose / code / math / structured / French | 29.9 / 49.1 / 54.3 / 63.2 / 33.6 tok/s |

FP8 changed no answer on a 64-question bench; +9–10 % perplexity on long prose, one known arithmetic drift.

## Conversation reuse, `TF_KV_SHARE=4` + live-half snapshots (windows 29–30)

| Workload | prompt tokens | cached | TTFT |
|---|---|---|---|
| turn 1, 120k conversation | 119,728 | 0 | 116.7 s |
| turn 2, same conversation | 119,755 | 118,784 | 1.20 s |
| 4 conversations of ~28k, turn 1 | ~28k each | 0 | 27.8–28.9 s |
| same 4 conversations, turn 2 | ~28k each | 26,624–28,672 | 0.5–1.4 s |
| 2 concurrent 247k fills, turn 1 | 247k x 2 | 0 | 246.8 / 495.2 s |
| same, turn 2 | 247k x 2 | 0 | (not reused: the pool fits one 262k window) |

Warm outputs are identical to cold. Snapshot size 302 MiB (438 MiB without live half).
MemAvailable: boot 3,759 MiB, low 3,971 MiB (4 x 28k) and 3,528 MiB (120k pair then 2 x 247k); 1 GiB watchdog never tripped.

## Reuse and short-first prefill, `TF_KV_SHARE=2` (windows 26–28)

| Workload | prompt tokens | cached | TTFT |
|---|---|---|---|
| turn 2, 120k conversation | 119,755 | 118,784 | 1.23 s |
| turn 2, 247k conversation | 247,085 | 245,760 | 1.65 s |
| turn 2 after a 197k fill evicted it | 247,085 | 0 | 260.6 s |
| 2 conversations of ~97k, alternating, turn 2 | 97,303 / 97,301 | 96,256 each | 1.29 / 1.29 s |
| short prompt during a 247k prefill (fcfs) | 25 | 0 | 1.01 s |
| short prompt during a 247k prefill (default policy) | 25 | 0 | 1.55 s |

MemAvailable low 4,360 MiB (window 28).

## Base serving (window 25, `c18970d85`)

| Workload | prompt tokens | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1 short prose | 25 | 0.2 s | — | 30.7 |
| 2 short prose | 25 + 25 | 0.2 / 0.5 s | — | 27.0 / 27.6 |
| one 64k fill | 70,320 | 71.5 s | 984 | 31.5 |
| 2 x ~130k fills | 129,541 + 134,441 | 133.5 / 277.9 s | 971 / 484 | 30.8 / 28.9 |
| 247k + 62k fills | 247,071 + 62,521 | 260.8 / 331.5 s | 947 / 189 | 27.8 / 25.5 |

## Concurrency summary (c1–c4)

`--parallel 2`: c3 and c4 were not measured (extra requests wait for a free stream).

| streams | prefill | decode |
|---|---|---|
| c1 | 968 (2.9k), 1,009 (14k), 984 (64k, f16 experts), 1,026 (120k) tok/s | prose 29.9 / code 49.1 / math 54.3 / JSON 63.2 / French 33.6; 30.7 on a 25-token prompt |
| c2 | 2 x ~130k: 971 / 484 a stream (~950 combined, f16); 2 x ~247k: 246.8 / 495.2 s TTFT, ~1,000 combined | 27.0 / 27.6 (two short prompts); ~1.2 while the other stream fills a long prompt |
| c3 | not measured | not measured |
| c4 | not measured | not measured |

## Mercury agent jobs on GLM (2026-11-16)

| run | wall | tokens in / out | steps / tool calls | notes |
|---|---|---|---|---|
| `veille-colombages` weekly, GLM | 90.2 min | 2.14M / 45.8k | 64 / 72 (6 tool errors) | before the fixes; cached only 0–4096 on ~51k prompts (~50 s TTFT a step) |
| same job on Qwen prod, 2026-11-13 | 65 min | 3.08M / 79k | 67 / 78 | |
| `veille-dgx`, GLM | 151.6 s | 24.4k / 1.4k | 4 / 3 | prompts 5.2–6.8k, cached 0 (`TF_KV_SHARE_MIN_PROMPT=8192` in that test), TTFT 5.7–7.5 s |
| `veille-dgx` on Qwen prod, 2026-11-14 / 15 | 125 / 127 s | 52.7k / 1.8k; 37k / 1.9k | 7; 5 / 4 | |

Replay of the weekly run (30 bot steps + 12 vision requests, same prompts):

| arm | TTFT sum | mean a step | cached / prompt tokens | vision |
|---|---|---|---|---|
| base | 794.7 s | 26.5 s | 196,608 / 798,411 | 12 local, max 273 s, 0 errors |
| `TF_KV_SHARE=4` | 787.0 s | 26.2 s | 210,944 / 798,411 | 12 local, max 297 s, 0 errors |
| + Mercury compaction hysteresis (stub boundary moves every 8 results) | 361.2 s | **12.0 s** | 882,688 / 1,046,901 (84 %) | 12 local, max 255 s, 0 errors |

## After the rebase onto upstream v1.3.1, at the new 2100 MHz clock cap (2026-11-20)

The Spark's SM clock is capped at 2100 MHz now (it reports 3003 MHz as `clocks.max.sm`, 2080–2093 MHz
observed under load), so these numbers are **not** comparable to the 16 Nov ones above: prefill loses
about 9 % from the clock alone. To separate the rebase from the clock, both pins were built and served
with the same `serve.sh`, weights and GPU clock, and benched back to back with one harness
(cold prompt: a unique first token defeats the prompt cache; decode: 256-token cap, temperature 0,
3 reps, span from first to last streamed token).

| measurement | `feat/glm53-rexl3` `2f01cde2e` (pre-rebase) | `rebase/1.3.1-glm53` `4f08e84ea` |
|---|---|---|
| prefill, 2.5k prompt | 876.5 tok/s | 874.5 tok/s |
| prefill, 12k prompt | 948.8 tok/s | 945.3 tok/s |
| decode, prose | 29.1 tok/s | 29.1 tok/s |
| decode, code | 34.5 tok/s | 34.4 tok/s |
| decode, math | 49.7 tok/s | 49.7 tok/s |
| decode, JSON | 38.4 tok/s | 38.3 tok/s |
| decode, French prose | 30.3 tok/s | 30.3 tok/s |
| cold TTFT, 12k | 12.63 s | 12.63 s |
| cached TTFT, 12k | 1.95 s (10,240 cached) | 1.95 s (10,240 cached) |

**The rebase costs nothing measurable**: every figure is within ±0.3 %, i.e. inside run-to-run noise.
Both pins report 1.3.1 as the served version, 638 KV pages, DFlash2 with 4 state copies, and load in
~80 s.

Two caveats. The decode figures above come from this repository's prompts, not from the window harness
behind the 16 Nov table, so `code` and `JSON` are lower than 16 Nov simply because the generated text
differs (draft acceptance follows the content) — the A/B line proves it is not a regression. And the
prompt cache matches **blocks**, not just prefixes: with `TF_KV_SHARE=4` the rebased build reported
10,240 cached tokens on the first call of a prompt it had seen with a different first line, where the
pre-rebase build reported 0 and cached them only on the second call.

## Knob sweep on the rebased build, 2100 MHz cap (2026-11-20)

Every arm below is the same binary (`4f08e84ea`), weights, `serve.sh` and GPU clock (2080–2093 MHz),
one restart apart; `TENSORFOLD_PREFILL_POLICY` stayed `fcfs` in every arm. 2 reps a point.

**Draft and expert form — the defaults already win.** Nothing here beats the shipped configuration:

| arm | prefill 2.5k | prefill 12k | prose | code | math | JSON | French |
|---|---|---|---|---|---|---|---|
| baseline (no extra knobs) | **877.5** | **950.8** | **29.2** | **34.5** | **49.8** | **38.4** | **30.3** |
| `TF_GLM_DRAFT_ADAPT=1` | 874.1 | 946.9 | 28.6 | 34.2 | 49.0 | 37.6 | 30.0 |
| + `DRAFT_CONFIDENCE=0.6` | 873.8 | 947.8 | 29.0 | 34.0 | 47.5 | 37.5 | 29.9 |
| + `DRAFT_CONFIDENCE=0.8` | 876.5 | 948.5 | 28.4 | 33.9 | 48.8 | 37.4 | 29.8 |
| + `DRAFT_CONFIDENCE=0.9` | 872.5 | 947.6 | 27.7 | 33.2 | 47.9 | 36.9 | 29.5 |
| `DRAFT_ADAPT=1` + `EXPERT_FORM=factored` | 874.0 | 949.7 | 28.6 | 34.3 | 49.1 | 37.6 | 30.0 |

Adaptive draft depth costs about 2 % here, and every confidence cut below or above the 0.7 default costs
more: on this drafter and these prompts the fixed depth at 0.7 is already the best point. Naming the
expert form changes nothing measurable, which says the first-load race lands on the same form either way.

**Prompt-cache knobs — neutral.** A five-turn growing conversation (a 10.8k-token document, its TTFT
summed) run twice an arm:

| arm | TTFT sum, conversation 1 / 2 | cached on the last turn |
|---|---|---|
| baseline (`TF_KV_SHARE=4`, 1 GiB cache) | 25.8 / 26.2 s | 10,240 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS=1` | 26.0 / 26.7 s | 10,240 |
| `TF_KV_SHARE_MIN_PROMPT=512` | 26.5 / 25.6 s | 10,240 |
| both | 25.7 / 28.6 s | 10,240 |

The 10,240-token figure is the same in every arm and in every conversation: it is what one kept state
covers here, not a knob's doing, so a longer prefix is re-prefilled past it whatever these settings say.

**`--parallel 3` is the one real win, and it costs conversation reuse.** `tools/bench_concurrent.py`,
256 tokens, temperature 0, 3 reps:

| arm | code c2 | code c3 | chat c2 | chat c3 | worst TTFT at c3 |
|---|---|---|---|---|---|
| `--parallel 2` (shipped) | 66.8 | 52.6 | 54.1 | 42.5 | 7.98 s |
| `--parallel 3`, `TF_KV_SHARE=4` | 67.0 | **86.7** | 54.3 | **70.9** | 0.50 s |
| `--parallel 3`, `TF_KV_SHARE=6`, 2 GiB cache | — | 86.6 | — | 71.0 | 0.64 s |
| `--parallel 3`, `TF_KV_SHARE=12`, 4 GiB cache | 66.7 | 86.4 | 54.3 | 70.7 | 0.50 s |

Three streams instead of two: aggregate **+30 % against the best two-stream figure and +65 % against what
three queued streams get today**, per stream −14 %, and the third request stops waiting 8 s. The cost is on
the conversation side: reuse worsens from 26.0 s to 27.3 s with `TF_KV_SHARE=6`, and `TF_KV_SHARE=12` with a
4 GiB cache — the pair the engine's own warning asks for — is far worse (26.7 s then **62.2 s**: twelve
states in four GiB leave the second conversation nothing, so every one of its five turns re-prefills).
The shipped `--parallel 2` is what prod runs; the three-stream trade is the owner's call, not a default.

Reproduce an arm without editing `serve.sh`:

    EXTRA_ENV="TF_GLM_DRAFT_ADAPT=1" bash scripts/serve.sh     # extra -e knobs, empty in prod
