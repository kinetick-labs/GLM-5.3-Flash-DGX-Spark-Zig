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
