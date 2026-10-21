# Qwen3.8-Flash-Next on one DGX Spark

### v5.3: NVIDIA's [FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) checkpoint on vLLM 0.32 — 9 of 12 gauntlet pages, 409,600-token context, 1.66M-token KV pool

One box, one model, three commands. The default checkpoint is
[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), served as shipped.
[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) runs on the same image: swap the `model:` line in
`recipe.yaml`, as do [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) and
[`INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) by [@azampatti](https://github.com/azampatti).

**Side by side with v5.1** — every number, gauntlet page and quality verdict: [myllmbox.com/?a=mbx-v53&b=mbx-v51](https://myllmbox.com/?a=mbx-v53&b=mbx-v51)

## v5.3 (2026-10-20)

v5.2 stays available: `git checkout v5.2`.

**New**

1. **vLLM 0.32.0**, with our patches rebased: RecoverSSM ([vllm#58863](https://github.com/vllm-project/vllm/pull/58863) by
   [@jschmied](https://github.com/jschmied), now with its exact replay of accepted tokens), the n-gram table on NVMe, the
   fused multi-step MTP draft, dynamic draft depth.
2. **Default checkpoint: NVIDIA's FP4-Local**, weights as shipped: NVFP4 experts with compressed scales (they stay
   compressed on the GPU), MXFP8 dense layers.
3. **Full CUDA graphs for decode**: table reads run off the critical path, so decode no longer leaves the graph.
4. **Faster drafts**: FP8 drafter projections, a two-stage draft head and draft-depth bookkeeping kept on the GPU — by
   [@Alssndr0](https://github.com/Alssndr0) ([#8](https://github.com/myllmbox/qwen38-flash-next-recipe/issues/8)). All
   exact: every draft is verified. Draft depth now ranges 1–7.
5. **400k context by default** (YaRN; config by [@vr8vr8](https://x.com/vr8vr8), 100 % needle retrieval at 407,552
   tokens measured by [@grau_marc](https://x.com/grau_marc)). On vLLM 0.32 it needs the M-RoPE cache fix from
   [vllm#62409](https://github.com/vllm-project/vllm/pull/62409) by [@wtogami](https://github.com/wtogami), built into the
   image (without it a request crashes the engine past 262,144 tokens). And an **fp8 KV cache**: 1,661,476 tokens in the
   same 27 GB.
6. **One image for four checkpoints**: FP4-Local (default), `hibrid48`, `hibrid48-uncensored` and INT4-AutoRound;
   `run.sh` applies each one's settings itself.

**Measured** (one DGX Spark, image v5.3, the shipped `recipe.yaml`)

| | v5.3 (FP4-Local) | v5.1 (hibrid48) |
|---|---|---|
| thinking-on request (pasture + fish), c=1: thinking / code / peak | 50.5 / 64.4 / 82.3 | 50.9 / 71.7 / 90.8 |
| gauntlet pages (pasture + fish) | **9 / 12** | 6.5 / 12 |
| max context per request | **409,600** tokens | 262,144 tokens |
| KV pool | **1,661,476** tokens (fp8) | 876,726 tokens (bf16) |
| prefill, 128k-token prompt | **2,301** tok/s | 2,053 tok/s |
| first token, 1k prompt | **0.62** s | 0.68 s |
| peak at 16 streams (thinking off, mixed) | 297 tok/s | 335 tok/s |

**Throughput** (tok/s, one DGX Spark, image v5.3, the shipped `recipe.yaml`; thinking off, every stream kept busy, 300-s windows, 3 runs each)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|---|
| mixed (code + explanation) | 54.2 | 83.3 | 119.9 | 153.8 | 179.7 | 229.4 | 266.3 |
| structured output (JSON schema) | 71.7 | 107.6 | 150.3 | 182.4 | 211.2 | 257.3 | 291.6 |
| long prose (7,000-word story) | 44.1 | 69.3 | 105.0 | 132.2 | 153.0 | 186.5 | 215.8 |
| code (TypeScript) | 71.9 | 106.2 | 144.6 | 177.5 | 202.8 | 247.5 | 283.9 |

## v5.2 (2026-10-05)

v5.1 stays available: `git checkout v5.1`.

**New**

1. **Default checkpoint: [INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)** by
   [@azampatti](https://github.com/azampatti), as published. Its 49 GB fp8 n-gram table is served from disk by the image's
   table reader like hibrid48's, so the memory goes to weights and KV; its fp8 side layers load through his
   `vllm_fp8_hybrid` module (MIT, after [@Saren-Arterius](https://github.com/Saren-Arterius)'s
   [spark-dflash-hybrid-fp8](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)).
2. **Same stack as v5.1**: RecoverSSM, dynamic draft depth up to 7, sampled drafts with block verification, 16 seats. With
   this checkpoint the draft uses its own MTP head routing 10 experts, and `run.sh` sets that up (`cache/draft-k10`).
3. **One image for all three checkpoints.** `hibrid48` and `hibrid48-uncensored` run on the v5.2 image exactly as on v5.1
   (same table map).

**Measured** (one DGX Spark, image v5.2, the shipped `recipe.yaml`; gauntlet and c=1–6 on one box, c=8–16 on a second)

| | v5.2 (INT4-AutoRound) | v5.1 (hibrid48) |
|---|---|---|
| thinking-on request (pasture), c=1, 12 runs: thinking / code / peak | **57.4 / 98.4 / 110.8** | 50.9 / 71.7 / 90.8 |
| KV pool | **777,693** tokens | 876,726 tokens |
| prefill, 128k-token prompt | **2,727** tok/s | 2,053 tok/s |
| first token, 1k prompt | **0.57** s | 0.68 s |
| peak at 16 streams (thinking off, mixed) | **414** tok/s | 335 tok/s |

**v5.1 → v5.2 per prompt** ([side by side on myllmbox.com](https://myllmbox.com/?a=mbx-v51&b=mbx-v52); thinking off, aggregate tok/s of all streams, averages of 3 runs; c=1 = one full answer, c≥2 =
300 s with every stream kept busy)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|---|
| mixed (code + explanation) | 55.6 → **64.7** | 84.6 → **111.0** | 119.7 → **145.5** | 157.8 → **186.4** | 192.1 → **225.1** | 239.2 → **286.0** | 282.7 → **338.8** |
| structured output (JSON schema) | 72.2 → **85.0** | 106.4 → **135.8** | 147.7 → **183.9** | 185.2 → **230.9** | 228.6 → **282.3** | 267.5 → **330.3** | 312.9 → **374.6** |
| long prose (7,000-word story) | 43.8 → **49.8** | 69.0 → **82.2** | 107.1 → **130.7** | 121.7 → **162.3** | 152.1 → **187.4** | 185.7 → **226.8** | 211.5 → **258.6** |
| code (TypeScript) | 76.2 → **86.3** | 104.4 → **141.0** | 145.8 → **193.4** | 174.5 → **233.6** | 209.9 → **287.2** | 243.9 → **341.7** | 286.9 → **382.7** |

## v5.1 (2026-10-02)

v5 stays available: `git checkout v5`.

**New**

1. **RecoverSSM** for the model's recurrent (GDN) layers and the PLE short conv — [vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863)
   by [@jschmied](https://github.com/jschmied), ported to 0.30. The drafts are verified from one saved state and only the
   accepted tokens are replayed: faster steps, and no per-draft state copies, so a deeper draft costs no memory. Pool
   **876,726 tokens** at depth 7 (v5: 813,457 at depth 6).
2. **Dynamic draft depth is the default**, up to **7** (v5: opt-in, up to 6). Fixed K=5: `mtp_depth.mode: off` and the two
   commented K=5 lines in `recipe.yaml`.
3. **Optional front proxy** (`proxy:` section in `recipe.yaml`, absent = no proxy): keepalive pings during long prefill and
   thinking, a generic loop guard (any phrase repeated, or `2222…` / `?!?!…` runs), your own pattern list, and optional
   logging of every request and response to `cache/requests/`. A cut answer ends with `<stopped reason="repetition"/>` (or
   `reason="pattern"`) and frees the seat. See [Front proxy](#front-proxy-v51-optional).
4. GB10 plan table for the model's small decode GEMMs, **off by default** (`MBX_SKINNY_GEMM_SM12X: "1"` in `env:`): faster
   kernels, neutral end to end on one Spark. Found and first tuned for two Sparks by
   [@sethforprivacy](https://github.com/sethforprivacy) ([vllm-project/vllm#59605](https://github.com/vllm-project/vllm/issues/59605));
   the table here is tuned for one.

**Measured** (one DGX Spark, image v5.1, the shipped `recipe.yaml`, one `bench/full.py` run)

| | v5.1 |
|---|---|
| thinking-on request (pasture), c=1 | **61.2** tok/s average · **86.2** peak |
| structured output (JSON schema), c=1 | **73.1** |
| long prose (7,000-word story), c=1 | **44.4** |
| peak at c=1 / 2 / 4 / 8 / 16 (thinking off) | **75 / 112 / 167 / 236 / 322** |
| average at c=1 / 2 / 4 / 8 / 16 (thinking off) | 59 / 84 / 128 / 197 / 282 |
| prefill, 128k-token prompt | **2,053** tok/s |
| first token, 1k prompt | **0.68** s |
| KV pool | **876,726** tokens |

**v5 → v5.1 per prompt** (same box, thinking off, v5 = the dynamic-depth runs below, average tok/s of all streams together, averages of 3 runs; c=1 = one
full answer, c≥2 = 300 s with every stream kept busy)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| mixed (code + explanation) | 55.0 → **56.4** | 82.7 → **83.3** | 120.0 → **126.3** | 151.3 → **161.4** | 181.0 → **192.1** | 214.3 → **239.2** |
| structured output (JSON schema) | 69.3 → **74.2** | 105.7 → **111.9** | 150.0 → **162.3** | 178.0 → **198.4** | 201.7 → **228.6** | 233.3 → **267.5** |
| long prose (7,000-word story) | 44.3 → **44.7** | **69.7** → 69.4 | 102.3 → **107.8** | 124.0 → **132.1** | 143.0 → **152.1** | 172.3 → **185.7** |
| code (TypeScript) | 69.7 → **74.6** | 103.0 → **107.3** | 142.3 → **151.3** | 169.0 → **185.7** | 185.3 → **209.9** | 213.3 → **243.9** |

## Dynamic draft depth

**How the dynamic depth works.** Speculative decoding drafts several tokens ahead and the model checks them in one step; how
many of them survive depends on what is being written. Long prose keeps about the first three, code and JSON often all six. A
fixed depth is therefore too deep for prose (every rejected draft is wasted work) and too shallow for code. With
`mode: dynamic` every request keeps its own depth between `min` and `num_speculative_tokens`: it watches, over its last
`window` steps, how often its deepest drafted position was accepted, and moves one or two steps deeper when that rate is above
`promote` and shallower when it is below `demote`. Requests that are batched together draft their average depth. Prose
settles at 3, code and JSON at 5–6, and a thinking-on request changes depth between its reasoning and its answer.

**v5 vs every fixed depth on one full thinking-on request** (the pasture prompt: reasoning, then an HTML/JS scene), each phase on its own, because
answers think for different lengths (averages of 3 runs, tok/s):

| phase | fixed K=3 | fixed K=4 | fixed K=5 | fixed K=6 | **v5 (dynamic)** |
|---|---|---|---|---|---|
| thinking | 50.7 | 50.0 | 49.7 | 48.0 | **51.4** |
| writing the code | 63.2 | 66.3 | 70.9 | 68.8 | **72.1** |

**v5 and v5.1 vs fixed K=3 and K=5 at every concurrency, aggregate tok/s** (same box and image, thinking off, averages of 3 runs; c=1 = one full answer,
c≥2 = 300 s with every stream kept busy; fixed K=3 / K=5 = `mode: dynamic` with `min` pinned and promotion off). The best
value of each row is bold; the mixed prompt is in the table above.

**Long prose (7,000-word story)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 44.0 | **70.0** | 103.3 | 124.3 | 143.0 | 172.0 |
| fixed K=5 | 39.7 | 61.0 | 90.3 | 108.7 | 124.7 | 144.7 |
| **v5 (dynamic)** | 44.3 | 69.7 | 102.3 | 124.0 | 143.0 | 172.3 |
| **v5.1 (dynamic, ≤7)** | **44.7** | 69.4 | **107.8** | **132.1** | **152.1** | **185.7** |

**Code (TypeScript)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 62.3 | 93.0 | 132.3 | 160.3 | 180.3 | 214.3 |
| fixed K=5 | 69.7 | 102.3 | 140.0 | 164.0 | 185.3 | 217.0 |
| **v5 (dynamic)** | 69.7 | 103.0 | 142.3 | 169.0 | 185.3 | 213.3 |
| **v5.1 (dynamic, ≤7)** | **74.6** | **107.3** | **151.3** | **185.7** | **209.9** | **243.9** |

**Structured output (JSON schema)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 61.7 | 95.3 | 139.3 | 168.3 | 193.0 | 229.7 |
| fixed K=5 | 69.7 | 103.7 | 149.3 | 178.3 | 203.7 | 232.7 |
| **v5 (dynamic)** | 69.3 | 105.7 | 150.0 | 178.0 | 201.7 | 233.3 |
| **v5.1 (dynamic, ≤7)** | **74.2** | **111.9** | **162.3** | **198.4** | **228.6** | **267.5** |

**On by default since v5.1** (deepest depth 7; the tables above are v5, deepest 6). **Fixed K=5 instead**, in
`recipe.yaml`: `mtp_depth.mode: off` and swap in the two commented K=5 lines next to `speculative-config` and
`compilation-config`.

**Knobs** (`mtp_depth` in `recipe.yaml`; `run.sh` writes them to `cache/mbx-depth.json`, and the serve re-reads that file
while it runs — edit it to change them live):

| knob | default | what it does |
|---|---|---|
| `min` | 3 | lowest draft depth, and the depth every new request starts at |
| `window` | 48 | steps of history each decision looks at (8–128) |
| `promote` | [60, 45] | +2 / +1 depth when the deepest position was accepted on more than this % of the window |
| `demote` | [25, 15] | −1 / −2 depth when below this % |
| `log` | false | `true` = one log line per decision: depth before → after and the measured rate |

The deepest depth is `num_speculative_tokens`; the capture sizes must cover every depth × seats (the shipped list covers
3 … 7 × 16).

## Front proxy (v5.1, optional)

Uncomment the `proxy:` section in `recipe.yaml`. `run.sh` then starts a second container (`qwen38-flash-next-proxy`) on the
public port and moves vLLM to `127.0.0.1:<port+1>`; `./stop.sh` removes both.

| key | default | what it does |
|---|---|---|
| `keepalive` | 30 | seconds between SSE pings while the model is silent (long prefill, long thinking) |
| `loop_guard` | on | any phrase of 3–`loop_period` chars repeated `loop_repeats` times in a row, or a `size`-char 2-char run |
| `size` / `loop_period` / `loop_repeats` | 12 / 64 / 8 | loop guard thresholds |
| `loop_fields` | `reasoning_content,reasoning` | where both guards look; add `content` to watch the answer too |
| `pattern_guard` | off | your strings, one per line in `cache/<patterns_file>`; a hit at `pattern_count` appearances |
| `stop` | false | false = log hits to `cache/logs/`; true = also end the answer with `<stopped reason="…"/>` and free the seat |
| `log_all` | off | every request + full response → `cache/requests/` (holds your users' prompts and answers) |
| `token` | — | optional Bearer token for the API |

## Measured performance (v5, one DGX Spark, `vm.compaction_proactiveness=0`)

**2026-10-01, vLLM 0.30, image v5, `hibrid48`, bf16 KV, 16 seats, Marlin MoE.** Mixed prompt (a code task plus its prose
explanation), thinking off, averages of 3 runs per row. c=1 is one full answer; c≥2 is 300 s with every stream kept busy, so a
finished request is replaced at once and its new prompt's prefill is part of the average.

| concurrent requests | `mode: off` (K=5, = v4) tok/s | **`mode: dynamic`** tok/s | per stream (dynamic) | tokens per step (off → dynamic) |
|---|---|---|---|---|
| 1 | 52.7 | **55.0** | 55.0 | 3.60 → 3.72 |
| 2 | 81.3 | **82.7** | 41.4 | 3.71 → 3.96 |
| 4 | 120.0 | **120.0** | 30.0 | 3.84 → 4.07 |
| 6 | 154.3 | **151.3** | 25.2 | 4.12 → 4.31 |
| 8 | 182.0 | **181.0** | 22.6 | 4.27 → 4.59 |
| 12 | 218.0 | **214.3** | 17.9 | 4.37 → 4.68 |

**Single-stream peak: 86.4 tok/s** — the best 10-second window, while a thinking-on request writes its code. **Prefill:
1,988 tok/s** on a 128k-token prompt, **first token in 0.71 s** on a 1k prompt. The KV pool is 830,582 tokens with `mode: off`
(K=5) and 813,457 with `mode: dynamic` (K=6 graphs); the v4 and v3 ladders are in their tags (`git checkout v4`).

## Quality (measured on this model, thinking on)

All three checkpoints, lm-evaluation-harness against a running serve, **thinking on**, temperature 0.6 / top-p 0.95 / top-k 20,
a 32k-token budget per answer, HumanEval complete and a fixed 200-question subset (seed 123123123) of the others — the same
763 questions for every column. Qwen publishes no numbers for these four tests (its card reports LiveCodeBench v6 91.9, GPQA
Diamond 91.7, IFBench 81.3, SWE-bench Pro 62.5). The v4 changes are output-exact (speculative decoding keeps the model's own
distribution), so the scores carry over.

| task | questions | INT4-AutoRound (v5.2) | hibrid48 | hibrid48-uncensored |
|---|---|---|---|---|
| HumanEval pass@1 | 164 | **93.3** | **95.7** | **94.5** |
| GSM8K exact match | 200 | **99.0** | **98.0** | **97.5** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 94.0 | **91.5** / 93.4 | **94.5** / 96.2 |
| MMLU-Pro (14 subjects, sampled by size) | 200 | **82.9** | **84.9** | **82.9** |
| answers that ran into the 32k budget while thinking (count as wrong) | 763 | 6 | 11 | 6 |

The IFEval gain is the one difference larger than the subsets' sampling error (about ±3 points); the other deltas are one to
four questions each. The abliterated body thinks shorter (median reasoning −8 %, 90th percentile −27 %) and runs away half as
often. Published leaderboard numbers use other prompts, few-shot counts and full sets — a sanity band, not a column.

## What changed

**v5.2 (2026-10-05): the INT4-AutoRound checkpoint is the default** — +13 to +40 % over v5.1; hibrid48 and its uncensored twin
stay one line away. See [v5.2](#v52-2026-10-05). v5.1 stays available: `git checkout v5.1`.

**v5.1 (2026-10-02): RecoverSSM, dynamic depth up to 7 by default, optional front proxy.** See
[v5.1](#v51-2026-10-02). v5 stays available: `git checkout v5`.

**v5 (2026-10-01): dynamic draft depth, opt-in.** Per-request draft depth from measured acceptance, live-tunable knobs,
+12 to +19 % on long prose at every concurrency, level on mixed text, JSON and code
([Dynamic draft depth](#dynamic-draft-depth)). With `mode: off` (the default) the serve is v4. v4 stays available:
`git checkout v4`.

**v4 (2026-09-25): vLLM 0.30, five sampled draft tokens, an 830k-token pool and 16 seats.**

1. **Engine: upstream vLLM 0.30** (was 0.29), with the fused multi-step MTP draft for this model's attention (proposed upstream
   as vllm-project/vllm#58449).
2. **Five draft tokens, sampled.** The drafter proposes 5 tokens instead of 3 and samples them from its own distribution
   (`draft_sample_method: probabilistic`); the model checks them with the exact probability-ratio test and block verification.
   Output quality cannot change — the text follows the model's distribution either way — but more drafts survive: 5.1 accepted
   per step on code (v3: 3.4 of 4).
3. **The n-gram table no longer takes memory.** On the first boot the image prepares a table map in `./cache` (~31G, a few
   minutes, once; it works for both checkpoints) and serves from it. The memory that frees goes to the KV pool: 8G fp8 →
   **27G bf16 = 830,582 tokens** (3.17 full 262K requests), and 16 seats instead of 8.
4. **Boots in about four minutes** instead of about twelve (`load-format: fastsafetensors`, and no per-boot compilation on 0.30).

v3 stays available: `git checkout v3` (image `…-vllm:v3`, vLLM 0.29, K=3, fp8 KV 8G, 8 seats).

**v3 (2026-09-14): vLLM 0.29 and a 4-bit output head.** The engine moved from the vendor SM121 pin to vLLM 0.29, and the
checkpoint's 1.18 GB bf16 output head became 0.33 GB of NVFP4 (`hibrid48`) — it was 27 % of every decode step: 60 tok/s
single-stream at 17.7 engine steps/s (v2: 50 at 14.4). v2 stays available: `git checkout v2`.

## Quick start

```bash
git clone https://github.com/myllmbox/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./run.sh        # downloads ~102G from HF on first run, serves the OpenAI API on :8000
```

`./stop.sh` stops it. `./view.sh` shows live stats (throughput, KV usage, speculative-decoding acceptance). Requirements: a DGX
Spark with docker + the NVIDIA container runtime, and free disk under `./cache` for the table map (~31G for FP4-Local and
hibrid48, ~52G for INT4-AutoRound). A boot reaches healthy in about seven minutes (weights ~2 min); the very first boot on a box
also prepares the map and compiles kernels (~15 min, once).

**All configuration lives in [`recipe.yaml`](recipe.yaml)** — one file: image, weights repo, port, context length,
KV budget, every vLLM flag. Nothing else to edit.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored
variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks
for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

## Which checkpoint

Four checkpoints run on this image; `recipe.yaml` ships with the first active and the others commented out under it.
Switching is comment one line, uncomment another, `./run.sh`; `run.sh` applies each checkpoint's settings itself:

| `model:` | what it is | on this kit |
|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` (default) | NVIDIA's quantization-aware-distilled checkpoint, served as shipped: NVFP4 experts, MXFP8 dense layers (GPQA Diamond 91.5 %, ToolEvalBench 88.7 % on its model card) | every v5.3 number above |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` | our NVFP4 body with an NVFP4 output head | the v5.1 numbers |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same speed as hibrid48 |
| `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | 5 of 512 experts per token, AutoRound int4 experts, fp8 side layers and n-gram table | the v5.2 numbers (KV pool on v5.3: 1,476,867 tokens) |

The uncensored repo is **gated**: open its Hugging Face page, accept the agreement, then `hf auth login` (or `export
HF_TOKEN=…`) before `./run.sh` — the kit checks both and tells you what is missing. Running several checkpoints at different
times? Set `served-model-name` to something distinct so clients and logs can tell them apart. FP4-Local weighs ~102 GB, the two
hibrid48 checkpoints 98 GB each; INT4-AutoRound is ~120 GB and prepares its own table map in `cache/` (~52G) on its first boot.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver, the page cache and the kernel share one pool, and this serve uses most of it on purpose:
the weights + the 27G KV pool + graphs and host processes. The kit never asks for your password; it stabilises memory with
what a user may do:

- **waits** after removing the old container until the box reports ≥ 100G available (unified memory takes 30–60 s to come back
  after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the image's loader reads each weight file once and drops it from the page cache as it goes.

Measured on v5.3: about 1.5–2 GB free at its lowest during 16-stream runs. If you run other things on the box, set
`kv-cache-memory` to 26G.

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0; shows the command, asks, then sudo prompts; persisted
```

`run.sh` reads the value before every launch (no privilege needed) and warns while it is not 0; it never applies the
change itself. The kernel's background page compactor migrates pages to build large contiguous blocks; on a Spark the
GPU's memory *is* those pages, so every migration first unmaps them from the GPU — measured as a 4–5 s slowdown every
~37 s. A serving box allocates once at boot and gains nothing from the upkeep.

## Tuning (recipe.yaml)

- **`max-model-len: 409600` + `hf-overrides`**: YaRN, 400k context (default). For the native 262,144: set `max-model-len:
  262144` and remove the `hf-overrides` line.
- **`kv-cache-memory`** (bytes): 27G fp8 = 1,661,476 tokens (26G for more headroom); with INT4-AutoRound `run.sh` uses 24G =
  1,476,867 tokens.
- **`max-num-seqs`** 16: with RecoverSSM each request holds one recurrent state (no per-draft copies).
- **`speculative-config`** up to 7 sampled draft tokens with block verification, depth chosen per request (`mtp_depth`).
- **`use-replayssm: true`**: RecoverSSM. Remove it for the stock GDN verify path (more memory per seat, slower steps).
- **`compilation-config`**: full CUDA graphs for decode; the capture sizes cover every draft depth × seats (16 × 8 = 128).
  Shorten them only together with `max-num-seqs`, or the upper rungs decode without CUDA graphs.
- **`block-size: 3312`**: the model's attention ring (12 at K=7) must divide the attention block; vLLM raises it to 6624.
- **`docker_args: --security-opt seccomp=unconfined`**: required by the image's table reader; the container still runs
  unprivileged otherwise.
- **`load-format: mbx`**: the image's loader. Keep it: FP4-Local's compressed expert scales go through it.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits one max-resolution image (~4.1k tokens).
- **`async-scheduling` on**. Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.
- **`mtp_depth`**: dynamic draft depth, on by default — see [Dynamic draft depth](#dynamic-draft-depth).
- **`proxy`**: optional front proxy, absent by default — see [Front proxy](#front-proxy-v51-optional).
- **`patches`** (server): vLLM patches from [`patches/`](patches/), applied at launch over the image's files; the image
  itself is unchanged. A patch made for an older image stops `run.sh` if it does not fit. The QSA logits-workspace fix
  ([vllm-project/vllm#57105](https://github.com/vllm-project/vllm/pull/57105) by Thien Tran, reported for this kit by
  [@anzax](https://github.com/anzax), #5) is part of vLLM 0.32.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v5.3` — upstream `vllm/vllm-openai:v0.32.0` plus patches, each an anchored or sha256-checked
edit that refuses to apply twice and fails the build if its target moved:

1. **RecoverSSM** ([vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863) by
   [@jschmied](https://github.com/jschmied), with its exact replay of accepted tokens), base files sha256-checked,
2. **the n-gram tables** (NVFP4, and INT4-AutoRound's fp8) on the image's table library (binary), read from NVMe off the
   critical path,
3. **fused multi-step draft metadata** — the code proposed upstream as
   [vllm-project/vllm#58449](https://github.com/vllm-project/vllm/pull/58449),
4. **dynamic draft depth** in full CUDA graphs, with the depth library (binary),
5. **the image's loader** and the Marlin MoE kernel for compressed NVFP4 expert scales (binaries),
6. **faster drafts** by [@Alssndr0](https://github.com/Alssndr0) (Apache-2.0): FP8 drafter projections and the two-stage draft
   head, with its bases for FP4-Local and hibrid48,
7. the output-head and hyper-connection quantization configs, the draft scale, and GB10 small-batch kernels for the bf16
   output head and the MXFP8 layers,
8. **INT4-AutoRound support**: its fp8 side layers (azampatti's `vllm_fp8_hybrid`, MIT) and its fp8 n-gram table,
9. **the M-RoPE cache fix for YaRN**: backport of [vllm-project/vllm#62409](https://github.com/vllm-project/vllm/pull/62409)
   by [@wtogami](https://github.com/wtogami),
10. **the front proxy** (`mbx_proxy`), started only with a `proxy:` section.

Digest: `sha256:cf52ff577851827110b34958bb21ced187369c2851f7de798eecd2de45ffa1de`.

v5.2 (`…-vllm:v5.2`, vLLM 0.30, digest `sha256:76eda2f6c0e51fd2a9913d97b6f36abd829965ca8cad661195fd70ec763ee7a3`), v5.1
(`…-vllm:v5.1`, digest `sha256:733f1a576e7e5a4192b0475ac4c3c53b5e5f27a182e92de0972ce88853ce1edd`), v5 (`…-vllm:v5`, digest
`sha256:695882cca3c64ff49d538fd37d72ae7db4537d039ff633da67909648b5ba4676`), v4 (`…-vllm:v4`, digest
`sha256:51629f438f5ba3f7a96db110826c783d69b91a6851c43fb07d447644f157dcc4`), v3 (`…-vllm:v3`, vLLM 0.29, digest
`sha256:61d2bc6ba5977024895734d0ef94918ac806d936c4e17a263906e246599f9862`), v2 and v1 stay available: `git checkout v5.2` / `v5.1` /
`v5` / `v4` / `v3` / `v2` / `v1`.

## The full box

This kit serves one model, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS
tunnel, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT. The table library is distributed in binary form inside the image.
