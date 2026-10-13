# GLM-5.3 Full on a four-node DGX Spark ring: EXL3/TensorFold and EXLR8/GR8 field report

**Test window:** 8–12 October 2026  
**Purpose:** Share reproducible observations and actionable follow-up tests with engine developers.  
**Scope:** Four NVIDIA GB10 DGX Spark nodes, 128 GiB unified memory each, connected as a switchless RoCE ring. All reported measurements are from this cluster. Published benchmark claims from other systems are excluded.

**Follow-up (12 October 2026):** We tested a newer pinned GR8 build on three additional TP4 boots. A 32K conversation remained cached after an intervening conversation, while prefill cancellation and the targeted Docker task still failed their gates. The two 64K requests each reserved 66,050 tokens, above that build's `SHARE_LIMIT=65536`; their serialization is consistent with the documented rule. See the [Goal 4.1 follow-up](GOAL4_1_FOLLOWUP.md) for pins, measurements, memory limits and the separate status of PR #41. The original paired quality scores below were not rerun on the newer build.

## Executive summary

GR8 at the pinned revision below ran GLM-5.3 Full EXLR8 over its native, two-rail verbs transport on our existing TP4 ring. On a matched 1,024-token single-stream generation plan, GR8 home delivered a median **34.025 completion tokens/s**, versus **27.731 tokens/s** for our TensorFold Full EXL3 3.0 bpw MTP3 profile: a **1.227×** client-observed ratio. Cold-prompt time to first token was also lower for GR8 on the matched 8K and 16K plans. These are useful results for short, controlled sessions.

We did **not** promote GR8 to our general multi-agent Full engine. In the pinned version, two 64K-history requests were serialized on two boots even when their combined reservation fit the configured KV capacity. A client disconnect during 32K prefill took approximately 24.7 seconds to reach an idle/abort state. Exact JSON and some tool workflows were also less reliable under the delivered API contract. These observations concern the **whole engine, frontend, weights, and harness combination**; they do not establish that EXLR8 quantization itself is intrinsically less accurate than EXL3.

The GLM-5.3 Flash NVFP4 service is included as an operational reference where measured. Flash and Full are different models, and those historical measurements are not a strict three-way weight-format A/B.

## Versions and test boundary

| Component | Pinned test version or configuration |
|---|---|
| GR8 | [`kindlingai/gr8`](https://github.com/kindlingai/gr8) commit `96aba98773768b59ba2efa92c7529524ab8b4ddf`; GHCR image digest `sha256:9abc77d2fc65ea9a48439c7aaea5d2102266c8348670b712e9789f0a093a7f90` |
| EXLR8 weights | [`kindlingai/glm-5.3-exlr8-k2-k3-k4`](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) revision `0a1b3c15d9e1519301b20c6a1efbef4c5f628d15` (`v0.5`), weight-set ID `2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb` |
| GR8 profiles | TP4 `SHAPE=ring`, native RoCE, MTP2, DFlash2 off for the home/bit325 comparison, `CAP=262144`, `SESS=4`. The low-memory MTP-versus-adaptive-DFlash2 control used `CAP=131072`, `SESS=2`. |
| EXL3 baseline | [`bertholomus/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw) checkpoint revision `7f0573836c44803030912ffe5d051ca7587edfd6`, running a locally patched TP4 TensorFold engine derived from [`bertholomus/glm-5.3-tensorfold-tp4-4xgb10`](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10) at `f971ed3bfb067b1860c701a64e54209b4ce8d8bb`. The benchmark profile used MTP3; the quality replay used 32K context, q5b KV, one serial slot. The separate FAST profile supports 64K and four slots; LONG uses DCP4 and one slot. |
| Flash reference | GLM-5.3 Flash NVFP4 TP4, running a different checkpoint and inference engine. Its figures are historical local controls, not GR8-versus-TensorFold paired measurements. |

Our TensorFold ring transport, scheduling, and cache changes are local patches. The exact patched source and all raw traces are not included here, so the EXL3 numbers describe this installation rather than an unmodified upstream build. We did not change host drivers, CUDA, firmware, cables, or system memory-compaction settings for this study.

### Switchless transport evidence

The physical ring alternated same-numbered port pairs (`f1↔f1`, then `f0↔f0`). We mapped the four HCA functions per host into GR8's logical `RING_DEVS` order, reversing that order on two opposite ranks, without changing cables or host interfaces. During a fixed 207-token completion, all **16/16 rank HCA functions** had RC/RTS QPs, all **8/8 neighboring rail pairs** were reciprocal, and RDMA counters rose on all four hosts with no new error counters. The measurement window was 16.56 seconds and included counter snapshots; its roughly 3.8 GB per-host traffic must not be attributed to the request alone. TCP traffic from the engine PIDs was only a few kilobytes per host in that window. This supports native RoCE operation; diagonal relaying is inferred from topology, counters, and the pinned code path rather than a packet-level trace.

## Protocol and what constitutes a paired result

- **Matched C1 generation:** Three repetitions of the same long prompt plan, with `max_tokens=1024`, temperature zero, the same request fields apart from model ID, and exactly 1,024 completion tokens reported by each engine in every pair. The client measures the interval after first SSE output; reported completion tokens may include reasoning.
- **C2/C4 generation:** The same request plans were sent to both systems, but actual completion lengths differed in every paired C2/C4 group. We report the observed aggregate rates without treating their ratios as controlled speedups. All aggregate windows include admission, first-token time, and the slowest completion.
- **Cold prefill:** Each plan has a unique prompt to avoid accidental prefix hits and reserves 16 output tokens. Times are client time to first token, including API and admission overhead. The 8K/16K GR8 and contemporary TensorFold controls used the same text and near-identical token counts. The Flash and optimized EXL3 FAST/LONG figures come from earlier series with different engine profiles.
- **Quality:** The original eight cases and the extended 40-case corpus used frozen prompts and oracles. All candidates used temperature zero and explicit output limits. TensorFold implements a numeric `thinking_budget`; GR8 used `reasoning_effort=high` without a numeric cutoff. We therefore compare delivered system behavior, **not** a perfectly matched reasoning budget or a pure quantization change.
- **Agentic tasks:** Seven isolated, executable fixtures were repeated three times per Full engine under the same corrected harness, tool set, temperature zero, `tool_choice=auto`, `reasoning_effort=none`, `max_tokens=2048`, and at most ten tool iterations. **Correction after upstream review (12 October):** in the pinned GR8 frontend, `none` is forwarded to the [pinned EXLR8 chat template](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/blob/0a1b3c15d9e1519301b20c6a1efbef4c5f628d15/chat_template.jinja), which maps any effort other than `low` or `high` to `Max`; the frontend's `enable_thinking` kwarg is unused by that template. These agentic scores remain observations of the delivered API behavior, but the two engines did **not** have a matched no-thinking policy. There was no Flash arm for this campaign.

## Generation throughput

| Local measure | TensorFold Full EXL3 MTP3 | GR8 EXLR8 home | Interpretation |
|---|---:|---:|---|
| Matched C1, 1,024 tokens, repetitions 1/2/3 | 27.860 / 27.294 / 27.731 tokens/s | 33.732 / 34.047 / 34.025 tokens/s | Median ratio **1.227×** for GR8; all three paired lengths match. |
| C2 aggregate, repetitions 1/2/3 | 31.242 / 21.780 / 24.608 tokens/s | 37.528 / 33.190 / 34.209 tokens/s | Descriptive: emitted lengths differ in each pair. |
| C4 aggregate, repetitions 1/2/3 | 38.108 / 31.115 / 34.280 tokens/s | 49.810 / 42.690 / 42.094 tokens/s | Descriptive: emitted lengths differ in each pair. |

The earlier operational series measured approximately **69.8 tokens/s C1 and 128.1 tokens/s C4 for Flash**, **27.7/48.4 for EXL3 FAST**, and **33.9/44.9 for GR8 home**. These series had different output lengths and profiles; they are useful capacity landmarks rather than a strict cross-model ratio. GR8 bit325 reached **32.864 tokens/s C1** on the same three long-plan lengths as home (home **33.934** on its first boot, an observed **3.15%** difference). Its C2/C4 rates, **31.892/40.230**, are descriptive because completion lengths differ.

Four additional code/prose/reasoning/JSON C1 plans consumed 1,024 reported GR8 completion tokens each but emitted **no visible content** on that boot. We do not count their internal-token rates as useful answer throughput. This is one reason to report visible output and stop reasons alongside API token counts.

## Cold prefill and prefix reuse

| Prompt depth | Flash reference TTFT | EXL3 FAST TTFT | GR8 home TTFT, boot 1 / boot 2 | GR8 bit325 TTFT |
|---|---:|---:|---:|---:|
| 8K | 1.82 s | 9.37 s | 5.934 / 5.877 s | 5.783 s |
| 32K | 6.79 s | 38.35 s | 24.019 / 24.805 s | 23.654 s |
| 64K | 13.38 s | 80.86 s | 50.041 / 50.522 s | 49.481 s |
| 128K | 26.98 s | outside FAST window | 111.218 / 113.187 s | 109.462 s |

GR8 home used 8,180 / 32,767 / 65,514 / 131,051 prompt tokens at those depths. The 8K–128K client TTFTs were within 3.27% between two fresh boots. The contemporary same-text 8K/16K TensorFold MTP3 control measured **15.532/31.570 s**, versus **5.934/11.773 s** for GR8; each depth has one paired run. These times do not isolate a prefill kernel and cannot by themselves establish a production percentile.

An **immediate exact 32K repeat** reached 0.175 s TTFT on GR8 home with a native S2 retained-prefix hit. The optimized EXL3 FAST profile measured approximately 0.422 s, and Flash approximately 0.634 s, in their respective qualified cache series. GR8's same 32K conversation returned to approximately **23.4 s** after a different 32K or 64K conversation. The immediate hit is real; multi-conversation retention is not established. At this commit, the GR8 Prometheus cache counters did not expose the observed S2 hit/miss decisions, so the native log and client trace were used.

## Context and concurrent long histories

| Profile | Locally demonstrated context behavior | Session behavior |
|---|---|---|
| Flash reference | Exact retrieval in a 256,061-token prompt, marker at 238,120. | Short-request C4 throughput measured; long-history parallelism was not part of this comparison. |
| EXL3 FAST | 64K configured and exercised. | Four slots; mixed prefill/decode pauses remain an operational consideration. |
| EXL3 LONG | Exact retrieval in a **1,020,053-token** prompt with a one-million-token window; cold TTFT approximately **66.17 minutes** in the optimized final profile. | One slot. |
| GR8 home | 128K requests completed across boots. A near-262K prompt failed with an S2 prefill error; **262K is not a qualified stable maximum**. Exact 128K retrieval quality was not established by the throughput probes. | Short histories can overlap. Two 64K requests were serialized on two boots despite a combined reservation under `CAP=262144`: observed `running=1`, `waiting=1`, no overlapping SSE generation. |
| GR8 bit325 | 128K request completed on one boot. | Long-history concurrency was not separately qualified. |

For the scheduler boundary, two requests with approximately **32,073 tokens reserved each** reached two running sessions; above the replicated-context threshold, approximately **34,261 reserved per request**, only one ran. Every test specified `max_tokens` to avoid an implicit large reservation. A new 8K prompt overlapped an active decode, while a new 32K prompt waited for it to finish. Thus a headline C4 rate on short prompts should not be used as a proxy for four long-running development agents.

**Upstream changes after our pin:** The [current GR8 changelog](https://github.com/kindlingai/gr8/blob/main/CHANGELOG.md) documents a TP4 `SHARE_LIMIT` of 65,536 tokens, on-demand S2 KV page growth, and optional KV tiers for evicted prefixes. Those changes directly address scenarios that were restrictive in our pinned build. We have **not** run this A/B on those newer revisions, so the 32,768-token boundary and cache-eviction result above are version-specific observations, not claims about current `main`. The new 65,536-token setting also changes the default maximum-context/capacity trade-off and needs its own memory qualification.

## Functional quality: distinguish answer content from output contract

| Corpus and category | TensorFold EXL3 MTP3 | GR8 EXLR8 home | GR8 EXLR8 bit325 |
|---|---:|---:|---:|
| Eight original cases | **8/8** | 4/8 | 6/8 |
| Forty extended cases, total | **16/40** | 15/40 | 15/40 |
| Extended Python | 5/12 | **9/12** | **11/12** |
| Extended FastAPI | 2/4 | **3/4** | 1/4 |
| Extended raw JSON | **9/16** | 3/16 | 3/16 |
| Extended static text contract | 0/8 | 0/8 | 0/8 |

The eight tasks cover Python idempotency, retry debugging, a ring-topology JSON answer, FastAPI, React/TypeScript, exact-ID Docker stopping, deduplication tests, and an outbox architecture plan. On that corpus, bit325 recovered the topology and Docker cases that home missed; both EXLR8 manifests still failed retry debugging and the outbox plan. EXL3 passed all eight on a contemporary MTP3 replay. A historical Flash run passed seven of eight with a larger completion allowance; its exact-ID Docker script failed one of ten fake-Docker scenarios. These are small, protocol-dependent samples, not population-level model rankings.

The raw JSON deficit merits a narrower diagnosis. The frozen grader calls `json.loads(content)` directly. GR8 home wrapped **13/16** JSON responses in Markdown fences, and all 13 failed that raw-output contract. In **12/13**, the JSON *inside* the fence exactly matched the expected object. TensorFold fenced **5/16**; all five inner objects matched. A symmetric, **exploratory** removal of only canonical JSON fences would yield **15/16 GR8 versus 14/16 TensorFold**. It does **not** replace the official **3/16 versus 9/16**: a client requiring machine-parseable raw JSON receives the actual fenced output. It does show that most of this difference is an API/output-format issue rather than wrong JSON values.

At the pinned GR8 revision, the frontend accepted an invalid `response_format` value instead of enforcing the requested schema, and `tool_choice=required` or a named tool choice returned HTTP 400. Basic synthetic tool interactions passed, including multiple and nested arguments, but those probes do not substitute for complete agent tasks. An initial logging mismatch between `message.reasoning` and `message.reasoning_content` was corrected for later records; original raw responses were retained, and that logging correction did not change the frozen output-content grades.

The corrected agentic harness yielded **6/21 strict successes and 9/21 green final oracles for GR8 home**, versus **4/21 strict and 12/21 green final oracles for EXL3 MTP3**. A strict success required both a green executable oracle and a terminal `finish` tool call. GR8 reliably completed Python and TypeScript (3/3 each); its FastAPI changes passed tests in 3/3 but omitted `finish`. EXL3 had more green final oracles, notably Docker and unit-test creation. The first harness revision had a Git patch-path defect and is excluded from this ranking. In a later full-task Docker check, raising GR8 `max_tokens` from 2,048 to 4,096 did not restore a tool call: the last turn ended at `length` on all four runs, with no write and a red oracle. A separate isolated third-turn replay *did* call `read_file` at both budgets. We cannot attribute this discrepancy to the budget alone or to the tool parser without additional controlled tests.

## Memory, speculative drafting, and recovery

- Home ran the 1K–128K cold-prompt series on two boots with no observed swapout. Its lowest available host memory in the repeated prefill series was **8.596 GiB**; this is close enough to the configured 8 GiB guard to require continued monitoring, not a warrant to raise capacity.
- The default-capacity adaptive DFlash2 arm was stopped after available memory fell to **7.491 GiB**, below our 8 GiB guard. A separate `CAP=131072`, `SESS=2` comparison measured C1 medians **34.034 tokens/s MTP2-only** and **33.960 tokens/s adaptive**. DFlash2 was selected in 109 of 459 rounds of the first C1 and not in the next two. No DFlash2 speedup was demonstrated.
- Decode-stream disconnects were observed to cancel under C2/C4. During a 32K prefill disconnect, the system reached idle/abort only **24.665 seconds** later; the immediate recovery request had **24.219 seconds TTFT**. Per-rank GPU work and KV release were not proved, so this remains a prefill-cancellation gap.
- The near-capacity prompt produced an S2 error within an HTTP 200/SSE stream and subsequently unhealthy frontend responses. Benchmarks must treat an SSE error as a failure even if the transport ends with `[DONE]`. The exact internal cause is undetermined.
- After the campaign, the four rank containers were stopped by recorded IDs; no candidate GPU processes or user RDMA QPs remained, and the final ring check passed **16/16** directional probes with no observed RoCE error/discard increments. This is a local stop/recovery check, not a general fault-injection certification.

## Constructive follow-up tests for engine and frontend maintainers

1. **Make JSON mode a real contract.** Either enforce `response_format`/schema and reject unsupported values, or explicitly report that the mode is unsupported. Include a raw `json.loads(content)` regression, plus an optional client-normalized score reported separately. Keep fenced and unfenced outputs visible in traces.
2. **Expose a bounded thinking policy and useful-output accounting.** Document how `reasoning_effort`, `thinking_budget`, `max_tokens`, and visible output interact; test prompts that exhaust 1,024 and 2,048 tokens without an answer. Record separate reasoning and visible token counts and stop reasons.
3. **Complete tool-choice behavior.** Test `auto`, `none`, `required`, and named tools; nested arguments; streamed arguments; multi-turn tool results; and a terminal tool call after successful code edits. Return an explicit unsupported-feature error where a mode is not implemented.
4. **Re-test long-history scheduling on the newer sharing limit.** Reproduce two 64K histories with explicit output reservations and a combined reservation below CAP. Record per-request queue, running and decode intervals, and reason for exclusive admission. Repeat on both sides of the new 65,536-token limit, with a generation already active, while monitoring the capacity and memory cost.
5. **Make prefill cancellation prompt.** Close the SSE client during admitted prefill, measure time to stop compute and release KV on all ranks, then immediately admit a short request. Include timeout and partial-rank failure recovery.
6. **Clarify prefix-cache retention.** Report actual S2 hits, partial hits, misses and evictions through metrics, then test immediate repeat, alternating conversations and return after another 32K/64K prompt at the same capacity. Repeat with the newer optional KV tier enabled and record its latency, storage, and recovery costs separately.
7. **Harden the capacity boundary.** Test near-CAP admission and an S2 prefill failure without reporting a successful `[DONE]`; verify a clean health state, no orphaned rank, and a reproducible restart.
8. **Attribute speculative decoding per request.** Emit timestamped MTP/DFlash2 selection, draft acceptance, verification cost and memory for each request before claiming a drafter speedup.

## Interpretation

GR8 demonstrates a credible **short-session C1 and cold-prefill advantage over this locally optimized EXL3 Full profile**, plus a very fast immediate 32K prefix hit. The current long-history scheduler, prefill cancellation, and frontend output contracts prevented us from using it as our default multi-agent Full service. The quality data are mixed rather than a simple quantization ranking: GR8 did better on several Python cases, while much of its raw JSON loss came from Markdown fencing. bit325 improved two original tasks but did not improve the 40-case total and was measured on one boot.

We would welcome an upstream reproduction using the pinned versions and a follow-up on the contract and scheduler tests above. A corrected frontend or scheduler should be compared using the **same prompts, output limits, oracles, and client-visible metrics**, with a new version pin and raw stop reasons. We would then re-evaluate promotion on complete development tasks, not only token throughput.
