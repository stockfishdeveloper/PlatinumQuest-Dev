# Improvement-loop plan (from the Fable / Astra planning conversation, 2026-10-29)

Source: CONVO.md at the repo root (the agreed plan and reasoning). Baseline: commit d2ad44771 (speed cap 11,
FAST_SUPPORTED off). Nothing below has started; the operator decides when phase 1 begins.

All durations are MY wall-clock time, working continuously, with runs overlapping build work where possible. They are
estimates, not measurements, except where a measured anchor is named. Anchors: a 20-pair native campaign ~1.5 h and 110
pairs ~8 h (10-28); a sim_bench whole-group batch 30-60 min (10-16); sim_bench itself was built in one night (10-16);
the points ledger tool took ~2 h (10-22). Calendar times assume a start at 12:30 MDT on 10-29 and no stop.

| Phase | Deliverable | My time | Finish (if started 12:30 10-29) |
|---|---|---|---|
| 1. Foundations | deterministic, conformance-checked sim with spawn sampler and respawn | 28-40 h | 10-31 ~12:00 to 11-01 ~06:00 |
| 2. Headroom pilot | the decision table: which component limits points | 40-50 h (half of it runs) | 11-02 evening to 11-03 |
| 3. Build | the component the pilot chose, plus the full-round sim | 4-7 days | 11-08 to 11-10 |
| 4. Confirmation | native rounds at live timing, predeclared criteria | 0.5-1 day | 11-09 to 11-11 |

## Phase 1: foundations (28-40 h)

Gate to pass: sim A/A identical to the decision; event conformance within declared tolerances on the regimes used for
labels; spawn sampler distributions match recorded rounds; throughput and memory measured.

| # | Task | My time | Notes |
|---|---|---|---|
| 1a | Baseline snapshot: hash, flags, runtime config archived; remeasure 10 native rounds for the record | 0.5 h build, 1.5 h run | one round per map at live timing |
| 1b | Deterministic sim: fixed candidate order and tie-breaks, fixed virtual work budget; A/A check, 20 legs x 3 maps | 3-5 h | sim_bench has the virtual clock; parallel_search ordering is the suspect |
| 1c | Controller state at a snapshot: replay prefix via `--lead`, or one declared reset protocol for all arms | 2-3 h | labelled "reset-state benchmark" where the prefix is not available |
| 1d | Event conformance tool: recorded controls replayed through the sim on 10 recorded rounds; pickup ticks, powerup timers, OOB and respawn timing, per-reply state error; tolerances declared first | 4-6 h build, 1 h run | repairs are the unknown: 0-8 h budgeted; a material mismatch blocks only the affected label regime |
| 1e | Respawn semantics in the sim: OOB detection, respawn point and delay, inventory as the game does it | 3-4 h | today the sim stops at a fall |
| 1f | Solo spawn sampler: port the spawn selection of huntGems.cs (gem groups, centre choice, radius growth, weights, testSpawn), powerup respawn timers, round clock; distribution checks (spawn sizes, spawn-to-spawn distances, gems per round) against recorded rounds | 8-12 h | the multiplayer branch (leader position and score) is documented, not ported |
| 1g | Throughput and memory measured: rounds per hour per process for the planner and for a 3 ms policy; processes that fit in RAM | 1 h | replaces the arithmetic estimates |
| 1h | Native `--fixed-lead` A/A (the cancelled det_aa): 3 maps, two identical arms | 0.5 h build, 1 h run | a diagnosis mode only, never the deployment test |

## Phase 2: headroom pilot (40-50 h, about half of it unattended runs)

Gate to pass: one table per arm per ledger bucket of common-horizon points (ratio of equal-map means, uncertainty
clustered by source round and layout), the clear-time diagnostics beside it, and the compute curve. The operator sees
the table before anything in phase 3 is built.

| # | Task | My time | Notes |
|---|---|---|---|
| 2a | Situation suite: ~200 situations drawn uniformly in round time from held-out rounds, stratified by the ledger buckets (approach ~124 s, in-group ~83 s, stalls, respawns), dev/eval split by source round and layout, hard subset reported separately | 2-3 h | |
| 2b | Branch-and-return harness: from a situation, execute the top-K candidates for 20-30 s with sampled planning futures (common random numbers across branches) and reserved evaluation futures; every forecast charged; per-candidate reason log (absent / rejected / admitted / lost on ranking) | 6-8 h | the first label set for phase 3 falls out of this |
| 2c | Feasibility pass: one map, 20 situations, all arms at 3x; then freeze effect floors and confidence rules | 2 h run, 1 h | criteria frozen before the reserved set is opened |
| 2d | Arms 1-3 and the gate audit over the 200 situations | 2-3 h wall | ~11 core-hours at K=5, 25 s continuations; 8 processes |
| 2e | Arm 4 ladder 3x / 10x / 30x / 100x, dominated budgets stopped early, 100x on a subset if needed | 8-12 h wall | the ceiling is a cap, not a quota |
| 2f | Analysis and the decision table; one predeclared fresh extension if inconclusive | 3 h | |

Decision rule (provisional, frozen at 2c): ranking first if the deployable out-of-sample selector gains about 5 percent
of common-horizon points; otherwise compare measured incremental gains and costs of broader proposals, gate admission and
runtime; a flat 10x-100x curve ends that search design.

## Phase 3: build what the pilot chose (4-7 days; branches)

| Branch | Work | My time |
|---|---|---|
| 3A ranking | label generation at scale from the phase 2 harness (~2,000 situations), transferable features (marble state, visible gems relative to the marble, inventory, meter, time left, local terrain), ranker training, held-out ranking accuracy, closed-loop sim rounds, integration inside the physics search | 3-5 days |
| 3B proposals and gates | a broader candidate generator (sampled or refined control sequences, multi-event) with executed-consequence admission replacing the heuristic terminal gates where the audit showed cost | 3-5 days |
| 3C full-round sim (always, in parallel) | the sampler and respawn in closed-loop whole rounds, per-map score agreement with native, repeated play / search / learn cycles | 1-2 days |

Gate to pass: a candidate with at least the frozen held-out gain in the sim and a full-round sim gain at least twice the
native campaign's resolution at the predeclared sample size.

## Phase 4: confirmation (0.5-1 day)

| # | Task | My time |
|---|---|---|
| 4a | Native deterministic diagnosis (fixed-lead mode) to explain the candidate's effect | 1 h run |
| 4b | Fresh native rounds at actual live timing (queue, wall deadlines, available lead), 40 pairs, pass criteria written before the run (sign-consistent across seeds, effect above the floor, no map loss beyond the bound), all ten maps reported | ~3-4 h run, 1 h |
| 4c | Situation replays as regression cases; best-scores sheet and CSV updated on any new best | 1 h |

## What can stretch the estimates

- Conformance repairs (1d): the sim has known small contact discrepancies (Mega chained contacts); if they touch a label
  regime, that regime waits.
- Spawn port details (1f): mission gem groups and spawn weights must be read from the mission files; the collision
  export already parses missions, so this is expected to be reuse, but it is unverified.
- Memory: each sim process carries ~2 GB of caches; 8 processes, so the arm 4 ladder is the long pole of phase 2.
- Phase 3 is the only phase whose content is not known yet; its estimate is the range across branches.

## Standing rules that apply

Points only; lobby-valid (the policy never sees a sampled future); transfer to any map (relative features, no map
names); settings baked into defaults; no training launch without the plain plan and the operator's OK; one session per
plan; every wait is a Monitor with a stall alert; the operator can stop at any milestone.
