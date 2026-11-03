# Edge-hit skill, campaign 2 review: 2026-11-03

**Status: standalone skill evaluation complete on sealed suites. Stopped before planner integration, as authorized.** No production planner, AI script, physics source or production DLL was changed. The skill lives in `ml_agent/tools/edge_skill/` and nothing in production imports it.

Open [the sealed showcase viewer](review_sealed_showcase.html) (36 varied accepted successes across the three sealed suites, 1x and 0.25x) or [the full real-map sealed viewer](review_real_test.html) (all 180 real-map trials, declines and failures included). Both show recorded game motion over collision geometry, not rendered video. Raw controls, per-tick game states, contacts and forecasts are retained in `native_*_test_model_v6a/`.

## Headline

Frozen controller + proposer v6a (hashes in `freeze.json`, frozen before the sealed suites were opened), budget 48, policy execution, unpowered:

| Sealed suite | Starts | Accepted | Accepted and succeeded natively | Sim/native mismatch among accepted |
|---|---:|---:|---:|---:|
| Flat generated geometry (2 untouched maps, 32 cells) | 576 | 222 (38.5%) | 222 (100%) | 0 |
| Sloped generated geometry (16 untouched cells) | 288 | 122 (42.4%) | 122 (100%) | 0 |
| Real maps (30 sealed edge locations, all 10 maps) | 180 | 72 (40.0%) | 72 (100%) | 0 |
| **Total** | **1,044** | **416** | **416** | **0** |

Pooled Wilson 95% interval for 416/416 is 99.1 to 100%. Trials share geometry cells and edge locations, so this is not an independent-trial certificate; the per-suite lower bounds are 98.3% (flat), 97.0% (sloped) and 94.9% (real maps). Declined starts were executed with the best failed candidate as an exploratory arm: 0 of 628 succeeded, so the gate is not throwing away working plans.

Compared with campaign 1 (11-02): real-map admission 0/12 then, 72/180 now; accepted execution 19/19 then, 416/416 now; later departure below the landing height after a successful trial 17 of 19 then, 16 of 416 now.

## Coverage by approach

| Suite | ground run-up | air rising | air descending | low-lip roll |
|---|---:|---:|---:|---:|
| Flat sealed (native) | 188/360 | 1/90 | 1/90 | 32/36 |
| Sloped sealed (native) | 120/192 | 2/48 | 0/48 | n/a |
| Real maps sealed (native) | 20/66 | 29/54 | 23/60 | n/a |

Numbers are accepted/starts; every accepted trial succeeded. The generated suites' airborne starts are arbitrary random airborne states and are mostly unreachable by design; they measure that the gate declines them, not controller weakness. The real-map airborne starts are synthetic jump encounters built from reachable trajectories, and there the skill accepts about half. Real-map ground starts are the weakest bucket (30%), mostly because random start distance/speed combinations on real floors often cannot reach the lip at a usable contact height.

Per real map (accepted/starts, all accepted succeeded): KOTM 9/12, Fractured 13/24, Concentric 8/22, Sprawl 7/23, Duplex 7/23, Bowl 6/12, CubeIsle 6/12, Vortex 6/12, Cragmire 5/18, TreasureBox 5/22.

## Reliability detail

- Game-reported falls during the 4.1 s recording horizon among accepted trials: 9 flat, 2 sloped, 1 real map (12 of 416). All occurred after the trial's success window (supported landing, 0.512 s of requested exit velocity, 0.384 s supported braking). The skill certifies a landing and exit, not the route that follows.
- Later departure below the landing height after success: 11 flat, 4 sloped, 1 real (16 of 416).
- Exact-action replay of every accepted trial through the simulator matches the game to at most 0.024 units (flat), 0.010 (sloped), 0.069 (real maps). Per-reply one-step prediction error on accepted trials was 0.0 at float precision. Large replay errors (up to 2.2 units) occur only in declined failed trials after chaotic bounces.
- The single "mismatch" line in the sloped suite is a declined start whose unaudited forecast passed the geometric check but failed the contact audit; it is a decline, not a physics disagreement.
- Zero setup errors across 1,044 native resets.

## Proposer and latency

| Model | dev flat | dev real | dev sloped | median / p95 ms (quiet machine) |
|---|---:|---:|---:|---:|
| v6a (frozen) | 240/576 | 101/194 | 129/288 | 46 / 75 (dev, under load); 29 / 84 (sealed flat) |
| v7a (neighborhood-augmented, 19,929 examples) | 238/576 | 98/194 | 134/288 | 32 / 85 |

v6a and v7a tie at 470/1,058 on development; v6a was frozen because it already carried native confirmation. The neighborhood augmentation (4,392 perturbed solves) did not improve coverage; it is retained as data only.

Sim admission from nominal starts versus native admission from the actual post-reset state: flat 239 vs 222, sloped 129 vs 122, real 95 vs 72. The real-map gap (23 starts) is the nominal-versus-observed sensitivity noted in campaign 1; it is now a coverage loss rather than a total failure.

Planning cost on accepted native roots (standalone, one thread): median 20 to 25 ms, p95 44 to 73 ms. Evaluation-wide p95 is 84 to 97 ms including declines, which spend the whole 48-candidate budget. The 64 ms reply interval is not certified by lockstep runs; a real-time test with the production planner's deadline is part of integration, not of this report.

## What was built (campaign 2)

- Contact-conditioned feedback controller with queue-aware planning (forecasts the already committed 64 ms before choosing the next correction); 1/16 input quantization instability fixed by the stable-input encoding.
- Private C++ controller DLL compiled from the unchanged production physics source, hashed under TEMP; Python/C++ parity on 96 flat and sloped trajectories, 290 compiled contact replays identical to the Python audit.
- Independent contact audit (lower-half lip impulse, no wall assist, approach mode) on both forecasts and recorded native substeps.
- Geometry: 128 flat cells over 8 maps and 64 sloped cells over 4 maps, split before optimization; 60 real-map edge locations (6 per map), alternating development and sealed.
- Teacher labels: 609/1,152 flat stable, 162/240 forced descending, 282/576 sloped, 108/128 sloped descending, plus 4,265 exact airborne continuations and the 4,392-case neighborhood set.
- Native development confirmation before the freeze: 88/88 real, 127/127 sloped, 33/33 flat accepted trials.

## Not established

- No points claim. These are landing-skill trials from teleported resets, not route opportunities in a round.
- No real-time deadline certificate (lockstep only).
- Active Blast or powerups, wall-face rebounds, obstructed landings and non-flat approach floors are outside the tested envelope.
- Coverage of arbitrary airborne states is near zero by construction; the reachable airborne encounters are the meaningful airborne number.
- Falls after the success window show the handoff back to the planner is unverified.

## Next stage (paused for review)

Milestone 5 of the plan: detect local lip opportunities in the planner, invoke the skill within the shared budget with two to four proposals, expose predicted time, landing state and measured reliability to ranking, audit generation/admission/ranking/execution separately, then paired full-round simulations and fresh native rounds. The decision to integrate is the operator's.

## Reproduction

```powershell
$env:PYTHONPATH = 'ml_agent'
$py = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
& $py -m unittest tools.edge_skill.test_skill -v
& $py -m tools.edge_skill.experiment evaluate --model proposer_v6a.pt --input real_test_cases.json --out real_test_model_v6a.jsonl
& $py -m tools.edge_skill.experiment native --model proposer_v6a.pt --input real_test_model_v6a.jsonl --out native_real_test_model_v6a --budget 48 --port 9581
& $py -m tools.edge_skill.review --input native_real_test_model_v6a --out review_real_test.html
```

`report.json` holds every number above plus the development runs; `freeze.json` the hashes; `PROTOCOL.md` the overnight log.
