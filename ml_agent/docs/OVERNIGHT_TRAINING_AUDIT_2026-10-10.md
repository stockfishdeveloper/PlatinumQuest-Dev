# Overnight real-map training audit, 2026-10-10

Later authorized repairs and the practice proposal are recorded in the
[follow-up handoff](HANDOFF_POWERUP_PRACTICE_PREPARATION_2026-10-10.md).
The findings below describe the code and runs at the time of this audit.

This is an independent review of checkpoint contents, trainer logs, evaluation
JSON/CSV streams, launch manifests, and the current code. No training, optimizer
update, checkpoint write, or policy change was performed. One finite hidden game
probe tested replay restoration; that game was closed. The dashboard remains up.
The A2/B2 STOP files remain present. The configuration's `training_approved` is
currently true from the overnight authorization; it is not the Oct 9 preparation
state described in the earlier handoff.

## Verified results and limits of the comparison

Both reruns start from the same reset checkpoint SHA-256:
`fa7c0649685430b2e7975a5be129f1f72e8616fa86c6711b2d36b9d123970776`.
Its final use layer has bias -3 and fresh small weights; the earlier layer and
other model weights remain. Adam state was retained, including the use layer's
step counter, rather than fully reinitialized. The retained first moments are
very small; this audit does not establish a harmful optimizer effect.

| Measurement | A2: route shaping | B2: game points only |
|---|---:|---:|
| Final update | 37760 | 37710 |
| Updates beyond 37575 | 185 | 135 |
| Added stored steps | 1,516,916 | 1,106,936 |
| Logged training GAME records (including partial rounds) | 398 | 290 |
| Ten-map mean evaluation points | 32.45 | 29.45 |
| Mean excluding KOTM | 21.78 | 21.89 |
| KOTM points | 128.50 | 97.50 |
| KOTM falls/round | 7.00 | 17.00 |
| Held-powerup firings across 40 evaluation rounds | 15 | 245 |

Each evaluation completed 4 rounds on each of 10 maps, assisted/noncompetitive
with rescue, deterministic actions, and no forced jumps. All 80 rounds completed
and points match the final engine scores. Current hashes still match each sweep's
protected checkpoint hashes:

* A2: `ae88f2e70a376fda06be6aed59e8bd17fd6b7b0a876bdde9c40838eadddbfb04`
* B2: `cdfd3e98194b92949faa4ac6fe4ff102f36c5310ac36a4f0900593e5cfde094a`

| Map | A2 points | B2 points |
|---|---:|---:|
| KOTM | 128.50 | 97.50 |
| Sprawl | 18.00 | 19.00 |
| Cube Isle | 19.50 | 30.25 |
| Bowl | 4.00 | 12.25 |
| Duplex | 29.75 | 27.75 |
| Fractured Islands | 71.00 | 62.50 |
| Cragmire | 4.00 | 6.75 |
| Vortex Effect | 0.00 | 0.00 |
| Treasure Box | 44.50 | 32.00 |
| Concentric | 5.25 | 6.50 |

A2 wins four maps, B2 five, with one tie. The ten-map difference is entirely
accounted for by KOTM's 31-point difference. A2 is the better current candidate
when retention matters, but these results do not establish generally better
learning on the other nine maps. Four rounds are noisy: A2's Cube Isle standard
error is 11.03 points, Duplex 14.76, and Fractured Islands 13.64. Raw-point averages
are not normalized for map difficulty, gem values, or round duration. These were
training maps and assisted settings, not held-out or competitive-lobby evaluation.

The comparison has equal wall time but unequal optimization and environment
exposure, so it answers which result the two-hour runs produced, not an isolated
effect of reward shaping. The earlier A/B is less comparable because internal
restarts repeatedly loaded the initial checkpoint. Current resume code and the
rerun logs confirm resumption from each run's own latest checkpoint.

The 30.7 'start' ten-map mean in log 40.85 comes from the earlier gated-use cluster
baseline, not a complete ten-map evaluation of the reset/open-use start. Reset
start KOTM does have a separate 32-round check, about 128.6. A2's 128.5 is consistent
with that result; claiming learning gains from 30.7 to 32.45 needs a matched reset
start baseline on all ten maps.

## Reward accounting is not policy-invariant

`vec_worker.SHAPING_GAMMA` is 1 while PPO uses 0.99 raised to transition duration.
Gem changes suppress the entire shaping term and reset the potential. The
potential freezes when no floor is within 3 u of the marble or a route is
undefined. Thus the implemented shaping is an empirical route-progress reward;
the policy-invariance claims in its docstrings/tests are inaccurate.

An explicit counterexample: +1.5 for approach followed by -1.5 for reversal 156
decisions later sums to zero without discount, but contributes +1.187 under PPO's
discount. An equally long idle sequence contributes zero. At a gem change from
potential -20 to -5, the code gives zero whereas the exact discounted formula
with k=.05 gives .7525. These demonstrate missing invariance, not evidence that
the overnight model actually exploited either case.

The positive idle stream with a negative potential under the exact formula is
not, by itself, a reason to replace its gamma: the guarantee concerns the complete
discounted return, including endpoints and the corresponding value shift.
Reference: [Ng, Harada and Russell](https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/).

The current term is also substantial on hard maps: A2's logged training means are
30.88 shaping versus 13.18 game points on Bowl, and 6.13 versus .41 on Vortex.
Its scale is small per unit but not small per round. Flight feedback remains
deferred until the marble is close to floor; it does not supply continuous
Helicopter-flight progress feedback. Both reward settings still identify saves
as `game_points_v1`; shaping settings need separate experiment metadata if
resumption and critic warm-up are to distinguish reward changes reliably.

## Powerup evidence and the exploration question

A2's deterministic evaluation fires 6 Super Jumps, 8 Super Speeds, and 1 Mega
Marble, all on KOTM. CSVs also show 7 use commands while holding nothing. B2 fires
124 Super Jumps, 98 Super Speeds, 12 Mega Marbles, and 11 Helicopters, plus 991
commands while holding nothing. An empty-held command is a blast request, not
proof a charged blast actually fired.

Training has regular trials under both rewards. A2 logs roughly 9.6 Super Jump
requests per Fractured Islands round, 2.35 Helicopter requests per Cragmire round,
and 4-8 actual detected blasts per round depending on map. The near-zero
deterministic preference should not be described as no exploration in training.
Holding-time measurements are censored by falls, rescues, replacements, and round
end. For example, Super Speed inventory segments average 13.49 s in A2 versus
5.22 in B2; these are not uncensored pickup-to-use delays.

The use floor is .02 per eligible 64 ms decision, not per collected powerup.
Even at zero learned preference that gives a mean sampled wait of 3.2 s over
continuously eligible observations; .10 would reduce it to .64 s. In the current
mixture, increasing epsilon can also weaken the gradient of sampled uses: at
learned p=.001, the use log-probability derivative is .0467 for epsilon=.02 versus
.00891 for epsilon=.10. Raising the floor is not a guaranteed way to make the
deterministic policy learn to use. Recorded 'powerup_escapes' only mean subsequent
progress within a window, not proven causation or skilled landing.

Both training runs also used 3% off-policy forced stall jumps, restored by the
operator after the original preparation. Keep those separate from learned skill
in any future report; do not silently change that authorized setting.

## Replay fixtures and remaining reconnect failures

The development escape set has 165 starts: 162 blast and 3 Super Jump, with no
Helicopter starts. Across all eight represented maps, the stored catalog has one
variant and exactly matches a sampled recent training recording, including item
order. The broad claim that the level catalogs changed is not supported by these
comparisons.

The actual-worker diagnostic used stage 4 on worker 1, Sprawl, with the same saved
dataset. Initial drill setup passed and all 14 Sprawl starts restored with the
existing strict pose, inventory/items, RNG, clock, and score checks. Evidence is
embedded in the companion audit JSON, with source script/output at
`%TEMP%/pq_checks/audit_replay_20261010.py` and `.json`. This does not certify
rotation, every map, recurrent warm-up, or multi-step policy trajectories. The
original intermittent mismatch remains unexplained; capture the actual received
and expected catalog plus mission/connection state on failure. Do not remove
validation or assume recollection is necessary.

The remaining rerun failures are identifiable in unchanged current code:

* A2: `_record_approach_at` dereferences current `env.replay_world['clock_ms']`
  after checking only the historical row. Reconnection can clear the current
  world to None. Evidence: the stderr snapshot saved under the B2 launch directory
  and A2's 02:33 resumed log.
* B2: rescue/countdown code accesses `env.info['round_ms']` after a reconnect
  cleared info. Evidence: current `logs/nav/stderr.txt` and B2's 04:51 resumed log.

Each rerun has two timestamped trainer logs: initial launch and one internal
resume. Logs support one mid-run restart per rerun, not the three/four counts in
the narrative. End-of-supervisor shutdown can also emit ConnectionResetError and
must not be counted as an additional learning restart. Supervisor snapshots are
copies of the shared stdout/stderr made BEFORE the new launch, so A2's directory
contains the predecessor's output and B2's contains A2's. They are not reliably
named evidence for the run whose directory holds them.

## Recommendation before another training block

1. Preserve A2/37760 as the current candidate and retain B2/reset as controls.
   Fix the two reconnect paths and make failure snapshots refer to their actual
   run. Verify stable rotation with finite diagnostics before longer training.
2. Choose the reward explicitly. For a policy-invariance requirement, implement
   the actual duration-discounted formula with consistent target/terminal/history
   accounting and trajectory-level tests. Otherwise label the existing term as
   intentional shaping and measure its tradeoffs. Do not claim unchanged objective.
3. Prioritize practice quality over a global use-floor increase. Use a minority
   of workers (initially one or two of eight) for normally reachable held-powerup
   starts with real landing/pickup opportunities and enough time for the full
   maneuver. Existing starts can help blast/SJ; collect separate pre-use
   Helicopter and Super Speed starts. Retain ordinary rounds and round-level
   development/evaluation separation. Active-flight snapshots currently reject
   scheduled powerup effects, so start before activation rather than inventing
   restorable flight state. Keep actual use permitted everywhere, as requested.
4. Test whether using beats saving on paired practice states, then check learned
   timing, successful landings/pickups, falls, and game points. Compare matched
   update counts and map exposure, alongside wall-clock throughput. Use several
   maps and the reset start as controls; recheck competitive/no-rescue behavior
   separately. This audit did not launch any new training.

Full computed metrics and the probe evidence:
[OVERNIGHT_TRAINING_AUDIT_2026-10-10.json](OVERNIGHT_TRAINING_AUDIT_2026-10-10.json).
Source sweeps: `%TEMP%/pq_checks/phase_r_20261010_042614/summary.json` and
`phase_r_20261010_063939/summary.json`. Reproducible offline export script:
`%TEMP%/pq_checks/export_overnight_audit_20261010.py`.
