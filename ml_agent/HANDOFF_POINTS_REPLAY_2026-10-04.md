# Next Super Speed training run: real game replay and a points critic

**Update 2026-10-05:** the operator authorized this agent to collect data and start training.
The run is now active; see [`RUN_POINTS_2026-10-05.md`](RUN_POINTS_2026-10-05.md) for dataset counts,
validation, process ownership and restart instructions. The earlier implementation-only handoff follows.

Code implemented and tested late 2026-10-04. **No main training loop was started, no trained checkpoint
was modified, and nothing was committed.** The operator wants another model to run overnight training.
This file supersedes the startup/data/reward instructions in `HANDOFF_SUPERSPEED_MANEUVER_2026-10-04.md`.
Keep its standing operator rules: no map hacks, no commits, no Horizon/Archipelago tuning, and do not
stop an authorized running training session on your own. Client/lobby work is outside this task.

## Do this first

1. Read the implementation and checks below. Training is still stopped at the old update 29665.
2. Collect new schema-2 starts using **inference only** with the protected navigator 28897.
   The existing `starts_live_*.json` and stages 1-3 cannot supply the missing world state.
3. Convert the collected JSONL files, verify both round-grouped splits are populated, then run the
   new matched-window evaluation. Keep evaluation source rounds out of the development set.
4. For the new training run, use 28897 as the input, four real replay workers and four ordinary
   round workers, all optimizing actual game points. The trainer resets the old value prediction
   and Adam state, then performs 20 critic-only updates before allowing the powerup heads to learn.
5. Check matched windows and full rounds against the protected base as well as the same model
   with use disabled. A window improvement alone is insufficient to claim a higher KOTM score.

There is **not yet a production schema-2 dataset**. One collected approach and the deliberately
constructed engine fixtures in `logs/nav/replay_*` are test artifacts, not an overnight curriculum
or a held-out performance gate. Collect several source rounds before splitting; aim for hundreds
of development starts and a useful evaluation set from separate rounds. Do not duplicate IDs to
reach an evaluation count, or copy development starts into evaluation when a split is empty.

## What was wrong and what changed

The old stage-4 replay started about 0.19 s before firing but targeted the post-fire continuation.
In the reviewed dataset, 180/201 first targets differed from the original approach target. It also
ended at the six-gem chain limit, and blocking recovery did not consume the nominal 12 s window.
Synthetic drills paid one unit per arrival while ordinary rounds paid a shaped reward. A shared
critic therefore assigned incompatible meanings to its value. Skipped restores and repeated IDs
also meant nominally equal evaluation counts did not produce equal paired samples.

The replacement has these contracts:

* `trainingReplay.cs` snapshots a local solo Hunt state: marble position, velocity and actual spin;
  held item; real gem/item visibility and item respawn timers; score, remaining/elapsed game time,
  blast state, spawn counters, last spawner, camera yaw and script RNG seed. The mission catalog
  describes real item locations and datablocks. The bridge is dormant until `REPLAYCAPTURE 1`.
* A source round records the state about **1,536 ms before a fire**, preserving its original current
  and next targets and up to 32 preceding raw observations/use-state records for GRU warm-up.
  Recording does not depend on the later manoeuvre succeeding. Stable IDs include session, round
  and game-clock time. The split hashes the source round, keeping overlapping approaches together.
* Stage 4 restores the real world and inventory. It grants no artificial item and supplies no
  synthetic gem chain. Actual game pickup volumes, point values and spawn rules decide the score.
  The same chooser used in ordinary rounds selects subsequent targets.
* A window ends after **12,032 simulation milliseconds** (188 ordinary 64 ms decisions), including
  falls and legal quick-respawn clicks. A recovery click occupies its own scored transition.
  No six-pickup stop, blocking uncounted recovery, or extra cosmetic MARK step is allowed inside it.
* Both real rounds and windows pay `gem_delta`, including multi-point gems. A round ending is a
  terminal transition; a window cut bootstraps the critic from its actual final observation and
  continuing GRU state. Discount and GAE use the measured transition duration, with the existing
  gamma 0.99 per 64 ms. There is no mean-value continuation bonus or assumed three-second fall cost.
* Checkpoints carry `reward_version=game_points_v1` and remaining critic warm-up updates. Migration
  resets the value output/PopArt statistics and optimizer, preserving actor weights. With the
  default frozen navigator, only the value head learns during warm-up. Twenty updates is an
  initial setting, not evidence of convergence: inspect point-scale values and value loss.
* The small residual can prepare a turn with Super Speed held and a target within 18 units, even
  before the current-speed firing forecast approves the kick. The firing mask, 24 u/s cap and
  existing terrain/braking checks remain in force. No map coordinates enter this policy rule.
* Stage 1-3 synthetic drills remain diagnostics. The main trainer rejects them and missing/legacy
  replay data before starting workers. `NAV_CKPT` now selects the input checkpoint; saving still
  writes `models/nav/nav_latest.pth` and numbered outputs, never the input path.

This is a **script-level replay for the currently tested static Hunt setup**, not a general engine
savegame. Multiplayer, competitive mode, pathed interiors, nonstatic pickup items, countdown/OOB
starts and active powerup effects are unsupported. Other map mechanics need a state audit before
using this as a general replay format. Restore verifies the catalog, pose, inventory, timers, score,
clock, RNG and spawn state, and reports a failure instead of substituting a different start.
The 32-observation warm-up is an approximation of source memory; it is identical across branches,
and PPO uses the same stored warm state that acting used, including the first fixture after connect.

## Collection and evaluation commands

Run Python from `ml_agent` using the installed Python 3.9. Launch each listener before its separate
game, on an unused port. These modules do not create an optimizer or write model checkpoints.

```powershell
$py = 'C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe'
& $py -u -m nav.ss_live_smoke --port 9978 --steps 60000 --ckpt models/nav/nav_v10_28897.pth
```

Start the corresponding game in another process from `Marble Blast Platinum`:

```powershell
.\marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport 9978 -offline
```

For background launches, use `Start-Process -WindowStyle Hidden`, redirect stdout/stderr and retain
the PIDs of the specific probe/game pair for cleanup. No main training launcher is needed for
collection. The existing `run_one.ps1` does not pass `-offline`; the direct launch above was tested.
Collection prints the exact JSONL path and accepted/rejected counts. Its duration is bounded by
`--steps`; collect more independent rounds/seeds if needed. Then:

```powershell
& $py -m nav.ss_drill_starts --live <collected-jsonl-paths>
```

Outputs are `datasets/ss_drill/starts_replay_dev.json` and `starts_replay_eval.json`. The converter
rejects legacy records, deduplicates IDs, rejects conflicting duplicates and preserves round splits.
Keep a fixed evaluation file for comparisons across checkpoints; do not repeatedly replace it
while selecting models. The old synthetic datasets remain untouched.

Launch a fresh listener/game pair for a matched gate:

```powershell
& $py -u -m nav.ss_window_eval --port 9978 --ckpt models/nav/<snapshot>.pth --starts datasets/ss_drill/starts_replay_eval.json --out logs/nav/<tag>.json
```

The default branches are `base,no_use,force,learned`: protected 28897 with use disabled; current
model with use disabled; current model firing at every approval; current learned use. The latter
three retain the current residual. Every branch restores every requested ID exactly once and uses
the same history. The output includes IDs, per-attempt failures, points, falls, pickup times, firing,
elapsed time, dataset hash and paired differences/standard errors over common successful IDs.
Always report requested, failed and paired counts. `--n` cannot exceed the number of unique starts.
Optional `fire_now,wait_now` branches differ only on the first decision and then follow the learned
policy. They support later counterfactual diagnostics; no separate use-head regression was added.
`ss_drill_eval --stage 4` now directs callers to this evaluator; old gate wrappers must be updated
to invoke it rather than assuming the old synthetic stage-4 format.

## Training settings for the next operator-authorized run

Do not resume 29665 blindly: widening the approach gate exposes its old residual in additional
states, and its value estimates came from mixed reward units. Start this experiment from 28897;
keep the old endpoints as comparisons. Preserve the existing stop checkpoint and logs before
launching. In the launcher's environment:

```powershell
$env:NAV_CKPT = 'models/nav/nav_v10_28897.pth'
$env:NAV_DRILL_PLAN = '0:4,1:4,2:4,3:4'
$env:NAV_DRILL4_STARTS = 'datasets/ss_drill/starts_replay_dev.json'
$env:NAV_TOUR = 'walk'
$env:NAV_REAL_GEMS = '1'
$env:NAV_LIVE_RECORD = '1'
$env:NAV_CRITIC_WARMUP_UPDATES = '20'
```

The code defaults match the four-window/four-round allocation for eight instances. Clear inherited
evaluation/no-use flags. The main launcher/restart/watchdog setup is still the existing one; this
change did not start or alter it. Ensure training game launches use the working offline recipe.
`NAV_CKPT` is the input on **every** process start: after creating a new points checkpoint, unset it
or point it at the new `nav_latest.pth` for crash recovery. Leaving it pinned to 28897 would restart
the experiment from the base. The reward version/warm-up counter then resume automatically.

Initially check that the log announces reward migration, actor weights stay fixed during critic
warm-up, `WINDOW` rows last 12032 ms and round rewards are game points. Monitor falls and ordinary
round performance as the residual starts learning. Run the matched four-arm gate periodically on
copied checkpoints, then the established full-round gates (at least 32 rounds per arm) against
the protected base and same-checkpoint no-use. The protected base is approximately 161-162 mean,
best 173 in the earlier recorded gates. No new score improvement has been demonstrated here.

## Verification performed without main training

* `python -m unittest nav.test_replay -v`: 11 passed. Covers real weighted points, no synthetic
  arrival credit/six-pickup terminal, grouped splits, duplicate/failure accounting, elapsed GAE,
  warm GRU action probabilities, approach preparation and input-checkpoint protection. A tiny
  in-memory critic-only unit update verifies actor weights are unchanged; no trained file is used.
* `nav.replay_probe`: three restores, identical trajectories, maximum error 0.000000.
* `nav.replay_probe --powerup`: collected a real Super Speed, restored its hidden item and remaining
  respawn timer, then fired it. Three 64-decision trajectories, inventory/timer records and points
  were identical; maximum pose error 0.000000.
* `nav.ss_window_eval` with identical base/no-use policies: both earned 9 actual points from seven
  pickups at identical times; both lasted exactly 12032 ms. This exposed and fixed a startup bug:
  the evaluator must wait until GO before restoring, because countdown events otherwise override
  the first trial after a successful restore acknowledgment.
* `nav.replay_worker_probe`: two identical straight-driving windows, each with falls and recovery;
  both lasted 12032 ms, summed transition durations matched, and reward matched the game score.
* Inference-only collector: 2900 decisions crossed a natural round boundary (151 points, one
  fall), recorded one eligible live approach and did not update any weights. This single sampled
  round is a plumbing check, not a score gate.
* The collected approach restored successfully in all four evaluator branches with warm history:
  base/no-use/learned each earned 10 points, forced use 11, with zero falls. This is one test case,
  not evidence of a powerup gain. Identical-policy branches had some later pickup times differ by
  one 64 ms step even though their scores agreed. The script restore does not promise bit-exact
  engine internals in arbitrary moving states. Measure repeatability on the larger collected set
  before interpreting small paired gains or producing counterfactual use labels.

Test reports and fixtures are under `logs/nav/replay_*`; the collected source record is
`logs/nav/live_starts_20261004_235652_0.jsonl`. Probe fixtures must not enter the development/eval sets.
All isolated probe/game processes were closed at handoff. The existing dashboard was left running.
`nav_latest.pth` still hashes identically to the saved `nav_r2_stop_29665.pth` endpoint.
