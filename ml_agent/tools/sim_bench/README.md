# Offline closed-loop benchmark of the physics planner (2026-10-16)

The real `ContinuousController` (nav/marble_dynamics/reliability_eval.py) plays from recorded journal states against the
planner's own 16 ms engine port: each 64 ms reply is advanced with `predict_replies` + a resume of the bridge state, gems
are collected by the game's per-tick item test (`gem_demo.tick_pickup` on the forecast's 16 ms states; before
2026-10-16 09:40 the sim used the 64 ms reply box, which rewarded plans that skim gems at speed), a fall = leaving the mission InBounds volume, no new
gem group ever spawns. The 2 s search deadline runs on a virtual clock (0.85 ms per 16 ms physics step, calibrated under
pool load), so an episode is deterministic and needs no game. One-step prediction error in the journals is ~0 (p99 0.000 u),
so the sim is the game minus the timing of the real search; it re-plays recorded falls (14 of 16 fall legs).

`--native` (2026-10-16): compiled forecasts (nav/native), as the game's planner has run since v41; the virtual clock then
advances `NATIVE_STEP_US` (100 us) per physics step (measured 61-138 us per step by map under pool load, since the Python
work per forecast dominates). Screen new planner changes with `--native`.

Run from the repo root with `PYTHONPATH=ml_agent`; processes set themselves to idle priority (keep <= 3-4 beside a pool:
idle priority does not stop hyperthread sharing, and the pools' 2 s budget is wall clock).

- `sim_bench.py` one map / one variant; starts = the decision after each recorded pickup (leg starts). Selections:
  `--far-min/--far-max` (straight distance to the next pickup), `--fall-legs`, `--stuck-legs`, `--edge-gems U`,
  `--lead N` (start N decisions earlier), `--held N` (legs that start holding powerup N; 2 = Super Speed). Variants: `--module search=path` (a modified planner module, loaded in place
  of the repo one), `--set gem_demo.TARGET_SPEED_CAP=13.5`, `--pflag=--fallback-cooldown=0` (planner launch flags).
  `--trace --only-decision D` prints every decision of one episode.
- `sim_batch.sh <outdir> <parallel> <common args> -- base "name:--module,search=x.py" ...` runs all maps x variants
  (one jsonl per variant and map); `VARIANT_MAJOR=1` orders jobs by variant. Do not edit it while a batch runs (bash reads
  scripts lazily).
- `sim_pairs.py dir1,dir2 variant ... [--per-map]` paired summary (episodes done in both: geometric-mean time ratio,
  better/worse > 0.3 s, falls, cap hits); `sim_compare.py dir base variant` counts a fall as cap + 5 s.
- Audits at recorded states: `escape_audit.py` (pickup plans found in stuck stretches, base vs variant module),
  `budget_audit.py` (2 s virtual budget vs unbudgeted), `fallback_audit.py` (states where A2 took over),
  `gate_rate.py` (terminal-gate rejection shares), `why_fail.py` (every candidate's verdict at one decision).
- Journal analyses: `leg_time.py`, `leg_time2.py`, `far_route.py`, `far_speed.py`, `far_slow.py`, `turn_audit.py`.
- `variants/`: the module copies and repo patch scripts tested on 2026-10-16 (see docs/HANDOFF_NAV_TRAINING_LOG.md).
  The module copies were taken from the repo code of that night; regenerate them from the current code before reuse.

Screen every candidate on all three sets before a pool: normal legs (`--far-max 20 --pickups 3`), far legs
(`--far-min 20 --pickups 2`) and edge-gem / fall legs (`--edge-gems 3 --lead 15`, `--fall-legs --lead 15`): v39's keep-plan
looked neutral on normal legs and lost in game on far-travel maps (Bowl).

Caveats: stuck legs selected because they were long re-simulate far faster (selection on bad luck); test stuck fixes with
`escape_audit.py` instead. A long lead-in diverges from the recording (fewer recorded falls reproduce).
