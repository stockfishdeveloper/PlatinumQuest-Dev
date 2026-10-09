# Model checkpoints: what is on disk and which ones matter

Checkpoints (`*.pth`) are NOT tracked in git (operator decision 2026-10-09; `*.pth` is in .gitignore). They live
only on this machine under `ml_agent/models/`. This file is the record of which file is which. Keep it current when
a new milestone is saved; the dated detail for every entry is in `docs/HANDOFF_NAV_TRAINING_LOG.md` (section numbers
below) and `HANDOFF_NAV_TRAINING.md`.

Snapshot 2026-10-09: 5,647 files, 29 GB (`models/nav` 1,918 files 17 GB; `models/checkpoints` 3,712 files 12 GB, the
old pre-navigator project; `models/learned_nav` 17 files 206 MB).

## The ones that matter now

| file | what it is |
|---|---|
| `nav/nav_latest.pth` | the resume point: the trainer saves here and resumes from here. 2026-10-09 15:00: update 37575 (end of the block-cluster discovery run, after the 2 h block of 10-09). |
| `nav/nav_points_20261005_parity_32565.pth` | KOTM base, 166.55 avg (best clean KOTM score, log 40.60-40.62). Every map fine-tunes from this line of models. |
| `nav/nav_best_20261002_tour_28897.pth` | navigator-only best before the points run: 160.1 on KOTM with the walk tour (log 40.2x). |
| `nav/nav_clusters_20261007_check1..14_<update>.pth` | the overnight check milestones of the block-cluster run (10-07 to 10-09), one per check; held-out 3 -> 111 mean over the run (log 40.64-40.70). |
| `nav/nav_points_20261005_best_<update>_<score>.pth` | the points run's 50-round highs (154-157), 10-05/06. |
| `nav/nav_points_20261005_stop_30295.pth` | Super Speed run end, 10-05 20:16 (operator stop; log 40.5x). |

## Families and naming

* `nav/nav_<update>.pth` (1,302 files) and `nav/nav_clusters_20261007_<update>.pth` (201), `nav/nav_points_20261005_<update>.pth` (147):
  the trainer's automatic saves, one every 25 updates, ~3-10 MB each. Resume safety only; none is a milestone.
* `nav/nav_night_<update>_<mean>.pth`: every new 50-round training high during the overnight runs of 09-23 to 10-04
  (`logs/nav/night_best.json` lists them). `nav_night_23705_142.pth` = `nav_best_1x_150_23705.pth` (150.6 over 78
  real rounds, the "Baseline before jump physics" rollback point, log 28.4x).
* `nav/nav_before_<change>_<time>.pth`: snapshots taken right before a code or reward change (09-19 to 09-23), for
  rollback. `nav_pre_*` the same. `nav_eval_*`: the checkpoint an evaluation batch used.
* `nav/nav_v3..v10_*.pth`: observation-version migrations (V3 09-23 ... V10 10-03): the same weights carried into a
  wider observation (`logs/nav/migrate_obs_*.py`).
* `nav/nav_p4a..p4i_*.pth`, `nav_r2_*`, `nav_run1_*`: the Super Speed runs of 10-03/10-04 (HANDOFF_SUPERSPEED_2026-10-05.md).
* `nav/nav_goal175_*.pth`: the 175-goal attempts of 10-05 (GOAL_175_2026-10-05.md).
* `nav/nav_best_*`: `nav_best_next1_18900` 126.6, `nav_best_parity1_19640` 131.5 (V2 obs), `nav_best_night_20945`,
  `nav_best_1x_150_23705` 150.6, `nav_best_20261002_tour_28897` 160.1.
* `nav/nav_stage0_flat_upd75.pth`: the very first flat-map navigator (09-17).
* `checkpoints/update_<n>.pth`, `checkpoints/best.pth`: the pre-navigator project (before 09-17), kept for history.
* `learned_nav/`: the learned-nav stage models (P0-6b, 09-26 to 10-04; memory "history-2026-09-26-to-10-04").

## How to add an entry

When a run ends or a check produces a milestone, copy `nav_latest.pth` to a named file and add one row above with the
score, the map, the date and the log section. Never rely on the file name alone for the score.
