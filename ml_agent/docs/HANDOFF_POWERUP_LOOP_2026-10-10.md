# Powerup practice loop — 2026-10-10

Status (17:49 MDT): operator requested stopping after current evaluations. Controller r3 is in evaluation-completion mode: finish already-running auto_006, review it, confirm A2/working best over4 rounds/map, then exit before any new training block. stop_after_evaluations=true verified in live state. A2 protected.

## Authorization and scope

The operator requested: "ok can you proceed with that plan and drive the loop yourself? occasionally report to me the progress while you continue". This authorizes the bounded practice/training/evaluation loop discussed in the prior handoff. It supersedes the earlier hold on launching training. Preserve A2, B2 and reset-start weights; use fresh run directories for candidates. Report periodically and document changes/results here.

The operator also explicitly requires logging every bug fix and medium/major action. Record the problem or purpose, concrete change, validation and limitations, and affected run/artifact here. Preserve stopped/invalidated runs and distinguish diagnostics, critic calibration, policy training, and held-out results. The review driver appends completed evaluations; failed reviews must also be documented before retrying or continuing.

## Implementation

- Added `nav/powerup_practice_eval.py`: finite real-engine matched windows for no use, handling activation during reset, deterministic learned use, and sampled learned use. The sampled branch samples only use while retaining deterministic movement, with **zero use exploration floor**; it is a diagnostic evaluation, not a training trajectory. Records engine points, falls, pickup times, estimated floor contacts/peak height, eligible use probabilities, and failures. Related starts remain grouped by source round. All windows are 12,032 simulated milliseconds after reset.
- Added `NAV_PRACTICE_MODE=handling|judgment` to stage-4 real fixture resets. Handling consumes an existing recorded Super Jump or blast through ordinary engine actions, confirms consumption, releases the key, and warms recurrent state on those reset observations. Those reset actions and points are excluded from PPO transitions and rewards, and logged separately. Unsupported/missing resources or a fall/boundary during reset fail explicitly. It teaches handling, not activation judgment.
- Added configurable `NAV_USE_EPS` (default .02 remains the training setting) and returned action-distribution use probability for diagnostics. No network shape/checkpoint conversion.
- Fixed window activation bookkeeping to include SJ/blast and delayed engine consumption. A request can precede consumption; an event is distinguished from merely requesting use. Fall clears the pending request. Initial screen's `fired` flags predate this correction and must not be used as accurate consumption counts.
- `prepare_real_maps.ps1` accepts finite `MaxWallMinutes`, practice mode and a pinned practice mission. Pinning requires exactly worker 0 at stage 4 plus an explicit pool. `run_game_loop.ps1` holds that worker on its practice map while the seven regular workers retain map rotation and crash relaunches. Checkpoints now record practice mode/plan/map/source file/exploration floor.
- `EscapeRecorder` has an opt-in `NAV_PRACTICE_CAPTURE=held` collection mode: records normally held resources during ordinary grounded play, with 32 prior observations, at most once every two game seconds. Captures approaches regardless of later success. The original escape-only behavior is unchanged by default. This collects SJ/SS/Helicopter etc. without granting inventory.
- Strict catalog checking remains enabled; mismatch errors now include lengths and first differing descriptor. Evaluator preserves both catalogs on failures.

## Checks and screening so far

- Original 59 regression tests passed after handling implementation; three new handling tests verify real-resource dispatch, exclusion of reset points/actions, and rejection of unconfirmed/unavailable inventory. Subsequent replay+handling checks: 16 passed. PowerShell launcher/supervisor ASTs parse.
- Initial development screen reproduced catalog mismatches and was stopped. Repeated strict no-use probes subsequently passed on all eight fixture maps: `logs/nav/powerup_handling_20261010/catalog_screen.json`. Root cause is not established; do not describe catalog mismatch as fixed.
- `dev_paired.json`: six development starts (three Fractured Islands, three Sprawl), four treatments each, all 24 windows completed without restoration failure. Exploratory source-round grouped handling-minus-no-use advantage .4 points, SE .245; learned-use-sampled advantage 1 point, SE .775. These tiny development results select practice conditions; they are not evidence of newly learned skill.
- All three selected Sprawl starts produced zero points in every branch. Two are the scarce SJ fixtures, one directly under a deck. They are unsuitable as an initial easy curriculum. Fractured Islands starts give prompt pickups, but some are also easy without a powerup; retain this distinction when reporting.
- Repeated no-use and deterministic trajectories can diverge slightly even with strict fixture checks; engine contact state/recurrent history is approximate, so small per-start differences are not conclusive causal estimates.

## Baseline and new held-state collection

Running `nav.phase_r --baseline --rounds 2 --instances 8 --profiles assisted --checkpoints A2` with `NAV_PRACTICE_CAPTURE=held`. Output: `logs/nav/powerup_handling_20261010/baseline_A2`; stdout/stderr: `powerup_baseline_A2.*.txt`. This is evaluation only. Every completed round must reconcile reward-derived points with final engine score. This provides a fresh full-map baseline and normally collected held-powerup fixtures with source-round dev/evaluation splits.

## Next actions

Evaluate held-out handling and learned judgment plus full-map/KOTM retention before extending or changing the mixture. Keep failures and sample counts visible; do not claim handling activation was learned.

## First pilot launched

- Fresh baseline: all 20 rounds completed and engine scores reconciled, all ten map jobs passed; A2 hash unchanged. Equal-map mean **33.1**, KOTM **124.5**. Per-map means: KOTM124.5, Sprawl25.5, Cube11, Bowl3, Duplex28.5, Fract91, Crag4.5, Vortex0, Treasure34, Concentric9. These are two-round estimates with substantial map variance, not precise rankings.
- New collection: 202 normally held-powerup starts, 122 development / 80 evaluation, grouped by source round. No inventory was granted. Includes SJ, SS, Helicopter and Mega approaches.
- First handling skill is **SJ landing on KOTM**, because the old under-deck starts provided no nearby success signal. Selected purely from development-state geometry/inventory: SJ held, target distance 3..10 u, vertical target difference <=2 u, horizontal speed <=10 u/s. This gives five development starts from KOTM round2, and four evaluation starts from round1. Narrow pilot only: one source round per split does not establish broad transfer.
- Locked pools: `baseline_A2/sj_easy_dev.json`, `baseline_A2/sj_easy_eval.json`. All five development handling windows strictly restored and completed; mean7.4 points/.2 falls. First two matched starts showed 7/8 points after reset-SJ versus 11/9 without using. The curriculum teaches recovering control after activation, not that spending here beats saving.
- Holdout reference `sj_holdout_A2.json`: all16 windows passed. Handling mean6 points/.5 falls; no-use8 points/.75 falls; deterministic learned7.75 points/.75 falls; sampled learned-use5.5 points/.25 falls. These four starts share one source round. Evaluated only after development selection was locked.
- Pilot run **real_maps_20261010_powerup_h1**, launch12:06:18 MDT, launcherPID7460. Protected A2 snapshot; 30-minute wall-time bound. Supervisor/trainer PID in `supervisor.log`; initial trainer29308. Worker0 handling on KOTM; seven workers ordinary rounds with map rotation. User can stop via `logs/nav/real_maps_20261010_powerup_h1/STOP`.
- Starting source update37760. Reward metadata change resets critic/Adam and requires20 critic-only updates before actor learning. Do not interpret that interval as newly learned skill.
- **Shared reward consistency:** discounted potential shaping k=.05 now also applies to stage-4 practice transitions, with the ledger reset after fixture restoration and no terminal zero at the artificial window boundary. This avoids sharing one value head across shaped ordinary transitions and unshaped practice bootstraps. Engine points remain separately logged; reset activation/setup points are excluded. Legacy mode retains historical points-only practice. Metadata names this `shared_discounted_potential_bootstrapped_v2`.
- Ordinary training workers also record fresh held starts at most once per8 game seconds, cap200 per map/process, under `held_starts/`. Clear recurrent capture history at round boundaries, falls/respawns and rescues. These are raw development candidates for future curricula, not automatically fed into practice.
- All **62** regression checks passed before launching. Launcher ASTs parsed. `nav.phase_r --checkpoint-file <path>` now evaluates isolated candidate files without editing shared checkpoint configuration; launcher `SourceCheckpoint` supports a fresh bounded continuation snapshot.
- WINDOW `resource` is the resource at fixture start; `fired` counts an additional confirmed use inside the window and may be blast/another item, not necessarily that starting SJ. Reset activation is separately indicated by mode/reset timing. Do not report reset activation as a learned use.

## Evaluation and continuation criteria

After the finite block, stop its owned games/workers, preserve final candidate, compare the same four held-out windows and fresh two-round/full-ten-map evaluations against A2. Inspect practice pickup/fall distribution and full-round/KOTM regression together. A short pilot can validate integration or reveal regression; it cannot certify skill. If handling improves with retention, expand development diversity before treating timing/use judgment as solved. If it degrades or remains unlearned, adjust the curriculum or continue a bounded control comparison rather than increasing a use bonus or blanket exploration floor. No concurrent baseline sweep with eight training games.

## Warm-up bug discovered; first pilot invalidated

- At12:08 MDT inspected `ppo_recurrent.ppo_update(critic_only=True)`: it removed policy/entropy loss but still backpropagated value loss through the shared encoder/GRU, changing the actor during purported critic-only warm-up. Initial log had KL .102 at update37762, consistent with drift. Existing regression used `freeze_base()`, inadvertently masking this issue.
- Created h1 STOP; supervisor ended12:08:59 and closed all owned processes. h1 is invalid for learning comparisons; retain its source snapshot/logs and do **not** resume its outputs. No full evaluation claims from it.
- Fix: before clipping/Adam stepping in critic-only updates, set every gradient outside `value_head.*` to None. This preserves the full shared actor and skips Adam momentum updates for those parameters. Value-head fitting and PopArt continue normally.
- Strengthened original preservation test to use an unfrozen network; added another test that first creates real Adam history with a normal PPO update and then verifies every actor/shared tensor stays bit-for-bit unchanged through critic-only PPO. **63 tests passed** including the repaired warm-up path.
- Added per-resource `activation_events` (confirmed consumption plus window-relative time) and `use_requests` to WINDOW results before the replacement pilot. This distinguishes subsequent blast/other-item use from the reset's starting SJ. Reset activation remains separate and never enters PPO.
- The operator reinforced: log every bug fix and medium/major change. This handoff is the running authoritative log, with links from the earlier preparation handoff.

## Replacement pilot and review automation

- h2 launched12:10:26 MDT from untouched A2, same five locked development starts and 30-minute bound. LauncherPID34032, initial trainer26720, game loop3696. Output `logs/nav/real_maps_20261010_powerup_h2`, models in its matching isolated model directory. h1 remains stopped.
- **Live warm-up evidence:** saved update37765 has15 critic-only updates remaining and **zero changed actor/shared tensors** versus source37760. Artifact `warmup_actor_check.json`. Warm-up NAV lines have KL approximately0 and clip0. This confirms the repair beyond unit tests.
- Added `review_powerup_pilot.ps1` and queued hidden review driverPID27656. It waits for this specific bounded launcher (45-minute wait cap), requires STOP and no running games, protects source SHA, freezes `review_checkpoint.pth`, runs all four held-out starts/four treatments, then two complete rounds on each of ten maps. It refuses to overwrite an existing review and retains failures.
- The review emits per-branch practice and full-map/KOTM comparison JSON under `h2/review/comparison.json`, verifies candidate hashes, and appends its completion summary here. Its90% retention flag is a screening heuristic, not a promotion or proof of learning. Continued training is not automatically triggered by that flag.
- `activation_events` distinguishes confirmed inventory consumption by resource/time; `use_requests` counts requests independently. These are active in h2. No forced reset use is logged as a learned activation.

## Real-flight control bug; h2 stopped before actor learning

- Old `SegmentManager.step` inferred a fall after four decisions outside the recorded terrain XY footprint (or below the lowest sampled floor), even with no game OOB flag. In real training this set `recovery_pending`; subsequent steps replaced policy movement/use with an OOBCLICK control and reset observation history. The engine legally refuses that click before true OOB, so it did not teleport illegally, but **it suppressed steering while a flight could still be recovered**. This matters directly to SJ/Helicopter and caused training/evaluation mismatch: ordinary `real_run` already uses the actual game flag.
- h2 STOP created12:26:10; review driver27656 canceled before it could compare invalid practice trajectories. Supervisor stopped all owned processes12:26:38. No h2 results are claimed as learning.
- Stopped saved checkpoint37775 has5 critic warm-up updates remaining and **zero actor/shared changes versus A2** (`stopped_actor_check.json`). It can serve as a critic-only warm-start: the actor remains original A2. Retain this distinction if used for h3.
- Fix: real-mode segments infer neither falls nor recovery from terrain bounds; only the engine's OOB flag triggers fall recovery. Synthetic diagnostic fall inference is preserved. Actual game OOB recovery still uses the existing legal click.
- Related bookkeeping fix: clear pending use requests before explicit OOB recovery and on accepted rescue; require a recent matching request before labeling held-slot disappearance an activation. This prevents inventory lost on respawn from being called powerup use.
- Added tests: twelve airborne steps outside the terrain grid leave real-mode steering/recovery untouched; the actual OOB flag still schedules recovery; synthetic fallback remains functional. Full targeted suite: **65 passed**.
- Evaluator now records outside-grid decisions, steering decisions outside the grid, and recovery controls sent without an OOB flag. Re-running all five development handling fixtures and all four holdout fixtures/four treatments with protected A2 under corrected dynamics: `sj_dev_after_oob_fix_A2.json`, `sj_holdout_after_oob_fix_A2.json`.
- Full ten-map A2 baseline stays valid because its `real_run` runner did not use the faulty segment fall inference. Practice references must be replaced by fresh, separately preserved artifacts. Review script now accepts `BaselinePractice` to avoid silently comparing different control dynamics.

## Corrected dynamics verified; h3 resumed

- All21 corrected A2 windows (five dev handling plus four holdout/four branches) completed with strict restoration and no failure. A sampled holdout maneuver spent14 decisions beyond the recorded grid,13 still steering; **zero recovery controls without game OOB**. The other outside-grid decision was the legitimate OOB recovery. This is real-engine evidence for the control fix, in addition to unit tests.
- Corrected holdout reference means: no-use7.75 points/.75 falls, handling6/.5, deterministic learned7.5/.75, sampled learned-use4.75/.5. Preserve the earlier reference separately; comparisons for h3 use `sj_holdout_after_oob_fix_A2.json`.
- h3 launched12:29:48 MDT, launcherPID5572, 30-minute wall bound. Source is the frozen h2 checkpoint37775 with five critic updates remaining; its actor/shared tensors are **identical to protected A2**. Retain its critic-only fitting rather than restarting that calibration. No policy update from either discarded attempt is inherited. Source SHA recorded in h3 `prepared.json`.
- Same five locked development fixtures; worker0 handling on KOTM, seven normal rotating workers; corrected engine-only OOB control. Outputs `logs/nav/real_maps_20261010_powerup_h3`, models in the matching model directory. The full A2 baseline remains33.1/KOTM124.5.
- New review driver waits for h3's finite completion and explicitly uses the corrected practice reference. Driver PID is recorded in the launch output; its stdout/stderr are h3 `review.*.txt`. It freezes the candidate and runs practice/full-map checks before any continuation decision.

## Live h3 warm-up verification

- Initial trainerPID30620, game loopPID27372, queued review driverPID32480. LauncherPID5572 remains the specific process the review waits for. Thirty-minute bound ends approximately12:59:48 MDT; evaluation follows owned-process cleanup.
- Saved update37780 has zero critic-only updates remaining and **zero changed actor/shared tensors** against A2 update37760. Evidence: `logs/nav/real_maps_20261010_powerup_h3/warmup_actor_check.json`. All five remaining warm-up NAV lines report KL approximately0 and clip0.
- Actual policy learning started at37781. Updates37781/37782 are finite (KL .049/.011), with no new trainer exception in the inspected stderr. These collection metrics establish execution health, not powerup skill; the queued held-out and full-map evaluations determine the next decision.

## h3 completed; automatic review cleanup race (13:37 MDT)

- Supervisor ended13:00:00 after its finite block. Last logged policy update37818; latest saved candidate will be frozen by review. No further training was launched.
- Queued review failed before creating its review directory: it observed a game process immediately after supervisor exit. At13:35 inspection there were no remaining game/training processes; this indicates shutdown latency, not an ongoing training overlap. Original review stdout/stderr preserved.
- Fix in review_powerup_pilot.ps1: wait up to30 seconds for game processes to finish closing, then retain the refusal if any remain. Never terminate an unrelated game for evaluation. Validate PowerShell syntax and retry the same stopped pilot in separate retry stdout/stderr; existing source/checkpoint/hash and no-overwrite guards remain.

## h3 review retry: held-out practice completed (13:37 MDT)

- Review retryPID17496; original failed driver output retained, new output in review.retry.stdout.txt/review.retry.stderr.txt. PowerShell AST validation passed after the bounded cleanup-wait repair. No games were killed for the retry.
- All16 named held-out windows passed strict restoration. Candidate handling mean6 points/.5 falls versus A2 6/.5: no measured handling improvement. No-use8.5/.25 versus7.75/.75; deterministic learned8.5/.25 versus7.5/.75; sampled learned-use5/.75 versus4.75/.5. Four related starts from one round are too few for a general learning claim.
- Deterministic candidate consumed SJ on one of four windows; A2 consumed none in that branch. Sampled use consumes resources in both policies. More activation is not itself success: sampled candidate still scores below saving on these approaches.
- Full ten-map/two-round review is running. No checkpoint promotion or next training block has occurred. Retain A2 until complete retention evidence is inspected.

## Automatic review completed 13:40:05 MDT
Run real_maps_20261010_powerup_h3, update 37815: ten-map mean 29.85 versus A2 33.1; KOTM 112.0 versus 124.5. Details: logs/nav/real_maps_20261010_powerup_h3/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


## h3 review decision (13:41 MDT)

- All ten jobs/twenty complete rounds passed, including reward-derived versus engine-score checks; checkpoint hashes unchanged. No remaining game processes after review. Frozen evaluated checkpoint37815 (the save before last logged update37818).
- Equal-map points29.85 versus A2 33.1 (-9.8%); KOTM112 versus124.5 (-10.0%). Equal-map falls21.75 versus21.15, KOTM13.5 versus7. Four practice handling starts remain6 points/.5 falls in both candidates.
- Scores A2 -> h3: KOTM124.5->112; Sprawl25.5->18.5; Cube11->17.5; Bowl3->16.5; Duplex28.5->8.5; Fract91->76.5; Crag4.5->11; Vortex0->0; Treasure34->29; Concentric9->9. Mixed map effects, not broad improvement.
- Retention screen is false because KOTM112 is just below its heuristic112.05 cutoff; this razor-thin threshold is not statistical evidence. Nevertheless, lower total points and unchanged handling offer no reason to promote h3. Retain protected A2; preserve candidate and failures for comparison.
- Next curriculum work should broaden source-round/map diversity and identify development situations where SJ/other powerups can help reach a useful landing/target. Include save cases for judgment. Current easy flat approaches teach control after activation but generally reward saving more; increasing use alone is not the goal. A matched ordinary-training control is needed before attributing full-map changes specifically to practice. No new training block was started during this status check.

## Broader curriculum preparation and development screen

- User requested proceeding with the plan after h3 review. Preserve A2, avoid promoting h3, expand source diversity and prepare an ordinary-training control.
- Assembled109 normally collected held states from h3 (81 development/28 evaluation), retaining round-level splits. KOTM states here came from rotated ordinary workers2/4/6, not practice worker0. Baseline assembled partitions remain authoritative; never repartition raw baseline rows with their unnormalized group labels.
- Added nav/prepare_powerup_curriculum.py: validates real fixtures and disjoint source rounds/IDs, hashes inputs, selects solely by geometry (3..40u target distance), prioritizes elevated/far candidates plus different source rounds/categories. Requires32 prior observations. No outcomes used to choose development or held-out candidates.
- Locked candidate files under logs/nav/powerup_curriculum_20261010: development15 starts/12 source rounds/5 maps, held-out11 starts/7 source rounds/5 maps. selection.json records input hashes and geometric annotations. Held-out outcomes are not inspected during curriculum selection.
- Extended handling reset to normally held SS(type2) and Helicopter(type5), alongside SJ/blast. Uses the existing engine use key, inventory confirmation and key release; no inventory grants or reset transitions/rewards in PPO. Unsupported types still fail. Evaluator accepts an explicit resource filter; historical default remains SJ/blast.
- New SS/Helicopter reset regression checks consumption and exclusion from rollout counters/reward. Full targeted suite66 passed.
- Started protected A2 development screen:15 starts, matched no-use/reset-activation windows, resource types1/2/5. Artifacts dev_screen_A2.json and dev_screen.stdout/stderr.txt. This selects candidate opportunities and checks fixture reliability; it is not a held-out skill result or proof that powerup usage is learned.

## Equal-update control preparation

- Trainer already supported NAV_STOP_UPDATE, but supervisor would restart a trainer that intentionally finished. Added launcher StopUpdate (recorded in prepared.json and explicitly set even when disabled); supervisor detects the exact requested finite-completion marker, creates run STOP, and cleans up instead of restarting. It also handles a completed checkpoint before ports are opened. Wall-time cap remains a second bound.
- Both changed PowerShell scripts parse. Isolated supervisor-function checks reject another update's completion and accept the requested update/create STOP; test artifacts retained under TEMP. No game or trainer needed for these checks.
- Planned ordinary control uses the same protected h2 critic-only source37775 as h3 and finishes at evaluated h3 update37815: five remaining critic-only updates, then35 actor updates, same shaping/optimizer/settings, eight ordinary map workers. This compares the prior narrow handling mixture against ordinary continuation; map exposure differs by design and one run cannot establish significance. The source critic was calibrated on the prior mixture, equally shared by both runs.

## Broader development screen result; ordinary control launched (14:17 MDT)

- All30 windows (15 development starts,12 source rounds,5 maps) passed strict restoration, including real SS/Helicopter activation. Source-round grouped activation-minus-saving points -1.292, descriptive SE .328; falls difference0, SE .282. No selected fixture scored more after activation than saving within the named window. This is development screening with A2, not evidence of learned skill or lack of possible utility in the game.
- Some SS states yielded early setup pickups (e.g.Cube reset2 points); setup points remain excluded from the learning window and separately recorded. Comparing window points alone therefore does not measure total causal game benefit. Snapshot/contact-history approximations and deterministic movement also limit causal interpretation.
- Preserve the entire failed/easy mix and do not label these candidates beneficial merely from geometry. Positive vertical offset does not establish powerup necessity; no held-out screening outcomes have been used to tune selection.
- Launched real_maps_20261010_powerup_control1: eight ordinary rotating workers, no practice, same protected source37775 as h3, target37815 (35 actor updates plus5 critic updates),35-minute wall cap. The source and final evaluation checkpoints are protected/frozen. Run STOP remains available. Review is queued after owned-process cleanup, using the same four SJ holdouts and fresh20-round/ten-map sweep as h3. Inspect review comparison before further policy promotion or practice training.

## Control review and curriculum limitations

- Actual control launcherPID3816 started14:16:14 MDT; trainer32824/game loop18992. First update37776 completed14:17:41 with critic-only KL approximately0. Ordinary NAV segment statistics stay0 because those are segment/drill metrics; GAME round logs and held-out evaluation are the ordinary performance measures.
- Extended review driver with optional CompareRun: outputs control-versus-h3 equal-source/final-update checks and score deltas, automatically appending the comparison to this handoff. AST validation passed. Replaced only the waiting review process (29300) with paired reviewPID25996; training was not restarted. Original waiting output preserved; new files review.paired.stdout/stderr.txt.
- Static routing review: RoutePlanner uses a ground-surface graph and does not add inventory-specific flight/boost edges. The policy also observes the gem separately and can choose its own movement, so ground routing is not a hard action restriction. Whether its intermediate waypoint harms flight guidance requires trajectory evidence; no navigation change was made on speculation.
- Among all development SJ/Helicopter recordings, geometric search for an active real gem2..18u horizontally and1.5..8u higher found only one source state (SprawlSJ, already in the screen). Existing recordings poorly cover nearby elevated opportunities. Do not manufacture inventory/spawns or relabel flat cases as beneficial.
- Curriculum preparation now optionally assembles fresh ordinary JSONL through --recordings while retaining source-round partitions. This will make the control's new collection available for the next selection. Held-out pool stays separate; no new powerup-training mixture has been launched without first checking useful conditions.

## Queued continuation and current execution health (14:20 MDT)

- Added complete_powerup_control.ps1; AST validation passed. Follow-upPID29728 waits for paired reviewPID25996 (55-minute cap). It requires completed comparison artifacts with matching source/update and no running games before proceeding. Failures are appended to this handoff and retained in followup.stderr.txt.
- After review it assembles the control's ordinary held-state recordings, locks new development/holdout candidates (baseline plus fresh control), then screens development only with protected A2, typesSJ/SS/Helicopter, no-use versus reset activation. Input hashes/source rounds are preserved. It does not inspect new holdout outcomes or automatically promote weights/start another practice mixture. This extends fixture collection without changing game rules or rewarding activation.
- Latest inspected control update37778 is still critic-only, KL approximately0/clip0, no new launcher or trainer exception. Actor training follows after37780. Exact-update stop and35-minute wall limit apply; if the wall cap wins before37815, follow-up refuses to call the control matched.
- Code validation so far:66 regression checks passed, all changed PowerShell scripts parse, isolated finite-stop detection checks passed;30 real-engine development windows all strictly restored. The finite-control supervisor path is additionally exercised by this live run; its eventual stop outcome must be logged, not assumed.

## Automatic review completed 14:52:15 MDT
Run real_maps_20261010_powerup_control1, update 37815: ten-map mean 40.4 versus A2 33.1; KOTM 125.5 versus 124.5. Details: logs/nav/real_maps_20261010_powerup_control1/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.

Compared with real_maps_20261010_powerup_h3: same source=True, same final update=True, mean points delta=10.55, KOTM delta=13.5. No automatic promotion; both learning and retention evidence must be assessed.


## Fresh control curriculum locked 14:52:16 MDT
Development 19 starts / 17 source rounds; holdout 10 starts / 6 rounds. Geometry-only selection, holdout outcomes uninspected. Artifacts: logs/nav/real_maps_20261010_powerup_control1/review/next_curriculum.

Fresh development screening completed: 38 windows, 0 failures. Preserved paired outcomes in next_curriculum/dev_screen_A2.json. No policy promotion or additional practice training; inspect useful cases and control retention before choosing the next mixture.


- Autonomous loop [2026-10-10 14:56:15] Working continuation source selected: C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; metrics {'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. A2 remains protected; screening is provisional.

- Autonomous loop [2026-10-10 14:56:16] Block1: ordinary real maps, seed101, 35 actor updates; source37815, target37850. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 14:56:16] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 14:56:16] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_001\launcher.stdout/stderr.txt

- Autonomous loop [2026-10-10 14:56:18] Loop error retained: bounded training exited 1; inspect C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_001\launcher.stderr.txt. No automatic retry of an invalid experiment.

## Continuous-loop authorization, implementation and first launcher failure

- User explicitly requested keeping the loop going until they say stop. Added nav/powerup_autoloop.py, with persistent phase/source/best/PID state and events appended to this handoff. Each experiment is bounded by35 actor updates (70 after3 non-improving candidates) and45 minutes; evaluations/screens run between experiments, never alongside eight training games. Per-phase timeouts stop owned descendants on failure. A loop STOP file cooperatively stops its active training run; phase cancellation only targets its own subprocess tree.
- Provisional continuation selection: control1 scored40.4/KOTM125.5/falls17.5 versus A2 33.1/124.5/21.15 and h3 29.85/112/21.75. Same initial source and evaluated update confirmed. Keep A2 protected; use frozen control1 as working source. Gains are broad but small-sample validation, not proof of powerup skill.
- Controller continues ordinary learning and collects states; every other iteration can select judgment practice only if development probes supply>=3 successful beneficial-use candidates from3 rounds,>=3 save cases, and>=2 held-out rounds on the same map/resource. Uses actual engine points including separately reported reset setup points for exploratory selection, requires positive window pickup signal, and preserves world/inventory. These are tentative opportunities, not causal necessities. Judgment training has no forced activation, activation bonus, or increased use floor. Practice promotion also checks sampled-use points/falls against its locked reference.
- Provisional best screen requires>=2% mean-score gain, KOTM>=95% of A2 and falls<=120% of A2. Modest ordinary fluctuations may continue from the candidate; practice regressions return to best. Repeated evaluation maps are validation maps, not an untouched generalization test.
- Added NAV_SEED / launcher Seed (default1 unchanged) so repeated experiments vary on-policy sampling rather than replaying the same random stream. Game/evaluation rules and network architecture unchanged. Tests69 passed, changed PowerShell syntax parsed.
- Initial controllerPID27188 / auto_001 failed before any trainer/game was started: PowerShell parameter Seed collided case-insensitively with existing local snapshot variable seed. Preserve its logs; no checkpoint learned from that attempt. Fix: renamed snapshot variable startingSnapshot. Added explicit resume-state support to retain controller best/source/iteration after failures without overwriting run evidence. Replacement gets a fresh controller directory and auto_002 run.

- Autonomous loop [2026-10-10 14:57:32] Working continuation source selected: C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; best metrics {'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. A2 remains protected; screening is provisional.

- Autonomous loop [2026-10-10 14:57:33] Block2: ordinary real maps, seed102, 35 actor updates; source37815, target37850. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 14:57:33] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 14:57:33] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_002\launcher.stdout/stderr.txt

## Controller recovery and live continuation verified

- Replacement controllerPID31604, state/events under logs/nav/powerup_autoloop_20261010_r1; stdout/stderr in sibling powerup_autoloop_20261010_r1.*.txt. Resumed best/source/iteration from failed controller state; auto_001 remains preserved, auto_002 is fresh. The operator's stop goes to logs/nav/powerup_autoloop_20261010_r1/STOP; controller forwards it to the active run and stops further work. Ordinary block auto_002 started14:57:33 MDT; launcher26336, trainer14516, game loop5328. Source hash C80867F93D660E440FD9B8E5D48191A3B4F829B3C392A4D0DDC17C4C004D716E, seed102, target37850,45-minute cap.
- Real policy update37816 completed14:58:53, KL .009/clip .10, finite PPO metrics and no new trainer/controller exception. This confirms the repaired snapshot/seed launcher is actually running, not just syntactically valid.
- Fresh control development screen:38 windows/19 starts/17 rounds, all restored. Grouped activation-minus-saving points -2.235 (descriptive SE .881), falls +.353 (SE .191). No held-out outcomes used to tune selection. Continue ordinary progress rather than asserting these cases establish powerup utility.
- Candidate-cap logic issue found before any judgment trial: historical two candidates per map/resource cannot supply the required three-use plus three-save pool. New preparation default is six per map/resource; historical locked pools are unchanged. Added integration regression showing six selected candidates can satisfy the gate and two cannot. Future subprocess preparation loads the corrected default without restarting current training.
- Operator-STOP propagation and resume-state preservation regressions passed. Full suite71 passed before replacement launch; the additional cap-selection regression passed afterward (six controller decision checks total;72 checks across the complete suite). Review now supports explicit resource types and an already-completed launcher sentinel0, avoiding waiting on Windows idle PID0. Historical default SJ/blast remains.
- This controller continues with experiments/evaluation autonomously; it does not publish/overwrite protected baseline policies. Watch state.json for active phase and current run. Any failed validity check stops further blocks and logs the error for inspection, instead of calling an invalid run progress.

- Autonomous loop [2026-10-10 15:22:45] practice and full-map review started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_002\review_driver.stdout/stderr.txt

## Automatic review completed 15:26:37 MDT
Run real_maps_20261010_auto_002, update 37850: ten-map mean 32.85 versus A2 33.1; KOTM 107.5 versus 124.5. Details: logs/nav/real_maps_20261010_auto_002/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


- Autonomous loop [2026-10-10 15:26:39] Review saved update37850, requested target37850; wall cap may stop an experiment early.

- Autonomous loop [2026-10-10 15:26:39] Candidate metrics{'points': 32.85, 'falls': 20.55, 'kotm': 107.5}; practice sampling retained=True; best retained{'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. Next source C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; no promotion. Modest ordinary fluctuations may continue; practice regressions return to best.

- Autonomous loop [2026-10-10 15:26:39] fresh curriculum preparation started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_002\curriculum_prepare.stdout/stderr.txt

- Autonomous loop [2026-10-10 15:26:41] development opportunity screen started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_002\curriculum_screen.stdout/stderr.txt

- Autonomous loop [2026-10-10 15:30:32] Block3: ordinary real maps, seed103, 35 actor updates; source37815, target37850. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 15:30:32] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 15:30:32] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_003\launcher.stdout/stderr.txt

- Autonomous loop [2026-10-10 15:50:45] practice and full-map review started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_003\review_driver.stdout/stderr.txt

## Automatic review completed 15:55:24 MDT
Run real_maps_20261010_auto_003, update 37850: ten-map mean 29.9 versus A2 33.1; KOTM 125.0 versus 124.5. Details: logs/nav/real_maps_20261010_auto_003/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


- Autonomous loop [2026-10-10 15:55:24] Review saved update37850, requested target37850; wall cap may stop an experiment early.

- Autonomous loop [2026-10-10 15:55:24] Candidate metrics{'points': 29.9, 'falls': 22.8, 'kotm': 125.0}; practice sampling retained=True; best retained{'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. Next source C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; no promotion. Modest ordinary fluctuations may continue; practice regressions return to best.

- Autonomous loop [2026-10-10 15:55:24] fresh curriculum preparation started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_003\curriculum_prepare.stdout/stderr.txt

- Autonomous loop [2026-10-10 15:55:26] development opportunity screen started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_003\curriculum_screen.stdout/stderr.txt

- Autonomous loop [2026-10-10 15:58:55] Block4: ordinary real maps, seed104, 35 actor updates; source37815, target37850. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 15:58:55] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 15:58:55] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_004\launcher.stdout/stderr.txt

- Autonomous loop [2026-10-10 16:20:39] practice and full-map review started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_004\review_driver.stdout/stderr.txt

## Automatic review completed 16:24:28 MDT
Run real_maps_20261010_auto_004, update 37850: ten-map mean 33.25 versus A2 33.1; KOTM 125.5 versus 124.5. Details: logs/nav/real_maps_20261010_auto_004/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


- Autonomous loop [2026-10-10 16:24:30] Review saved update37850, requested target37850; wall cap may stop an experiment early.

- Autonomous loop [2026-10-10 16:24:30] Candidate metrics{'points': 33.25, 'falls': 22.05, 'kotm': 125.5}; practice sampling retained=True; best retained{'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. Next source C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; no promotion. Modest ordinary fluctuations may continue; practice regressions return to best.

- Autonomous loop [2026-10-10 16:24:30] fresh curriculum preparation started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_004\curriculum_prepare.stdout/stderr.txt

- Autonomous loop [2026-10-10 16:24:32] development opportunity screen started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_004\curriculum_screen.stdout/stderr.txt

- Autonomous loop [2026-10-10 16:27:28] Block5: ordinary real maps, seed105, 70 actor updates; source37815, target37885. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 16:27:28] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 16:27:28] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_005\launcher.stdout/stderr.txt

- Autonomous loop [2026-10-10 17:08:13] practice and full-map review started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_005\review_driver.stdout/stderr.txt

## Automatic review completed 17:12:05 MDT
Run real_maps_20261010_auto_005, update 37885: ten-map mean 33 versus A2 33.1; KOTM 126.0 versus 124.5. Details: logs/nav/real_maps_20261010_auto_005/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


- Autonomous loop [2026-10-10 17:12:06] Review saved update37885, requested target37885; wall cap may stop an experiment early.

- Autonomous loop [2026-10-10 17:12:06] Candidate metrics{'points': 33.0, 'falls': 20.4, 'kotm': 126.0}; practice sampling retained=True; best retained{'points': 40.4, 'falls': 17.5, 'kotm': 125.5}. Next source C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_powerup_control1\review_checkpoint.pth; no promotion. Modest ordinary fluctuations may continue; practice regressions return to best.

- Autonomous loop [2026-10-10 17:12:06] fresh curriculum preparation started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_005\curriculum_prepare.stdout/stderr.txt

- Autonomous loop [2026-10-10 17:12:08] development opportunity screen started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_005\curriculum_screen.stdout/stderr.txt

- Autonomous loop [2026-10-10 17:16:33] Block6: ordinary real maps, seed106, 70 actor updates; source37815, target37885. No activation bonus or exploration-floor increase.

- Autonomous loop [2026-10-10 17:16:33] No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.

- Autonomous loop [2026-10-10 17:16:33] bounded training started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_006\launcher.stdout/stderr.txt

- Autonomous loop [2026-10-10 17:43:18] Adopted active ordinary run real_maps_20261010_auto_006, launcher7176; training uninterrupted.

## Plateau audit and controller recalibration (17:43 MDT)

- User asked how things are going; earlier authorization to keep trying until stop remains active. All completed automatic trials since control1 were ordinary training, not judgment practice. Results (ten-map points / KOTM / handling-window points / sampled-use window points): auto_002 32.85/107.5/5/5; auto_003 29.9/125/6/4; auto_004 33.25/125.5/5.75/5; auto_005 33/126/5.25/4.25. None surpass control1's two-round40.4. Report this as a plateau, not continued skill gain.
- Repeated two-round comparisons to an unusually high initial40.4 anchor reset every completed candidate to source37815. This prevented accumulation across these trials and may select a noisy peak. It does not prove40.4 was noise or that continued training would solve it; confirm both the anchor and A2 before further selection.
- Latest development pool:50 starts/31 source rounds,27 held-out candidates/16 rounds;100 development windows, no fixture failures. Useful-use cases are isolated across CubeSS/CubeHelicopter/DuplexHelicopter/SprawlSJ, and some map/resource holdout counts are0..1 rounds. The judgment gate has not qualified. Do not claim powerup practice trained or powerup timing improved.
- Controller change: four-round/ten-map confirmation of A2 and the working best after current training review; recalibrate score/fall/KOTM anchor from those separate preserved artifacts. A retained current ordinary candidate can continue cumulatively without being called best. Future apparent best improvements also require four-round confirmation before selection. Confirmed reference metrics are preserved across restart and actually used by the decision loop (previous local reference always reloaded the historic two-round baseline).
- Added active ordinary-run adoption: wait for the specific existing launcher, propagate operator STOP to it, then review without retraining/overwriting its evidence. Replaced only controller31604 with controller29968; verified launcher7176's parent/command before replacement. Current trainer28456 and game loop31912 were left running. auto_006 update37859 completed17:43:08, finite KL .013/clip .10; original attempt remains1.
- New controller state/events: logs/nav/powerup_autoloop_20261010_r2; stdout/stderr in sibling powerup_autoloop_20261010_r2.*.txt. Active STOP path is now powerup_autoloop_20261010_r2/STOP. It forwards to auto_006/STOP while waiting; later phases/blocks stay owned and sequential. r1 logs remain historical.
- Full72 regression checks passed and controller compiled before replacement. Live r2 state confirms adopted training still running, with no new controller exception. Four-round confirmations are queued, not yet results. No reward, action, exploration-floor, route or game-rule change in this adjustment.

- Autonomous loop [2026-10-10 17:48:51] Adopted active ordinary run real_maps_20261010_auto_006, launcher7176; training uninterrupted.

## Operator requested stop after current evaluations (17:49 MDT)

- User: "when you're done with the current evaluations go ahead and stop the loop". This supersedes permission to continue indefinitely. The current bounded auto_006 block is still running; finish its already-queued review and A2/working-best four-round confirmations, then stop. No further training block is authorized by the previous continuous-loop request.
- Added --stop-after-evaluations to controller: persists the pending stop in live state, completes only the adopted review/confirmation sequence, writes loop STOP and final stopped state, logs completion here, and returns before the next training iteration. If evaluation fails, existing error handling also prevents another block. Immediate STOP remains available if the operator changes their mind.
- Replaced only controller29968 with evaluation-completion controller9428 under logs/nav/powerup_autoloop_20261010_r3. Resume state retained. Existing launcher7176/trainer28456/game loop31912 are unchanged. r3 live state confirms stop_after_evaluations=true and adopted auto_006; no auto_007 directory exists.
- Controller decision tests6 passed; module compiled after adding the completion mode. Live mode/ownership verification completed with no new controller exception. Future final results and the stopped state are appended automatically by the controller; do not claim it is already stopped at17:49.

- Autonomous loop [2026-10-10 17:58:46] adopted ordinary block review started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\real_maps_20261010_auto_006\review_driver.stdout/stderr.txt

## Automatic review completed 18:02:55 MDT
Run real_maps_20261010_auto_006, update 37885: ten-map mean 27.3 versus A2 33.1; KOTM 113.0 versus 124.5. Details: logs/nav/real_maps_20261010_auto_006/review/comparison.json. This is a screening result; continuation requires inspecting the retained map/window evidence.


- Autonomous loop [2026-10-10 18:02:57] four-round confirmation reference_A2_four_rounds started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\powerup_autoloop_20261010_r3\reference_A2_four_rounds.stdout/stderr.txt

- Autonomous loop [2026-10-10 18:09:39] Four-round confirmation reference_A2_four_rounds: {'points': 34.4, 'falls': 22.3, 'kotm': 122.0}; forty valid rounds required by runner. Reused maps remain validation.

- Autonomous loop [2026-10-10 18:09:39] four-round confirmation working_best_four_rounds started; output C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\nav\powerup_autoloop_20261010_r3\working_best_four_rounds.stdout/stderr.txt

- Autonomous loop [2026-10-10 18:16:59] Four-round confirmation working_best_four_rounds: {'points': 32.25, 'falls': 22.25, 'kotm': 121.5}; forty valid rounds required by runner. Reused maps remain validation.

- Autonomous loop [2026-10-10 18:16:59] Recalibrated continuation anchor: {'points': 34.4, 'falls': 22.3, 'kotm': 122.0}, C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\nav\real_maps_20261010_A2\nav_latest.pth; historic two-round40.4 is no longer the automatic reset target.

- Autonomous loop [2026-10-10 18:16:59] Requested evaluations complete; loop stopped before any next training block. All checkpoints and evaluation artifacts retained.
