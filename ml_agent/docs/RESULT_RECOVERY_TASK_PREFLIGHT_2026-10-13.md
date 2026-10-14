# Task-preserving recovery preflight and archived actor continuation

Two engineering rounds completed from natural starts, seed781016, with the frozen A2 navigator and the analytical full-map controller. Both opt-in corrections were enabled: preserve the actor's observed gem/resource task while physical recovery brakes, and retain the broad-search cooldown when a new recovery is selected. This is not a paired comparison or a complete ten-map evaluation. KOTM has no special gate.

| Map | Native points | Falls | Accepted / attempted rescues | Wall seconds | Nominal time executing recovery | Pickups while executing recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cube Isle | 47 | 0 | 4 / 8 | 346.076 | 81.600 s | 0 |
| Bowl | 9 | 0 | 5 / 9 | 699.414 | 253.056 s | 0 |

The controller still spends too much time safely stopping without collecting gems. Zero falls alone does not establish useful hunting behavior. Preserve the earlier failed full-pool attempt and its unfavorable scores; different seeds and controller revisions cannot be pooled into an improvement claim.

Independent audits pass all7,809checked decisions with zero pending-reply, reconstructed actor-output or missing playable-goal mismatches. A separate audit compares675selected-action128ms prefixes against actual native transitions with no movement/camera mismatch and maximum position error2.04e-8u. Only the initial native snapshot enters prediction; later native states are comparison targets. This verifies the input queue and short prefix, not a complete40-reply recovery, full-round safety or game-rule bounds.

Before follow-up source edits, all170archived source members,172runtime fingerprints and protected A2/native/bridge files were verified unchanged. The two full rounds are complete; the ten-map pool is not. Canonical compressed native/decision recordings and original source archive remain at [preflight records](../dynamics_reports/checked_recovery_task_cooldown_preflight_20261013/summary.json). The [interactive gallery](../collision_maps/viewer/checked_recovery_task_cooldown_preflight_20261013/index.html) shows actual marble, physical stop target, real task and actor input goal separately.

## Follow-up bugfix

Six Cube and37Bowl decisions retained a stale physical goal after a completed recovery cleared its state and the remaining search hit its deadline. The outer exception path did not receive the newly selected gem. The fix retains the entry recovery context and restores an actually observed task if that boundary is crossed. Both checked-backup and unchecked-fallback regression outcomes pass, along with86relevant contract tests. Historical recordings are unchanged. One Bowl cooldown reset follows a failed landing revalidation and is legitimate; the future log marker now distinguishes actual retained-plan revalidation from selecting another backup.

## Longer actor continuation: unsuccessful offline result

The bounded development probe selected every first successful recorded recovery per native body/respawn life: five Cube and six Bowl contexts. Actual actor history was independently reconstructed: all7,935recorded actor outputs match, with no untrusted history. Hypothetical later replies advance a clone only; the real first reply is reused to avoid updating the recurrent actor twice.

The variant proposes eight actor replies followed by braking, keeping the same40total replies and0.2second budget. It requires the existing complete physical support/ambiguity/stopping checks plus greater than0.3u task progress or a possible pickup under observed native item bounds. Execution completion would wait until the prefix plus delayed inputs passed. Sixty relevant tests pass, including first-reply reuse, clone isolation, task-progress refusal and delayed completion.

**No extended candidate qualifies in any of the11contexts.** Four variants return the original actor1/brake backup; seven exhaust the budget without a selection that the baseline finds. Of the eight extended candidates whose evaluation reaches an audited rejection, two have ambiguous contacts, two fail the stopping-speed limit, three lack supported continuation and one moves away from the task. Three further evaluations hit the deadline before producing a complete candidate audit. The result includes all failures and alternating branch order; source and checkpoint fingerprints remain unchanged. [Exact offline result](../dynamics_reports/ordinary_recovery_actor8_archived_probe_20261013/result.json).

This variant is not integrated into gameplay. Its failure suggests that adding a longer frozen-actor prefix under this budget is not a useful next step. The wider plan remains full-map geometry, general marble mechanics, verified interventions and native task outcomes. The current neural student is not deployed; PPO and checkpoint promotion remain paused. A separate native fixture prototype is the next engineering step, with protected executables and sealed datasets preserved.
