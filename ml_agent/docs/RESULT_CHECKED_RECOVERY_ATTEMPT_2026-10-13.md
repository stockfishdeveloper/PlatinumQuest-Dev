# Checked recovery: incomplete ten-map attempt

The declared seed681014 comparison stopped during its fourth round when Bowl's recovery controller exceeded the900s wall bound. This is an unsuccessful engineering attempt, not a complete ten-map result. KOTM did not run and played no role in stopping or rejecting it. Long policy training and checkpoint promotion remain paused.

| Map / branch | Full-round points | Falls | Accepted / attempted rescues | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Cube / recovery | 27 | 0 | 6 / 8 | 400.95 |
| Cube / reference | 91 | 5 | 0 / 1 | 341.51 |
| Bowl / reference | 66 | 0 | 1 / 1 | 408.03 |
| Bowl / recovery | **Incomplete** | 0 before stop | 5 / 13 before stop | 901.10 |

Bowl recovery accumulated13points over206.592game seconds before termination. Keep that partial accumulation separate from a full-round score. The three completed rounds passed score/input checks, totaling7,828checked decisions with zero pending-reply mismatches, actor-output mismatches or missing playable goal contexts. No implicit OOB key requests occurred. Full-round audit reports expected20rounds, complete_declared_rounds=false and complete_pool=false.

Cube's recovery branch spent1,347decisions,86.208nominal game seconds, in physical recovery with zero gem delta during those intervals. On1,321of those decisions its actor destination differed from its logged real task. This is descriptive evidence, not causal attribution. Source inspection also found that recovery selection clears the failed-search cooldown, allowing repeated broad searches after short stops. The next experiment preserves actor task and primary cooldown as two separately logged opt-in settings while retaining recovery revalidation.

Evidence is preserved in [the attempt directory](../dynamics_reports/checked_recovery_full_pool_20261013), including original source ZIP, failed native/decision archives, failure context, partial report and completed-round audits. All170archived source members verified unchanged at termination; A2, both native executables and mlAgent.cs retained their protected hashes. No failed result was deleted or converted into a favorable score. The follow-up protocol is in [the recovery plan](PLAN_CHECKED_RECOVERY_2026-10-13.md).
