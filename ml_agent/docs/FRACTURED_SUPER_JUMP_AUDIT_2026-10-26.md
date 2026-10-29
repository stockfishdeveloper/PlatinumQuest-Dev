# Fractured Islands: wasted Super Jump investigation

The user stopped the visible escape-only candidate run. Its owned planner,
eight forecast workers and game were terminated. Do not restart automatically.
The game had disconnected before termination; this is an incomplete viewing
run, not a full-round result or improvement claim.

## Recorded event

Evidence folder: `dynamics_reports/traversal_recovery_20261026/escape_visible_fractured_v2_20261026/FracturedIslands_Hunt`.
Journal: `FracturedIslands_Hunt_20261026_220702.jsonl`.
At tick 3175 the marble collected a 1-point gem. At tick 3176, with
165.824 seconds remaining before the action, it pressed use while holding
Super Jump. Label: `direct_use_0_jump_-1_ballistic`. The target reconstructed
below was gem 32344 at (-37.0014, -33.0064, 2.52302), approximately 2.5 units
away. A 5-point gem at (-42, -38, 2.05) was also still visible.

The gzip trace was interrupted and lacks its final footer. The audit reads
only complete recoverable lines, explicitly retaining that limitation. There
are complete observations through tick 3184, covering the use. The journal
continues through tick 3229. There is no full-round score.

## Reconstructed candidate decision

`audit_fractured_sj.py` builds the normal live planner world with eight candidate
workers and the same flags. It searches the recorded tick-3176 state and also
the state predicted from tick 3171 through the five recorded actions. Both
select the exact recorded Super Jump label for gem 32344. These are unbudgeted
diagnostic searches, not a replay of the original CPU deadline or a recording
of its original complete candidate list.

| Accepted forecast | Target pickup | Side-impact penalty | Total cost, lower wins |
| --- | ---: | ---: | ---: |
| Super Jump, direct ballistic | 34 replies = 2.176 s | 0 | -70.6774 |
| Best-scoring non-use, frozen navigator landing | 47 replies = 3.008 s | 50 | -37.9455 |
| Ordinary jump, direct jump 9 ballistic | 18 replies = 1.152 s | 100 | -0.6038 |

All times above are forecast pickups, not measured counterfactual live times.
The Super Jump forecast also pays approximately 20.57 cost units for moving
away from the following gem and 12.288 for extended forecast duration.

## Causes supported by the code and reconstruction

1. `search.cluster_penalty` charges 50 units for every predicted side impact.
   Pickup delay is priced at only 4 units per second. An ordinary jump's two
   predicted impacts therefore outweigh its 1.024-second faster pickup by a
   large margin. The scoring rewards avoiding contacts more than saving time.
2. `assess` records `resources_consumed=1`, but default `useful_candidate`
   ranks by cost without consuming-inventory cost. `preserve_resources` is
   false in the live flag set. The special one-gem inventory penalty and
   last-gem reservation apply to Super Speed, not Super Jump. The value of
   keeping Super Jump for the next group is not represented.
3. `--use-margin-s 1` becomes an 8-unit COST margin; it is not a requirement
   to save a measured second. This candidate beats the best non-use score by
   32.73197 units and passes, even with faster accepted ordinary jumps.
4. The two-gem time reranker returns early because the route from this gem
   to the next is 12.065553 units, slightly above its hard 12-unit limit.
   Consequently it does not compare the time to both remaining gems here.
5. The escape-only experiment preserves a forecasted current-target pickup.
   It imposes no inventory-value rule, so it preserves this bad local choice
   too. The independently searched 5-point target has a no-use pickup option;
   this is not evidence that its entire round would be better.

In the same candidate set, enabling resource-preserving selection chooses
`frozen_navigator_landing` with no consumption. That isolates the inventory
selection issue; it is not a recommendation to globally enable that option
without throughput testing, since it still picks a slower ordinary option.

## Next correction to evaluate

Require an expendable-powerup plan to justify spending inventory by predicted
task/cluster completion time or necessary reachability. Contact penalties must
not manufacture a time saving. Include the subsequent state and retained
inventory in that comparison; do not invent knowledge of future randomized
spawn locations. Cover this recorded state as a regression case, then test
full-round points. No production policy was changed by this investigation.

Machine evidence: `fractured_sj_candidate_audit.json`,
`fractured_sj_projected_state.json`, `fractured_sj_trace_readable.json` under
`dynamics_reports/traversal_recovery_20261026`.

## Authorized follow-up: full viewing round, then correction

The user subsequently requested another Fractured 1x round, closure, then the
fix. `escape_visible_fractured_full_before_fix` completed normally: 114 points,
1 fall, 0 rescues, 189.1 seconds wall time in the scored loop. The game closed
before source changes began. `summary.json` is complete; no post-fix live score
has been measured in this task.

For a held Super Jump, production search now calls `apply_use_gain` after
physical backup checks and the existing cluster reranker. `--use-margin-s 1`
requires one second of forecast completion-time gain for Super Jump in addition
to the existing cost-margin check. A mixed-cost advantage alone cannot justify
a slower pickup.
`assess` records held-inventory consumption separately from Blast/script-event
consumption, so the gate works for retained-plan labels and does not mistake a
Blast for a held powerup. This is a Super Jump rule, not a map-specific or
gem-count exception. Other powerup types retain their previous cost-margin
behavior, including the existing Super Speed ride exception.

When accepted non-consuming forecasts collect the target and its next visible
tour gem, compare completion of that same pair. A consuming forecast missing
that completion cannot claim it is faster. This comparison does not have the
old 12-unit route cutoff and makes no additional forecasts. Otherwise compare
target pickup times. With no accepted non-consuming target pickup, leave
existing travel/acquisition/reachability choices available. This is bounded
forecast evidence, not proof of global unreachability or a learned long-term
inventory value. Future randomized spawn positions remain unknown.

The recorded actual state and its five-reply predicted state both reject
`direct_use_0_jump_-1_ballistic` as `use_no_time_gain`, recording a -1.024-second
gain against the faster ordinary jump and requiring +1 second. Both select
`frozen_navigator_landing` consuming no inventory. This fixes the wasteful use;
the remaining contact-cost preference for a slower ordinary plan is a separate
throughput issue, not silently solved by this patch.

Evidence for the final Super Jump scope:
`fractured_sj_final_fix_checks.json` (passed, sources unchanged). The focused
search/runtime/native-wrapper suite passes 44 tests, including 12 new cases
covering this event, tiny savings, meaningful savings, necessary-use forecasts,
unsafe counterfactuals, retained labels, Blast distinction, and pair completion.

A ten-map short replay comparison ran against the archived pre-fix search.
Its first attempt failed because override module import order caused
incompatible proposal class identities; it was stopped, not counted as a loss
or success. The corrected broad-inventory run `use_gain_screen_v2` LOST:
85 -> 77 replay points, 1 -> 3 incomplete/failing cases. Crag hit the 40-second
cap, Vortex fell, TreasureBox completed with a lower-valued next-group pickup.
All evidence is preserved and that broad policy was rejected. A traced Crag
diagnostic subsequently completed with 11 points, but is not a replacement for
the failed quiet case; its logged guard rejection concerned Super Speed rather
than Super Jump.

The correction was narrowed to held Super Jump. The final guard still rejects
the original Fractured use in both recorded states, and 44 tests pass again.
The subsequent ten-map rerun is `use_gain_screen_sj`, with the same archived
baseline and starts. The archived baseline has only an engine-attribute
compatibility adapter; both arms use identical escape-only target selection.
These replay cases do not establish live round score improvements.

The Super Jump-only screen completed all 10 pairs: 82 -> 85 replay points,
2 -> 1 incomplete/failing cases. Equal-point shared completion time ratio was
0.9722 over eight cases. The per-case reversal on Vortex and substantial
baseline variation on Crag reinforce that this small screen is diagnostic.

After this screen, the final conservative adjustment retained the old
Super Jump cost-margin check too, making the new time check additive rather
than admitting any previously cost-rejected Super Jump plans. This preserves
the old guard on travel where neither plan forecasts a pickup. The final code
was rechecked with 44 passing tests and the actual/predicted incident searches;
both keep inventory. The final additive adjustment has not had a new ten-map
screen or a post-fix live round. Do not attach the precursor screen's score
increase to that final source hash as a confirmed gain. No games remain open.
