# Strategic review: moving beyond the behavior-patch loop

Planning recommendation, 2026-10-29. No training, games, behavior changes, or
checkpoint selection were performed for this review. Existing working-tree
changes were inspected and left intact. The experiments below are proposed,
not executed or previously approved as an implementation campaign.

## Recommendation

Make the main project an automatic policy-improvement loop: use the existing
physics simulator to discover better multi-event trajectories offline, learn
which continuations are valuable, and use that learning to guide a bounded live
search. First prove that the offline search can produce substantially better
outcomes across all ten maps. Do not start a large student-training run before
that proof.

The useful architectural assets already exist: collision maps, native motion
simulation, observable game state, feedback controllers, journals, demonstrations,
and native evaluation. The main proposed change is how better decisions are
discovered, evaluated, and retained across runs.

## What the evidence establishes

- The deployed controller is predominantly a programmed physics planner. A2 is
  a frozen learned proposal/fallback, not a policy being improved by the recent
  overnight campaigns. See HANDOFF_NAV_TRAINING.md and realtime.py FLAGS.
- Recent native baseline means are roughly in the 130s. For example, the two
  qr3 baseline seeds average 136.2 points across the ten maps. This is a fixed
  0.172-second native search protocol, not identical to live queued execution.
  A table of each map's best round is not an estimate of current expected score.
- Recomputed from the native comparison JSONs: launch completion gained 3.2
  points in the first 20 pairs and 5.1 in the later 20, or 4.15 across 40 pairs.
  Its reported ten-map live pilot was -2.9. This is promising native evidence,
  not an established live improvement of that size.
- HOLD_RESEARCH went from +3.05 over 20 pairs to -5.7 in its later campaign;
  QUERY_ROUTE_COPY went from +3.0 to -4.35. The surrounding flag configurations
  differed, so these are not clean repeats of one fixed treatment effect.
  Neither survived the broader testing and both were rolled back.
- still2b measured +6.15 over 20 pairs with only one activation of its actual
  rule. That gain cannot credibly be attributed to fixing the targeted behavior
  across the pool. The night audit found pre-intervention divergence in 37/40
  hold pairs, with median first divergence at 6.3 seconds.
- The October 26 ledger covers six rounds per map: approximately 140 seconds
  per round were classified as approaches. Its TreasureBox comparison was
  planner 174 versus human 217; both approach time and within-group time favored
  the human. These are unmatched layouts and gap-inferred groups, not causal
  estimates or proof of a precise ten-map human deficit.
- Current uncommitted speed-cap, fast-support, launch, and Helicopter changes
  postdate those broad results. The situation document measures one Cube Isle
  moment; it does not establish the current working tree's ten-map strength.

Evidence: docs/NIGHT_POINTS_2026-10-28.md,
docs/TAKEOVER_POINTS_2026-10-27.md, docs/TRAVERSAL_REVIEW_2026-10-26.md,
docs/SITUATION_REPLAY_2026-10-29.md, and the corresponding
dynamics_reports/overnight_20261026/*_comparison.json files.

## Working diagnosis

The strongest hypothesis is a ceiling in the decision-making and improvement
process, rather than a demonstrated fundamental limit of the simulator.

1. **The search space largely consists of programmed maneuver families.** More
   runtime searches those families more thoroughly; it does not automatically
   invent a qualitatively different sequence or preserve a discovery as learning.
2. **The objective truncates important consequences.** The live base horizon is
   48 replies, about 3.1 seconds, with longer special-case forecasts. A pickup
   receives a large fixed credit; route distance, speed, and behavior penalties
   approximate later value. The two-level evaluator really does forecast a
   second pickup, but only for a shortlist and a selected next gem. Its failed
   follow-up uses a fixed substitute. This is not whole-route optimization.
3. **Intermediate states are the task.** Reaching the same location with different
   velocity, spin, heading, inventory, and remaining time can create very
   different future opportunities. A locally good arrival is not necessarily a
   good beginning for the next part of the route.
4. **Evaluation noise hides small improvements.** Wall-time cutoffs and live
   scheduling make identical openings diverge. Screening many small changes and
   selecting positive pilots compounds that problem.

These explain why several seemingly separate visible flaws may have a common
cause. They remain hypotheses until the experiment below separates proposal
quality, ranking quality, computation, and execution.

Code anchors: nav/marble_dynamics/gem_demo.py candidate_cost;
search.py apply_two_level, useful_candidate, and search;
reliability_eval.py order_choice; parallel_search.py;
realtime.py FLAGS and dynamic planning allowance.

## The first experiment: establish a compute-to-quality curve

Start with approximately 200 saved situations, 20 per map. Include a random
representative sample plus separately reported difficult strata. Do not select
only mistakes from replays. Preserve full input-pipeline, motion, inventory,
timer, gem, and controller context. Split development and evaluation by complete
source rounds/layouts, never neighboring frames of the same trajectory.

Compare these arms on identical starts and common scoring windows:

| Arm | Question |
| --- | --- |
| Frozen baseline at the normal budget | What is the current reference? |
| Same planner with substantially more deterministic work | Is runtime starvation the principal limit? |
| Existing proposals ranked by longer executed continuations | Are good options already present but misranked? |
| Multi-event search plus trajectory refinement and longer continuations | Can broader search discover better behavior? |

Use several declared work budgets, for example 1x, 4x, and 16x, with complete
candidate evaluation and deterministic merging/tie-breaking. A repeated A/A
must reproduce decisions in the deterministic mode. Its simulated work budget
is a diagnostic, not a claim of real-time feasibility. Retain a separate actual
deadline/1x benchmark.

For the broader-search arm:

- Keep several physically different intermediate states, not only the cheapest
  arrival at each gem. Use a small beam over successive pickups, resource
  acquisitions, launches, landings, and route junctions.
- Initialize with existing controller trajectories and compatible human
  demonstration segments, then iteratively vary steering, throttle/braking,
  event timing, and target order. Use multiple starts to avoid local traps.
- Search across useful event boundaries; test roughly 10-20 seconds initially
  and measure horizon sensitivity. This range is an experimental choice, not a
  known optimum. Execute a short prefix, observe, and replan.
- Evaluate real gem values at a common elapsed-time horizon, with later
  continuations to expose exit-state and inventory value. Whole-group completion
  time is a useful diagnostic, but does not alone price the next journey.
- Preserve all legally relevant state. Future random gem groups are hidden:
  continuation estimates may average over independently sampled possibilities,
  but the selected action must not know the actual upcoming group/seed.

The existing offline benchmark is a useful starting point, not already a full
game training environment. It stops on a fall and normally stops at group clear;
its optional next-group mode introduces a recorded group and ends at the next
pickup. Validate broader item events, respawns, and stochastic continuation
before using full-return labels. Use native execution to verify sampled winning
trajectories, including small realistic start perturbations. Forecast success
alone is insufficient.

An initial engineering go/no-go target could be a clear, repeatable improvement
around 10% in held-out task outcomes across multiple map types. That is a proposed
investment threshold, not a statistical guarantee or predicted gain. Report
fixed-window points and completion outcomes separately; do not convert faster
isolated segments directly into a promised round-score improvement.

If expanded search does not beat the baseline even offline, stop the learning
phase and inspect whether useful trajectories are absent, pruned, misranked, or
lost during execution. Human trajectories can test those distinctions without
turning their maneuvers into another set of special-case rules.

## If the slower teacher succeeds: make its discoveries reusable

First learn a continuation-value/ranking model and a proposal model while
retaining physics evaluation in the live loop. Rank choices by their measured
future outcomes; propose several plausible maneuvers instead of regressing all
successful modes into one average steering command.

The value input must include geometry/route context, the known remaining gem
set and values, motion/contact state, resources, and time remaining. Prefer a
reusable geometry representation to map-name rules. A model trained only on the
current controller's outcomes estimates that controller's limitations; expand
its labels with improved search and subsequent on-policy data.

The proposed recurring loop is:

1. Play and save representative states automatically.
2. Spend offline search on those states and compare actual continuations.
3. Keep improved trajectories and informative losing alternatives.
4. Train value/proposal models; test closed-loop performance on separate rounds.
5. Use the improved models to guide the next search/data-collection cycle.

Include perturbed starts and states visited by the new controller so small
execution errors become part of training. Keep a representative sampling quota
beside automatically prioritized high-loss states. This prevents replay-selected
mistakes from becoming the entire curriculum.

Only after this earns points should we consider distilling more control into a
standalone actor or fine-tuning it with full-round RL. Human demonstrations are
valuable seeds and standards of comparison; this should not require the user
to find and explain every new failure.

## How this differs from previous attempts

- The October 13 student loop already collected corrections on student-visited
  states. Merely calling a new loop DAgger is not new. Iteration 3 reached 11,590
  cumulative corrections but scored 31.9 versus paired A2 44.0. The proposed
  differences are a demonstrably stronger teacher, multi-event outcome labels,
  and learning search guidance before replacing the entire controller.
- Six fixed chain variants failed. A fixed gem order with an open-loop chain is
  not the proposed search over evolving physical state with feedback execution.
- The 12-perturbation refinement already failed its native screen. It used one
  local round and the existing objective/gates. Repeating it is not the proposed
  iterative offline search with longer measured consequences.
- Time-price and speed-cap changes already failed broader checks. A learned
  continuation value is not another global speed reward or seconds coefficient.
- Parts of this direction appear in the October 26 traversal review. The change
  is to make it the central development program and test its central hypothesis,
  rather than resume a queue of isolated flags.

## Whole-project success criteria

The primary result remains higher expected full-round points, equally weighting
the ten maps. Show per-map results, uncertainty, and poor-round tails; do not
require every map to win independently or promote a change from a best round.
Reserve unseen round seeds/layouts for confirmation and account for repeated
experimentation. Real-time confirmation must include actual queued execution
and holds; native fixed-budget points alone cannot establish deployment value.

Freeze and record the baseline/configuration before the pilot. Establish the
deterministic A/A and representative experiment without turning this into an
open-ended infrastructure rewrite. Defer feature additions during that pilot.
No new neural physics model or extra hardware is a prerequisite. Native
batching/shared map data may be needed if measured teacher throughput is poor;
the existing memory pressure makes simply adding workers a poor default.

## Research context

These papers support algorithmic directions, not predictions of success here:

- Anthony et al., Expert Iteration: improve plans with search, learn from them,
  and use the learner to guide stronger search.
  https://arxiv.org/abs/1705.08439
- Howell et al., Predictive Sampling: physics-based online trajectory sampling
  can synthesize feedback behavior; the work also emphasizes efficient search.
  https://arxiv.org/abs/2212.00541
- Ross et al., DAgger: imitation must address states induced by the learned
  policy. The repository's previous on-policy correction attempt remains direct
  evidence that this principle alone does not guarantee a useful student.
  https://proceedings.mlr.press/v15/ross11a.html
