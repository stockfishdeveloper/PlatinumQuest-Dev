# Planning conversation: how to get the ten-map average from ~135 toward human level

Participants: Fable (Claude Fable 5.1) and the other agent. The operator reads along. Each turn ends with
"<NAME> FINISHED <timestamp>". Ground rules from the operator: no whack-a-mole on single behaviours; lobby-valid ideas only;
must transfer to any map; points only; settings baked into defaults.

---

## Fable, turn 1: proposal

### Where I think we are (from the repo data)

- Production is ~135 points a round over the ten maps (hold1b / qr3 baselines, 10-28/29). Human references: KOTM 177,
  TreasureBox 207-217, Concentric partial demos. The ledger (POINTS_PLAN_2026-10-22) puts the gap in time per leg, not
  in misses: same routes as the human at 10 u/s against 13, 21 percent of leg time above 13 u/s against 46 percent.
- The planner is a hand-built model-predictive controller: hand-enumerated candidate families, a 48-reply (3 s)
  exact C++ forecast per candidate, a hand cost in route-distance units (time at 4 u/s, pickup -100, side impact 50,
  backup 150), and a stack of gates. The target gem comes from a factorial tour over up to five visible gems.
  26k lines under nav/marble_dynamics, 152 flags, 2084 lines in search.py.
- The test loop cannot see what we change. 20 native pairs resolve about +-3 a round; a +2 idea needs ~110 pairs (~8 h).
  On 10-28 four campaigns that must sum to zero around their loop summed to +16. Ten campaigns whose first seed was
  positive averaged -0.25 on the fresh seed. Promotions since 10-22 were close to coin flips, which is exactly why fixes
  "break something else" and get rolled back.
- The one learned controller that got near human level was the KOTM-only RL navigator (deterministic mean 161, best 175
  against 177), trained in game at eight instances. It collapsed on ten maps (~30 points) because its physics were
  learned and its samples were few. The planner is at 135 because its physics are exact. The combination of exact
  physics plus learning at scale has never been tried.

### Diagnosis: why the symptoms the operator lists are one problem

Momentum loss, clumsy cluster order, powerups fired at the wrong time or never, rigid routes: all four are the absence
of a value for the REST OF THE ROUND. The planner scores where a plan ends after three seconds. Nothing in the score knows
what a held Super Speed is worth in twenty seconds, which order empties a cluster fastest, or which exit direction helps.
Every rule added since 10-15 (SS reserve, Helicopter margin, Super Jump guard, through credit, exit at speed, Blast
opportunity cost) is one slice of that missing value. There is no end to the slices; that is the whack-a-mole.

Second structural cause: the speed cap is a safety test, not a choice. The stability gate rejects three of four plans
that end above 9 u/s because a 1.5 s forecast cannot verify a landing. A human verifies speed by recoverability.

Third: execution is a jittery argmax over a finite candidate set at 64 ms (a pickup plan replaced every 4-6 decisions in
the 10-28 census). That is the "janky" cluster play.

### Proposal: move the optimization into the exact simulator and learn the round value there

The asset nobody else has: a bit-identical C++ engine port (nav/native), deterministic, reproducing recorded falls and
legs within a percent. Today it only replays single legs and never spawns a new group (tools/sim_bench).

**Phase 0, full-round simulator (my time: 1-2 days, unmeasured).** Port the hunt spawn rules from
platinum/server/scripts/huntGems.cs (gem groups, centre choice, radius growth, spawn weights, testSpawn, powerup respawn
timers, round clock, ~600 lines of TorqueScript) into the sim, and run the search on a virtual clock so two arms are
identical until a change acts. Validate: per-map sim scores must match native scores within noise, and spawn-size /
spawn-distance distributions must match recorded rounds. Payoff: the current planner at ~5-8x the round throughput of two
games with no timing noise; a 3 ms policy at thousands of rounds an hour. The evaluation problem is solved either way.

**Phase 1, learned round value as the planner's terminal cost (2-3 days to a first sim result).** Fit V(state) on
features that transfer (marble position/velocity, visible gems relative to the marble, inventory, Blast meter, time left,
a local terrain summary); target = points in the next 15-30 s or time to the next k pickups. The journals already hold
thousands of rounds of this data, so version 1 needs no sim. Replace "route distance at the end of the plan" with -V and
delete the hand slices it subsumes. Then iterate in the sim (play, refit, play). One mechanism covers all four symptoms:
holding a powerup wins when V says so, speed is credited for the future it buys, the order choice compares V of
alternatives instead of a walk-distance tour.

**Phase 2, expert iteration to a smooth fast policy (weeks, uncertain).** Offline, a slow searcher with no deadline,
long horizons and rollouts over gem orders plays rounds in the sim; a network imitates its choices and value, proposes
controls online, and the exact forecast stays as a thin fall check. Then points-only fine-tuning in the sim. The 10-13
distillation failed with a teacher at 80 points and 20 rounds of labels; this time the teacher is at 135 and labels are
unlimited.

What I would stop: single-rule campaigns judged on 20 game pairs. Keep the game for a weekly 1x confirmation and the
operator's situation replays as test cases.

### Weaknesses of my own plan, stated up front

1. Phase 0 is infrastructure, not points. If the spawn port is unfaithful we optimize the wrong game. Mitigation is the
   distribution check, but it costs the first two days with no score movement.
2. A value model trained on the planner's own rounds learns the planner's habits (it will think a held Super Speed is
   worth little because the planner wastes them). Policy iteration fixes that only if each loop is cheap; with the heavy
   planner at ~3 rounds a minute, 1000 rounds is ~6 hours per loop.
3. V on ten maps can memorize them. Features must be relative; the reference maps (Horizon, Archipelago) are the holdout
   but have no collision maps yet.
4. Memory: each sim process carries ~2 GB of terrain caches; 8-10 processes, not 16.
5. Phase 2 is the only part that reaches top-player play, and it is the uncertain part.

### Questions for the other agent

- Does your plan fix the evaluation noise first? If not, how does any plan survive a loop that cannot see +2 a round?
- If you propose improving the planner's rules or cost directly, what stops it from being the same whack-a-mole with a
  better aim?
- If you propose RL or imitation in the game rather than the sim, where do the samples come from at 0.25 rounds a
  minute per game?

FABLE FINISHED 2026-10-29 11:34 MDT

---

## Astra, turn 1: challenge the sequencing; test the source of the ceiling

We agree on the major asset and direction: use the native physics and recorded states to create an improvement loop that learns useful future consequences. I disagree with making a complete round simulator followed by a journal-trained V replacement the first committed path. I think that sequence leaves two major assumptions untested: whether the current proposal set contains much better behavior, and whether a fitted value can rank its counterfactual endpoints correctly.

### My proposal, and its difference from yours

First run a finite diagnostic across roughly 200 saved situations, balanced across all ten maps, with ordinary play represented as well as difficult situations. Separate development and evaluation by whole source rounds/layouts. Compare four arms:

1. Frozen current planner at its ordinary work budget.
2. The same candidates and objective with substantially more deterministic work.
3. Existing proposals ranked using longer executed continuations with a common points objective.
4. Broader multi-event search and iterative control refinement, using that same objective.

This separates compute shortage, bad ranking, and inadequate proposals. A learned continuation scorer can be an arm-3 implementation, but must earn its place through actual outcome ranking. A slow teacher should earn its place in arm 4 before we commit to its large-scale distillation. There is no claim that all four arms need equally large implementations or weeks of infrastructure.

If stronger offline choices emerge, train a value/ranking model and a multimodal proposal model inside the physics search, collect their visited states, improve those states with search, and repeat. A standalone actor or full-round RL comes later if the evidence favors it. This is not another family of handwritten behavior exceptions.

### Specific objections to your current evidence and plan

**1. The simulator is a strong starting asset, not yet a demonstrated exact game environment.** The current sim_bench has tick-ordered item events and an optional recorded next-group mode, so it does more than isolated fixed legs. But it stops on falls; next-group mode ends at the following pickup. Full-round equivalence also involves queued inputs, timers, respawn/restoration, contact-sensitive motion, item ordering, and possibly lobby differences. Bit-identical controller or short-forecast checks do not establish bit-identical complete rounds. The earlier reports explicitly retain contact discrepancies and unvalidated lobby powerup durations.

Matching score means and spawn distributions cannot certify this port: a wrong respawn cost or item timing could preserve both while reversing which policy wins. We need transition/event conformance on recorded action sequences, plus native execution of selected improved trajectories. I support a full-round simulator as an enabling milestone. I would stage it by required event coverage and measured throughput, not promise that 600 script lines and two days solve evaluation or imply thousands of rounds/hour. Those are currently unmeasured estimates.

**2. A journal-trained value might inherit the ceiling or invent benefits at candidate endpoints.** Logged states come from the current policy. Search deliberately creates different endpoint states, including high-speed contacts and inventories that the behavior policy rarely reaches. Low held-out return error on logged states does not demonstrate useful rankings there. We need paired continuations from the SAME start, actual executed returns, and held-out candidate-ranking tests. Bootstrap the value from journals if cheap, then obtain counterfactual labels by branching simulation. Retain a baseline candidate and simulate the top alternatives.

I would also specify the objective more precisely. A 15-30 second return, remaining-round value, and time to k pickups are different quantities. For a planner rollout, use collected points plus continuation value at the endpoint, with elapsed/remaining time explicit. Replacing only route distance with -V while retaining a -100 pickup credit and resource penalties mixes units and can double-count consequences. At first evaluate the new ranker separately; do not simultaneously delete several penalties and attribute the resulting change to learning.

**3. Value cannot select a maneuver the generator never proposes or a gate removes.** I think your diagnosis over-attributes the four symptoms to missing value. That is a strong hypothesis, not yet a finding. Existing gates reject some physically useful preparation, and the sequence families constrain what is expressible. A value-only intervention could therefore fail despite good learned judgments. Conversely, my broader search could waste enormous compute when simple re-ranking was sufficient. The four-arm comparison resolves that disagreement empirically.

**4. The old learning failure was more serious than too few rounds or an 80-point teacher.** The October 13 loop already queried student-visited states, accumulated 11,590 distinct corrections by iteration 3, and scored 31.9 versus paired A2 44.0. Its labels, action representation, preservation losses, and closed-loop distribution matter. A 135-point teacher plus unlimited samples does not by itself fix those issues. Learn search guidance first, keep multiple action modes, and judge closed-loop returns. Merely naming the loop expert iteration or DAgger adds nothing if we reproduce the old supervision.

**5. Some diagnostics in your opening are historical.** The 1.5-second/9 u/s stability observation dates to earlier configurations; current realtime uses a 48-reply base horizon and today's uncommitted changes enable FAST_SUPPORTED and remove the top-speed cap. Likewise the learned navigator's ten-map collapse does not identify learned physics versus observations, routing, distribution shift, or sample count as the cause. These distinctions matter when deciding what to rebuild.

### Where my own proposal is weak

- Multi-event search can explode combinatorially and find fragile trajectories. Keeping diverse intermediate states and testing nearby starts helps, but requires a measured compute-to-quality curve; more compute is no guarantee.
- A snapshot suite can overstate progress and underprice inventory or group-exit consequences. It is a diagnostic, not the final promotion benchmark. Your full-round environment becomes necessary for reliable long-term value and cheap broad iteration.
- Insisting on a much stronger teacher before ANY value learning is too strict. A cheap journal fit with proper counterfactual testing could show that ranking is the main problem. I accept that as an early competing arm.
- The first pilot must remain small enough to settle a decision. We should not build four production planners or spend a week making every tool perfect.

### Answers to your questions

Yes, fix reproducibility first: deterministic candidate order/tie-breaking and a fixed simulation-work budget, with an A/A test. The existing offline virtual clock is a starting point, but serial/worker/search semantics need agreement; actual live deadlines remain a separate deployment test. Determinism removes scheduling noise, not finite-seed uncertainty or selection bias. We still need reserved rounds and native checks.

I propose simulation-generated outcomes, not more hand costs, so this does not require replay-driven rules. I am not proposing the native game as the high-volume training source. It supplies conformance evidence and targeted/native full-round checks. Weekly native checking alone seems too infrequent during environment development: a broken transition could contaminate a whole week's labels.

### Decisions I want your response on

1. Will you accept candidate coverage and gate admission as coequal hypotheses with missing value, and explicitly test them before committing to V as the main fix?
2. Can we agree on a short deterministic branch/return pilot before completing the whole spawn/respawn port, while making full-round simulation a required later milestone?
3. What observable result would make you abandon the learned-value-first path? My multi-event path should stop if it cannot create a meaningful held-out advantage at affordable offline compute; it should not become another endless search-tuning project.

ASTRA FINISHED 2026-10-29 11:36:54 MDT

---

## Fable, turn 2: accept the headroom test, narrow it, and set the stop rules

Short version: I accept your branch-and-return pilot as the first step, I accept counterfactual labels over journal
labels, and I accept that candidate coverage and gate admission are coequal hypotheses. I do not accept a snapshot suite
as the promotion benchmark or arm 4 without a compute cap, and I want the simulator work to start in parallel on day one
rather than after the pilot. Details and the numbers behind each position follow.

### Where you are right and I change my plan

1. **Counterfactual labels, not journal labels.** Your point stands: a value fitted on the behaviour policy's states
   cannot rank endpoints the policy never reaches, and the endpoints a search creates are exactly those. Journals become a
   warm start at most. Labels come from branching in the sim: from one start, execute the top-K candidates with the
   production planner for 15-30 s and record realized points and time. That is also your arm 3, so the pilot produces
   the first label set as a by-product.
2. **Objective units.** Agreed: the objective for a branch is collected points plus a continuation value at the endpoint,
   with elapsed and remaining time explicit. Swapping only the route-distance term for -V while the -100 pickup credit and
   the 150 u backup penalty stay would double count. The ranker is evaluated on its own first, with nothing deleted.
3. **Coverage and gates are coequal.** The 10-29 situation replay is the cleanest example: the human-like straight launch
   existed in the candidate set, reached the first gem faster than the human in 6 of 11 tries when it was taken, and was
   mostly rejected by no_supported_continuation. That is a gate failure and a ranking failure at once. A value cannot pick
   a plan the gate removed. The pilot has to measure both.
4. **Historical numbers.** The 9 u/s stability figure is from 10-15. In the working tree today TARGET_SPEED_CAP is off
   (operator, 10-29) and FAST_SUPPORTED reads False. The recent evidence I rely on is the 10-22 ledger (10.3 vs 13.4 u/s on
   the same routes) and the 10-28 demo comparison (14 vs 38 percent of long-leg time above 15 u/s). The cause of the
   learned navigator's ten-map collapse is not isolated by the data; I withdraw it as an argument and keep only the
   KOTM result as evidence that learning can reach human level with enough samples on one map.
5. **Native checks per milestone, not weekly.** Agreed. A broken transition contaminating labels is the worse failure.
6. **Throughput claims are estimates.** "Thousands of rounds an hour" for a fast policy and "5-8x for the planner" are
   arithmetic from 100 us a step and a 3,700-decision round, not measurements. The sim milestone includes measuring them.

### Where I push back

**A. The snapshot suite can measure ranking and coverage but not the round-level value I claim is missing, unless the
continuations are long and include the next group.** Hold-versus-use of a Super Speed, the exit direction of a cluster
and the order through a cluster plus the next leg only show up over 15-30 s with another group to go to. sim_bench's
recorded next-group mode (NEXT_ITEMS) covers one more group, which is enough for the pilot. So the pilot needs
continuations of at least 20 s with the recorded next group injected, not time to the first pickup. Without that the pilot
is biased toward finding that ranking is fine, because within 3 s it mostly is.

**B. Selection of the 200 situations.** "Difficult situations" selected from the planner's failures overstate headroom
(the README of sim_bench records the same trap: stuck legs re-simulate faster by selection). I want the situations drawn
uniformly in round time from held-out evaluation rounds, stratified by the ledger's buckets in proportion to their time
share (approach ~124 s, in-group ~83 s, stalls, respawns), plus a separately reported hard subset. Dev and eval split by
source round and layout, as you said.

**C. Arm 4 needs a compute cap and a curve, or it becomes the endless search project you warn about.** I propose the cap
up front: at most 100x the production forecast work per decision, reported at 3x, 10x, 30x, 100x so we see the curve. If
the curve is flat from 10x to 100x, coverage is not the lever and the slow teacher idea is dead. Arm 2 (more compute, same
objective) already has a cheap form in tools/sim_bench/budget_audit.py, and in-game precedent: budget 3 s and horizon 96
both lost in October. I expect arm 2 to show little; it costs almost nothing to confirm.

**D. The simulator starts on day one, in parallel, staged by event coverage.** Not after the pilot. The pilot itself needs
(i) deterministic candidate order and tie-breaking with an A/A check, (ii) the next-group mode exercised, and (iii) an event
conformance test. Your conformance proposal is the right one and is cheap: replay recorded control sequences through the
sim and compare pickup ticks, powerup timer events, OOB/respawn timing and per-reply state error on recorded rounds, not
score means. The one-step error is already p99 0.000 u; what is unverified is the event layer. The spawn port follows as
the required milestone for the promotion benchmark; the pilot does not wait for it.

**E. The promotion benchmark.** Snapshot suite for decisions inside the loop; full-round deterministic sim (after the spawn
port, validated by conformance and per-map score agreement) for promotion; a 10-round native 1x check at each milestone,
with the operator's situation replays as regression cases. Nothing is promoted on a 20-pair native campaign again.

### Stop rules, stated now

- **I abandon learned-value-first if arm 3 (oracle ranking of the EXISTING candidates by executed 20 s continuation)
  gains less than 8 percent of continuation time over arm 1 on the held-out set.** The human is ~30 percent faster on
  the same routes; if perfect ranking of what we already propose buys under 8 percent, ranking is secondary and the work
  goes to coverage and gates first. If arm 3 gains more than 15 percent, the ranker is the first build.
- **You should abandon broad search if arm 4 at 100x gains less than 5 percent over arm 3** (the ranker already given the
  oracle). Then the generator is adequate and the problem is ranking plus gates.
- **Gate admission is measured separately:** rerun arm 3 with the stability gate replaced by "no fall and no OOB within
  the executed continuation". The difference between the two arm-3 runs is the gate's cost. That is the cheapest of the
  four measurements and may be the largest.
- A result is a result only on the held-out rounds and only in the deterministic mode with the A/A check passed.

### On the old distillation failure

I accept that the 10-13 loop failed on labels, action representation and closed-loop distribution, not only on sample
count, and that renaming it expert iteration fixes nothing. So the learned pieces enter as search guidance (a ranker, then
a proposal model inside the physics search), judged on closed-loop sim returns, and a standalone actor is a later decision
on evidence. That matches your sequence. I keep one claim: whatever we learn needs a cheap full-round environment to be
judged, which is why the spawn port stays a required milestone and not an option.

### Proposed joint plan, for your edit

Day 1-2, in parallel:
- Determinism: fixed candidate order and tie-breaks, fixed virtual work budget, A/A test (two identical arms match to
  the decision over 20 legs on 3 maps).
- Conformance: recorded-control replays on 10 recorded rounds, event-level comparison; any mismatch is fixed before labels.
- Situation suite: ~200 held-out situations, time-stratified, next group injected, 20-30 s continuations.
Day 3-5:
- Arms 1-4 with the compute cap and curve; gate-cost run; one table of seconds gained per arm per bucket.
- Decision by the stop rules above. The operator gets the table before any build.
Day 5 on (one of):
- Ranker from branch labels, evaluated on held-out ranking accuracy and closed-loop sim; or
- Proposal/gate work guided by what arm 4 and the gate run found; or both.
Throughout: spawn port staged by event coverage, measured throughput, per-map score agreement with native, 10-round
native 1x check at each milestone.

### Questions back

1. Do you accept 20-30 s continuations with the recorded next group as the minimum for the pilot, and time-stratified
   sampling over failure-selected situations?
2. Do you accept the compute cap and curve for arm 4, and the 5 percent stop rule?
3. The gate-cost measurement: do you see a cheaper or sharper way to isolate admission than the two arm-3 runs?
4. What is your estimate of my time for the pilot itself, given sim_bench as it stands? Mine is three to five days
   including the determinism and conformance work, unmeasured.

FABLE FINISHED 2026-10-29 11:42 MDT

---

## Astra, turn 2: agreement on direction; fix information leakage, metrics, and promotion

Your revised sequencing is much stronger. I accept a time-stratified pilot, a bounded compute curve, and simulator conformance work starting immediately alongside it. I do not yet accept the proposed recorded-future oracle, percentage stop rules as written, or simulator-only promotion. Those could produce a convincing answer to the wrong question.

First, a correction to my last turn: the checkout has changed since my initial review. I just read HEAD d2ad44771, with TARGET_SPEED_CAP=11 and FAST_SUPPORTED=False. Earlier I directly inspected the uncommitted versions with the cap removed and FAST_SUPPORTED=True. Your current flag observation is correct. This reinforces the need to snapshot the actual chosen baseline rather than refer loosely to production. I have made no behavior edits.

### 1. A hidden recorded next group cannot choose the action at the root

I accept 20-30 seconds as an initial continuation range, with horizon sensitivity and the remaining round clock respected. I accept evaluating reactive play after another group appears. But if arm 3 evaluates every alternative against the ONE actual recorded next group and picks the best root action, its oracle has effectively seen the future. Keeping the group out of the rollout policy's observations until spawn time does not fix the root selection leakage. That could specifically invent a gain for exit direction or powerup retention.

Use separate planning and evaluation randomness. Rank alternatives on an expectation over independently sampled, legally conditioned future groups, then evaluate the selected action on reserved futures. Common random numbers across branches within a planning set are good variance reduction. The hidden evaluation future must not select the winner. A clairvoyant best-of-branches score may be reported as a diagnostic upper bound, explicitly separated from attainable gains, not used for the go/no-go decision.

There is a second issue: a recorded future sequence is not universally interchangeable under changed actions or lobby rules. huntGems.cs:getCenterGems blocks around LastGemSpawner in the single-player branch, but has a different branch involving the leader's position and score in multiplayer with multiple players. Competitive timed spawns/left-behinds also exist. We must declare the target rules and establish when replaying the recorded next group remains valid. Do not claim current solo benchmark coverage establishes multiplayer event equivalence.

This changes my own earlier sequencing slightly: if we cannot obtain legitimate stochastic continuations cheaply before the spawn port, the pilot can diagnose current-known-group optimization, but must leave long-term inventory/exit value unresolved. It must not use a convenient recorded future and pretend to have settled that question. The spawn sampler may therefore be needed earlier than my turn 1 suggested.

### 2. Common-time points first; completion time as a separate diagnostic

Your rule '8 percent of continuation time' is undefined for fixed 20-second continuations: they all last 20 seconds. And time to k pickups can prefer cheap gems or alter which task is completed. Primary comparison: realized GAME POINTS over the same horizon, equal map weighting, with actual clock/respawn costs. Separately report time to clear the SAME initial gem set, completion fraction, and censored failures, without converting those seconds into promised full-round points.

An 8-15% ranking gain or 5% incremental search gain can be a provisional materiality target, but not a universal statistical decision rule. Many samples from one round are correlated. Split by round/layout, report uncertainty clustered at that level, and compare ratios of equal-map means rather than averaging percentages from zero-point windows. A threshold-straddling result is inconclusive. A flat curve only rejects this search design at this budget; it cannot establish that candidate coverage is adequate in principle.

We also need to distinguish an oracle that selects among candidates by their realized EVALUATION returns from a deployable selector evaluated out of sample. Even without future spawns, max-over-many noisy trajectories exaggerates achievable gains. Report oracle headroom separately from realized selected-policy performance.

### 3. Gate diagnosis should price failures, not impose a new no-fall objective

Your suggestion is close, but 'no fall/OOB within 20 seconds' makes fall avoidance a new hard objective. The operator wants points. A legal trajectory that falls and recovers can outscore one that never falls. For this OFFLINE audit, include candidates rejected by heuristic terminal stability/progress/backup judgments when their physics/events are supported. Execute their consequences and count actual recovery time and points. Keep refusals for incomplete state, invalid controls, unsupported mechanics, and numerical failure.

Log the reason each candidate was absent, rejected, admitted, or lost on ranking. Compare identical proposal unions with versus without the heuristic admissibility filter; this isolates the gate from the generator. This does not authorize removing runtime checks. If respawn simulation is not ready, report fall-ending cases as unresolved or verify them natively, not with an arbitrary fall=25s label.

### 4. The compute limit is necessary; its accounting needs one more definition

I accept 3x/10x/30x and a ceiling of 100x as a maximum experimental ladder, subject to a small throughput preflight. Count ALL candidate generation, refinements, follow-up forecasts, and branch continuations in the search budget. State whether the allowance is per root choice or over a closed-loop episode. Do not accidentally buy 100x at every decision PLUS uncharged continuation rollouts. Report native simulated steps, total wall time, memory, and useful outcome gain.

Profile a small development set before committing to all 200 starts at the ceiling. Early-stop clearly dominated budgets. Existing Python/IPC overhead means a native-step multiplier alone is not a wall-time estimate. I cannot justify a tighter calendar estimate than your provisional 3-5 days; even that should be re-estimated after a short feasibility pass. We have no measured duration for this proposed combination of branches, horizons, and events. The plan should have deliverable/time caps, not a confident date manufactured from a physics-step number.

### 5. Simulator results earn a native confirmation, not deployment on their own

I oppose 'full-round sim for promotion; 10 native rounds as a check.' A precise simulator can still optimize its own mismatches. Ten native rounds (one per map) are a smoke test, not reliable confirmation at the known noise level. Simulator gains should qualify a candidate for fresh full-round native testing at actual live queue/deadline settings. Use a predeclared confidence or practical-effect criterion and enough seeds for the observed variance; neither 10 rounds nor 20 pairs has universal adequacy. Do not choose the sample count after seeing a favorable sign.

The workflow improvement is that small ideas are screened cheaply and only coherent, substantial candidates reach that expensive test. Native full-round points remain the deployment criterion.

### 6. One pilot implementation trap we should include

sim_bench:episode constructs a fresh ContinuousController for each snapshot. This drops retained plans, actor history, target/order locks, resource/stall history, and route state. That is not the live policy at that moment. Preserve/reconstruct controller state through a deterministic replay prefix where feasible. Otherwise use a declared identical reset protocol for all arms and call it a reset-state benchmark; verify selected benefits in uninterrupted episodes too. Physical state must include queued inputs and timers. This is a focused validity requirement, not an invitation to rewrite every tool.

### Revised joint sequence I can support

A. Freeze a source/configuration snapshot; perform small all-map determinism, state-reset/prefix, event-conformance and throughput checks. Work on the minimally needed spawn/recovery semantics at the same time. Stop expanding workloads until these checks are interpretable.

B. Run the time-stratified development/held-out pilot with the four arms, a separate gate audit, common-time points, finite work budgets, and planning/evaluation futures separated. Report clear-only diagnostics separately. Choose the next investment from its results: ranker, broader proposals, admission/continuation representation, or runtime efficiency.

C. Complete the validated full-round simulator and use it for repeated search-guided learning. Train the ranker/proposer inside the physics loop first; leave standalone control and points-only RL as later evidence-based decisions. Keep independent native milestone checks.

D. Promote only a candidate with fresh native full-round evidence under actual deployment timing. User demonstrations and observed mistakes feed a representative automated dataset, not a special-case patch queue.

Can you agree to these information, objective, and deployment boundaries? If so, I think our remaining differences are implementation hypotheses that the pilot can answer. I would like your explicit response on whether a legitimate spawn sampler belongs in A or whether you prefer initially limiting B to the known group with long-term value explicitly undecided.

ASTRA FINISHED 2026-10-29 11:43:08 MDT

---

## Fable, turn 3: boundaries accepted with two conditions; spawn sampler goes in A; proposed final text

I accept your four boundaries: planning and evaluation futures separated, common-time points as the primary metric,
gates priced by executed consequences rather than by a new no-fall objective, and native full-round evidence as the
deployment criterion. Two conditions on the last one, below. Then answers to your direct question and a draft of the
final plan for your edit. I think we are one turn from settled.

### Concessions, with what changes

1. **The recorded next group leaks.** You are right, and my stop rule leaned on it. If every branch is scored against
   the one future that actually happened, the "best root action" has seen the future, and exit direction and powerup
   retention are exactly the quantities it would fake. So: rank on an expectation over independently sampled legal futures
   with common random numbers across branches, evaluate the selected action on reserved futures, and report the
   clairvoyant best-of-branches only as an upper bound. This needs the spawn sampler before the pilot's long-horizon
   part. My answer to your question: **the sampler belongs in A.** The pilot's short-horizon part (known group only) can
   start before it and must say "long-term value undecided" until the sampler exists.
2. **Which spawn rules.** The benchmark game and every recorded round run the single-player branch of getCenterGems (one
   marble; the LastGemSpawner exclusion). The multiplayer branch keys on the leader's position and score. Declared target:
   the solo branch, because that is what the native confirmation runs and what the recordings are; the multiplayer branch
   is a documented difference, not a claim. Lobby validity is preserved because the policy never observes a sampled
   future; sampling is evaluation randomness only.
3. **My percentage stop rule was undefined.** Fixed-length continuations all last 20 s. Replace it: the primary quantity
   is realized game points over the common horizon, per situation, aggregated as a ratio of equal-map means, with
   uncertainty clustered by source round and layout. Time to clear the same initial gem set, completion fraction and
   censored cases are separate diagnostics and never converted into round points. Provisional materiality, not a
   statistical rule: ranking goes first if the deployable out-of-sample selector gains at least 5 percent of common-horizon
   points and the clairvoyant bound is at least 15 percent; coverage goes first if the clairvoyant bound over the existing
   candidates is under 8 percent; anything straddling is inconclusive and the pilot is extended, not decided. A flat
   compute curve rejects this generator family at this budget, nothing more.
4. **Gate audit prices falls, it does not forbid them.** Agreed, and it follows the operator's standing rule (points only,
   falls count only through time). Admit candidates the heuristic terminal gates reject when the physics and events are
   supported, execute them, and charge actual recovery time and points. Refuse only incomplete state, invalid controls,
   unsupported mechanics and numerical failure. Log absent / rejected / admitted / lost-on-ranking per candidate so the
   gate is isolated from the generator. This requires minimal respawn semantics in the sim (OOB detection, respawn
   point and delay as the game does them); today sim_bench stops on a fall. That goes in A with the sampler. Until it
   exists, fall-ending branches are reported as unresolved, never labelled with a fixed penalty.
5. **Compute accounting.** All forecasts count: generation, refinements, follow-ups, branch continuations. The ladder is
   per root choice for the diagnostic and per closed-loop episode for anything deployable. A feasibility pass on one
   map and 20 situations at 3x before the 200 starts; dominated budgets stop early. Report native steps, wall time, memory
   and outcome gain together.
6. **Controller state at a snapshot.** Agreed, a fresh ContinuousController drops retained plans, locks, stall history
   and route state. The pilot uses a deterministic replay prefix where the journal allows it (sim_bench's `--lead N` is
   the start of that), declares one identical reset protocol for all arms otherwise, labels the result a reset-state
   benchmark, and verifies any selected benefit in uninterrupted episodes.
7. **Dates.** No calendar promises. Caps: A gets at most three days of my time before a re-estimate from the feasibility
   pass; the pilot at most three more. If A overruns, the operator hears why before day four.

### Two conditions on native promotion

I accept that native full-round points at live timing decide deployment. Conditions:

- **The native test must be sized for the effect, and that needs the deterministic native mode.** Twenty pairs resolve
  about +-3; a +2 effect needs about 110 pairs, which is the bottleneck we are escaping. The `--fixed-lead` mode (budget
  that almost never binds) exists untested; its A/A check (det_aa) was cancelled on 10-29. Validating it is part of A.
  With it, rounds where the candidate never acts score identically and the pairs that differ carry the signal.
- **Only coherent candidates go to native, with a predeclared criterion.** A candidate qualifies when its sim effect is
  at least twice the native campaign's resolution at the declared sample size; the sample size and the pass criterion
  (sign-consistent across seeds, effect above a stated floor, no map loss beyond a stated bound) are written down before
  the run. No single-rule campaigns, no sample size chosen after a favourable sign.

With those two, "native decides" stops meaning "noise decides".

### The baseline snapshot

The working tree today differs from HEAD d2ad44771 (speed cap off, POW_LAUNCH_DIRECT, LAND_SHORT, situation replay
tooling; FAST_SUPPORTED is False in both). The pilot needs one frozen snapshot with a recorded hash. My preference: the
operator says which flags are the baseline, we record the hash, and all arms share it. Either choice is fine for the pilot
as long as it is the same for every arm; what is not fine is "production" as a moving target.

### Draft final plan (edit this; if you accept it, we are done)

**Goal.** Replace rule-by-rule fixes judged on noisy game pairs with a measured, deterministic loop that learns from
executed consequences in the exact engine port, and promotes only coherent candidates by native full-round evidence.

**A. Foundations (parallel; cap 3 days of Fable time, re-estimate after a one-map feasibility pass).**
- Frozen baseline snapshot with a recorded hash; operator picks the flag set.
- Determinism: fixed candidate order and tie-breaks, fixed virtual work budget, A/A check (identical arms match to the
  decision) in the sim; validate the native `--fixed-lead` mode with its A/A check.
- Event conformance: recorded control sequences replayed through the sim on recorded rounds; compare pickup ticks,
  powerup timers, OOB and respawn timing, per-reply state error. Mismatches fixed before any label is produced.
- Minimal game semantics the pilot needs: respawn after OOB, and a spawn sampler following the solo branch of
  huntGems.cs (gem groups, centre choice, radius growth, weights, testSpawn, powerup respawn timers, round clock), with
  distribution checks against recorded rounds. Multiplayer branch documented as a difference.
- Controller state at snapshots: deterministic replay prefix or a declared identical reset protocol.
- Throughput measured, not estimated.

**B. Headroom pilot (cap 3 days).**
- ~200 situations drawn uniformly in round time from held-out rounds, stratified by the ledger buckets, dev/eval split by
  source round and layout, plus a separately reported hard subset.
- Continuations 20-30 s; planning futures sampled with common random numbers across branches; evaluation futures
  reserved; clairvoyant bound reported separately from the deployable selector.
- Arms: (1) frozen baseline at its work budget; (2) same candidates and objective, more deterministic work; (3) existing
  candidates ranked by executed continuations on the common-time points objective; (4) broader multi-event search and
  control refinement on the same objective, on a 3x/10x/30x/100x ladder with all forecasts charged.
- Gate audit: identical proposal unions with and without the heuristic admissibility filter, consequences executed,
  recovery time and points charged, per-candidate reason log.
- Output: one table per arm per bucket of common-horizon points (ratio of equal-map means, clustered uncertainty),
  plus the clear-time diagnostics. Decision by the materiality targets above; a straddling result extends the pilot.
  The operator sees the table before any build.

**C. Build what the pilot points at.** A ranker and then a proposal model trained inside the physics search from
branch labels, judged on held-out ranking and closed-loop sim returns; or admission and continuation representation
work; or runtime efficiency. The validated full-round simulator becomes the screening environment for repeated
search-guided learning. Standalone control and points-only RL remain later decisions on evidence.

**D. Promotion.** Native full-round rounds at live timing, deterministic mode, predeclared sample size and criterion,
coherent candidates only. The operator's situation replays and observed mistakes feed the situation dataset, never a
patch queue.

### Residual disagreements, for the record

- You hold that a flat compute curve says nothing about coverage in principle. True. I hold that we still act on it,
  because the alternative is an open-ended search project; the plan above acts on it and says so.
- You would keep the full-round sim strictly as a screen. I expect it to carry most of the evidence once conformance
  is established and the native confirmations keep agreeing with it, and I would let the native sample shrink as that
  agreement accumulates. That is a later argument with data, not a blocker now.

If you accept the draft, say so and we close; if you edit it, I will take one more turn.

FABLE FINISHED 2026-10-29 11:49 MDT

---

## Astra, turn 3: accept the program, with four final operational corrections

I am satisfied with the central program now: a bounded headroom pilot; minimal spawn/recovery semantics and conformance from the start; select the investment by evidence; then search-guided learning inside the physics loop. The simulator is essential infrastructure for that loop. I no longer insist that a stronger slow teacher must precede the first value experiment. You no longer insist that a value replacement must be the first main build. That is a substantive resolution of our original disagreement.

I accept your answer that the spawn sampler belongs in A. Known-group diagnostics can start earlier, but cannot decide long-term resource or exit value. I also accept the separately reported clairvoyant bound, time-stratified sampling, outcome-priced gate audit, controller-state caveat, and finite compute ladder. Four corrections remain before I can endorse the final text as written:

### 1. Native deterministic diagnosis and live deployment are different tests

Your D currently says 'at live timing, deterministic mode.' The existing --fixed-lead 16 changes the allowed search work: run_flag_native.py computes min(2, max(.05, lead*.064-.02)), giving 1.004 seconds versus 0.172 at lead 3. Its wrapper is documented as direct observations without actual live queue latency. It therefore cannot be the deployment test just because it becomes repeatable.

I accept validating that mode in A. It gives a useful native deterministic/cause-isolation test when A/A confirms it. Final deployment evidence must additionally use the actual realtime queue, wall deadlines and available lead. Scheduling noise is then part of the system being deployed, and we measure over enough independent rounds. If the new controller removes that sensitivity at actual timing, excellent; we must establish it rather than erase it in the benchmark.

So D needs two distinct rows: native deterministic diagnosis, then actual live confirmation. A strong simulator result can justify doing fewer repeated exploratory native tests. It cannot substitute for evidence that the candidate operates within real-time constraints.

### 2. Materiality is an investment decision; don't make clairvoyance a mandatory gate

I support a provisional target around 5% for a deployable held-out improvement, subject to uncertainty and compute cost. But requiring that gain AND a clairvoyant bound of 15% rejects useful rankers for no clear reason: a selector gaining 6% out of an 8% attainable opportunity is a good result. Likewise, a small existing-candidate oracle gain does not by itself establish that broad search will help; arm 4 must demonstrate its own incremental value.

Use the headroom bounds to diagnose the bottleneck and the actual out-of-sample gains to choose the build. On a flat 10x-100x curve, stop THIS search design and redirect; we agree there is no reason to keep tuning it indefinitely. The wording 'ranking first above 5%, otherwise compare measured incremental gains and costs' is enough for planning. Exact effect floors and confidence rules should be frozen after the small feasibility/variance pass, before opening the reserved evaluation set.

Also, 'inconclusive extends the pilot' needs a finite limit. Predeclare one fresh extension or an appropriate sequential rule within the total compute/time cap, then report unresolved. Repeatedly inspecting and expanding the same held-out set until it passes reproduces our existing selection problem.

### 3. Do not require a new user decision just to freeze the baseline

We can choose and document the actual current agreed checkout/flags when implementation begins, archive the exact sources and runtime configuration, and remeasure it. Historical scores remain labeled with their original versions. A baseline does not need to be the global best to support matched causal comparisons. The user can override a substantive baseline preference, but we need not block on asking them to adjudicate flags.

For the record, my fresh check at 11:45:48 MDT read HEAD d2ad44771, git status containing only untracked CONVO.md, cap 11, FAST_SUPPORTED=False. I will not infer further working-tree changes from earlier notes. A new source snapshot during implementation resolves the moving-target issue.

### 4. Conformance needs declared tolerances and an honest scope

Replay exact inputs to compare event ticks and chained physical states, with predeclared tolerances and explicit coverage of the regimes used for labels. Known material mismatches block the affected labels or cause a repair; they need not block all useful work on unrelated validated regimes. 'Any mismatch fixed before any label' could become an indefinite exactness project given the existing small contact discrepancies.

The initial environment validates the solo ten-map benchmark. It preserves legal information boundaries, but we should not call multiplayer performance or mechanics validated. The policy may use the known rules to average possible futures; it must never condition on the actual hidden spawn RNG. Those are compatible statements.

### Settled work program, if you endorse those corrections

1. **Bounded feasibility and validation.** Snapshot the current baseline. Check sim A/A, candidate/worker semantics, checkpoint-versus-reset state handling, and relevant event transitions. Build the necessary solo spawn sampler and real recovery semantics. Validate native deterministic diagnosis separately. Measure throughput/memory on a small development sample. Re-estimate the implementation cap from that measurement; your provisional three-plus-three-day cap is a stopping/review bound, not a delivery promise.
2. **The headroom pilot.** About 200 representative all-map states with round/layout-separated development and reserved evaluation. Compare ordinary budget, extra work, longer-return ranking, and broader sequence search, plus the isolated gate audit. Use common-time points, separate planning/evaluation futures, and a charged compute curve. Keep clear-time and oracle-headroom diagnostics separate. Allow only a predeclared bounded extension if inconclusive.
3. **Build the winning source of improvement.** Learn ranking/value and diverse proposals inside the exact-physics planner when supported; otherwise focus on the demonstrated proposal/admission or runtime limitation. Complete full-round simulation for repeated improvement. Training data includes representative new-controller states, improved trajectories, and losing alternatives. Standalone control/RL is a later experiment.
4. **Confirm whole-round gains in the deployed system.** Use the simulator and native deterministic mode to screen and explain. Use fresh full native rounds at actual live timing to establish deployment gains, with sample size/criteria fixed in advance and all ten maps reported. No single replay fix becomes a production campaign by default.

This is a plan for a self-improving system rather than another list of maneuver fixes. It preserves the useful existing architecture while testing which component actually limits performance.

Please explicitly accept or amend these four corrections. If accepted, I consider the debate settled and will append a short closure for the operator. We have not authorized or started implementation in this conversation.

ASTRA FINISHED 2026-10-29 11:47:18 MDT

---

## Fable, turn 4: all four corrections accepted; settled

1. **Two native rows.** Accepted. The `--fixed-lead` mode widens the search allowance (about 1.0 s against the live
   0.172 s) and bypasses the real-time queue, so it is a cause-isolation test and nothing more. D has two rows: native
   deterministic diagnosis to explain a candidate, then fresh full native rounds at actual live timing with the queue,
   wall deadlines and available lead to establish deployment. Scheduling sensitivity is part of what is deployed and is
   measured, not benchmarked away. My sizing condition survives in the form you gave it: coherent candidates only, sample
   size and criteria fixed in advance, all ten maps reported.
2. **Materiality as an investment decision.** Accepted. The clairvoyant bound diagnoses the bottleneck and is never a
   gate; the deployable out-of-sample gain chooses the build (ranking first above about 5 percent, otherwise compare
   measured incremental gains and costs); arm 4 must show its own incremental value; a flat 10x-100x curve ends that
   search design. Effect floors and confidence rules are frozen after the feasibility and variance pass and before the
   reserved evaluation set is opened. An inconclusive result gets exactly one predeclared fresh extension within the cap,
   then is reported unresolved.
3. **Baseline freeze without a user decision.** Accepted: snapshot the checkout and flags in use when implementation
   begins, archive sources and runtime configuration, remeasure it, keep historical scores labelled by their versions.
   The operator can override. One thing for the record: our two checks disagree. My session's git status at its start
   listed modified gem_demo.py, realtime.py, reliability_eval.py, search.py, test_search.py, play_realtime.ps1 and
   sim_bench.py plus the untracked situation tools, and my read of gem_demo.py showed TARGET_SPEED_CAP=math.inf with a
   10-29 comment; yours at 11:45 showed a clean tree at d2ad44771 with cap 11. We are probably reading different
   worktrees or the tree changed between reads. Either way the snapshot rule resolves it: whichever tree implementation
   starts in is hashed and archived, and the operator is told which flags it contains.
4. **Conformance with tolerances and scope.** Accepted. Exact-input replays compare event ticks and chained states against
   predeclared tolerances, with explicit coverage of the regimes used for labels. A material mismatch blocks the affected
   labels or triggers a repair; validated regimes proceed. The environment validates the solo ten-map benchmark; the
   policy may average over legal futures and never conditions on the hidden spawn RNG; multiplayer mechanics are not
   claimed.

I endorse your settled work program (1 bounded feasibility and validation, 2 headroom pilot, 3 build the winning source
of improvement, 4 confirm whole-round gains in the deployed system) as the final plan. Nothing has been implemented or
authorized in this conversation; the operator decides whether and when step 1 starts.

Summary of what each of us gave up, for the operator: I dropped learned value as the first build, journal-only labels,
the recorded next group as an oracle, a percentage-of-time stop rule, and simulator-only promotion. Astra dropped
requiring a stronger slow teacher before any value experiment, the pilot before any simulator work, and failure-selected
situations, and accepted the compute ladder with a stop rule and the spawn sampler in stage 1. The shared conclusion: the
limit is not a list of behaviours; it is that nothing in the system learns from executed consequences and nothing in the
test loop can see a small change. Stage 1 and 2 measure which component limits points before anything large is built.

Please append your closure. I consider the debate settled.

FABLE FINISHED 2026-10-29 11:53 MDT

Correction (Fable, 11:55): the worktree is now clean at d2ad44771 ("Before next architecture change", committed 11:25). The
uncommitted speed-cap-off edit in gem_demo.py I read at the start of my session was discarded by that commit; HEAD has
TARGET_SPEED_CAP=11 and FAST_SUPPORTED=False. Astra's reading was the current one. The baseline snapshot is d2ad44771.

FABLE FINISHED 2026-10-29 11:55 MDT

---

## Astra, final closure: joint plan agreed

I accept Fable's turn-4 endorsement. Both participants are satisfied that the following is the best next work program given the available evidence. The debate is closed; no further acknowledgement is required.

### Decision

Build toward automatic improvement from simulated and executed consequences, retaining the existing physics planner. First run a bounded experiment to identify whether the largest opportunity is better ranking, broader proposals, heuristic admission, or computation. Neither a value-network replacement nor a large slow-teacher training campaign is selected in advance.

### Agreed sequence

| Stage | Concrete deliverable | Decision it enables |
| --- | --- | --- |
| 1. Feasibility and validation | Frozen source/configuration baseline; repeatable sim A/A; declared controller-state handling; relevant event/physics conformance; minimal legal solo spawn sampling and recovery; measured throughput/memory | Whether the pilot can produce trustworthy comparisons, and at what cost |
| 2. Headroom pilot | Approximately 200 representative all-map situations, development/evaluation split by source round/layout; four arms and a separate gate audit; common-time points and a compute-to-quality table | Which component deserves the main investment |
| 3. Evidence-selected improvement loop | Ranker/value and diverse proposals inside physics search, or the demonstrated proposal/admission/runtime improvement; validated full-round simulator; repeated play/search/learn cycles | Whether better decisions accumulate and generalize across the ten maps |
| 4. Deployment confirmation | Fresh full native rounds under actual live queue/deadline conditions, with predeclared criteria and all ten maps reported | Whether the candidate improves the system the operator actually watches |

The four pilot arms are: baseline at ordinary work budget; same planner with more work; existing proposals ranked through longer executed continuations; broader multi-event search/refinement with that same points objective. The separate gate audit evaluates physically supported candidates excluded by heuristic filters and prices their actual consequences, including recovery.

### Non-negotiable interpretation rules

- Future randomness used to choose actions is separate from reserved evaluation futures. A recorded actual next group cannot secretly choose the current action. Clairvoyant scores are diagnostic bounds only.
- Actual game points over a common horizon are primary. Clear time, failures, speed, and inventory are explanations or separate diagnostics. They are not substituted for points.
- The initial simulator validates solo benchmark rules. No claim of validated multiplayer mechanics follows automatically.
- Identical reset-state tests are labeled as such; uninterrupted execution is checked before accepting their apparent advantage.
- Every forecast and continuation is charged. A small feasibility pass precedes the 3x/10x/30x/100x ladder; the ceiling is a cap, not a quota. Flat or dominated designs stop.
- Around 5% deployable held-out improvement is a provisional investment target, not a promise or a mandatory clairvoyant-headroom gate. Freeze effect/confidence criteria after the feasibility pass and before inspecting reserved results.
- An inconclusive pilot gets at most one predeclared fresh extension within the cap; otherwise report it unresolved. Do not tune repeatedly on the reserved set.
- Native fixed-work/long-lead diagnosis and actual live confirmation are separate. Repeatability gained by giving the planner more time is not proof of deployable performance.

The current checked baseline is commit d2ad44771, with cap 11 and FAST_SUPPORTED=False. Implementation begins by archiving and remeasuring the exact checkout/runtime configuration then in use; old result tables retain their original source identities. No extra user decision about routine flag bookkeeping is required.

The initial three-day foundation and three-day pilot bounds discussed above are provisional review/stop caps, not delivery estimates. Re-estimate after measured feasibility. There is no open-ended infrastructure or search project hidden behind this plan.

### What changed through the discussion

Fable moved from full-simulator-then-value-first to competing hypotheses with counterfactual labels and native deployment evidence. I moved from a stronger-teacher prerequisite toward an early value/ranking experiment, and accepted that legal spawn sampling and recovery belong in the initial validation stage for long-term questions. We jointly resolved leakage from recorded futures, ambiguous time-based metrics, reset artifacts, compute accounting, and the distinction between deterministic diagnosis and live success.

No gameplay code, training run, or experiment was started during this discussion. The completed deliverable is this agreed plan and the reasoning that produced it.

ASTRA FINISHED 2026-10-29 11:50:21 MDT
