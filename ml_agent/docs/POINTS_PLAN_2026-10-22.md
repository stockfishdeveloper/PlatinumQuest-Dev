# Points plan (operator approved 2026-10-22, about 11:15)

Written so that any agent can pick this up cold. Read ml_agent/docs/HANDOFF_NAV_TRAINING_LOG.md (tail) and the memory
index for the standing rules; the ones that matter most here: points only (falls never count), settings baked into code
defaults (flags only as matched controls), must transfer to any map (no map names in logic), never edit anything under
nav/marble_dynamics while a reliability_eval pool runs (the pools hash those files), equal load for any pair
(4+4 workers, both starts launched together), estimates are agent time with measured run durations.

## Why this plan

- Reference: about 130 points a round pooled over the ten real maps (v86 defaults, edge hits and v23 wall rebounds on).
- Humans on the same maps: roughly 170 to 210 (TreasureBox demo 207 against our 165 to 178; KOTM 173 against about 150).
- The gap is time per gem inside a spawn and time between spawns, not misses. About 25 spawns a round: 0.5 s saved per
  spawn is about 12 s, 8 to 10 points.
- Known waste (journals, demos): Blasts gaining under 1 u (32 of 52 in the cluster audit), pad-wall bumps, corner stalls,
  braking into every gem (7.7 u/s at pickup against the human 11.4), 1.6 gap jumps a round against the human 12 to 15.
- Six cluster variants (spawn_plans.py chain plans, pad hops, pad arrival, through credit fix) all lost in ten-map pairs.
  They were picked by anecdote. The order below starts with measurement so the next change is aimed at the biggest
  bucket.

## Step 1: points ledger (measurement, first)

Tool: ml_agent/tools/points_ledger.py (new, outside nav/marble_dynamics so it never breaks a running pool).
Input: pool journals dynamics_reports/<pool>/<map>/<seed>/physics_plan/<n>/decisions.jsonl.gz (one row per 64 ms
decision: raw_observation, before.native.state, choice.label, outcome.gem_delta) and the human demos
ml_agent/demos/demo_*.npz (16 ms ticks with positions and gem events).
Output per round and pooled per map: seconds in each bucket, points per bucket, and the bucket's share of the gap to the
human demo where one exists (TreasureBox 207: demo_20261021_151105.npz; KOTM demos in ml_agent/demos).

Buckets (every decision falls in exactly one):
1. approach: from the last pickup of a spawn to the first pickup of the next spawn (a new spawn = gap > 47 decisions
   between pickups, as travel_split.py defines it), split into moving (route progress > 0.5 u/s) and stalled.
2. in-spawn: between the first and last pickup of a spawn, split into moving and stalled.
3. stalled: no route progress for 1.5 s (24 decisions) anywhere; also report where (near a pad wall: a vertical face
   within 1 u; in a corner; under a ledge).
4. respawn: OOB flag set until the marble moves again.
5. powerup detours: decisions whose choice label starts with opportunistic_powerup or whose goal is an item.
6. endgame: the last 10 s.
Also per round: Blasts fired and the height each gained (blast_cf.py has the counterfactual sim), pickup speeds,
exit speed from the last gem of each spawn, gap jumps used.

Comparison: the same buckets on the human demos (pickups from the gem events, spawns by the same gap rule).
Deliverable: a table in HANDOFF_NAV_TRAINING_LOG.md and one line per map in this file under "Ledger results".
Agent time: 2 to 3 hours (writing the tool, running it over the latest two ten-map pools and the demos).

## Step 2: score plans by time

Where: nav/marble_dynamics/gem_demo.py candidate_cost and the landing/ballistic extensions in search.py.
Today a plan is scored by where it ends (route distance plus 0.2 x straight distance plus 0.12 x end speed), a pickup
is -100 plus 4 u/s x pickup time, and plans of different lengths (48 replies against 88 with a landing extension) are
compared on end distance alone. Change: every plan is scored as route progress per second over a common time, so a
5.6 s plan never beats a 3.5 s plan on end position, and a Blast only wins when it saves time against the best
non-Blast plan. Screen offline first: sim_bench --whole-group over the recorded starts (30 to 60 min), then one
ten-map pair (45 min, 4+4 workers). Keep only on a positive equal-map total.

## Step 3: the spawn as the planning unit, cheaply

Where: reliability_eval.py order_choice / chain_order (CHAIN_ORDER_S) and spawn_plans.py (order solvers already exist:
nn, straight, aligned, sweep, topdown, rows). Change: solve the gem order once per spawn with pad-aware costs (down
hops free, up hops priced by hop time, a row that lines up priced as one jump), commit to it, and let the existing
families chase gem k with through credit on gem k+1. Do NOT drive the chain open-loop (that is what the six losing
variants did). One pair after the offline screen.

## Step 4: never stand still

Where: reliability_eval.py pick / safe hold. A stall rule: no route progress for 1.5 s switches the target to the next
gem of the order or backs off 3 u and re-runs the approach at speed. Covers the corner-sitting round and the pad-wall
bumps. One pair.

## Step 5: exit the last gem of a spawn at speed toward the likely next spawn

The next spawn is unknown but constrained (30 u or more away, only real spawn points: terrain.gem_spawns). Aim the exit
through the last gem at the centroid of the reachable spawn set instead of braking into it. Where: the through
context for the last gem (search.py last_gem / keep_ss area, engine.next_for). One pair.

## Step 6: Blast priced by seconds saved

Falls out of step 2: a Blast plan must beat the best non-Blast plan by a time threshold (start at 0.8 s). If step 2 does
not cover it, add the threshold in search.py where the blast specs are added (blast_available ... blocks).

## Wall climbs (parked, in the code, untested in a pool)

State at 11:30 on 2026-10-22, all in the working tree (uncommitted):
- crossings.py: WALL_CLIMBS (set False at 11:45 to isolate the first pair; True stages a run-and-jump at a vertical face whose rebound lands on the higher level
  (_wall_crossing, chosen by a time estimate, lane corridor and turn penalty).
- reliability_eval.py pick(): a staged wall climb reserves the Blast for the whole decision (engine.wall_staged) and
  skips the cluster-order pass; engine.blast_reserved also set inside the crossing loop.
- search.py: a wall-climb plan still airborne at the horizon gets the landing extension (WALL_AIRBORNE_VZ); plans toward
  a staged wall climb's landing pay WALL_TIME_RATE_U_S (8 u/s) per reply beyond the horizon.
- Sim evidence (scratchpad wall_trace.py, TreasureBox snapshot 2290 moved to (105,-10.2) at 8 u/s): the x118 wall plan
  with trigger 3 u at 16 u/s jumps at reply 17, peaks z 6.96, is on the deck by reply 39 (2.5 s). Not yet shown to win
  a full planner episode; no pool pair yet.
To finish: one ten-map pair, default against WALL_CLIMBS=False (edit the constant for the control arm only via a copied
module, never while a pool runs). Keep on a positive equal-map total, else set WALL_CLIMBS=False.

## Validation protocol (every step)

1. Offline: sim_bench --whole-group on the recorded starts (deterministic, 30 to 60 min).
2. Pool pair: ten maps, two starts, 4+4 workers, both arms launched together (scratchpad launch_pair.ps1 does this;
   copy it to ml_agent/tools if it is still only in the scratchpad), about 45 min. Judge the equal-map total; a Monitor
   with a 5 min stall alert on every wait.
3. Log every result in HANDOFF_NAV_TRAINING_LOG.md with the time of day, and the per-map numbers.
4. A kept change becomes the code default; the control flag stays only as a matched control.

## Ledger results (11:40, tools/points_ledger.py over wallclimb1 + spawn6 arms = 80 rounds, 4 distinct per map)

Per round, mean over the ten maps (240 s rounds): 129 pts, 17 spawns, 95 gems. Approach (between spawns) 139 s moving
+ 3 s stalled; in-spawn 95 s + 0.3 s stalled; 4 s pre-round neutral; tail 5 s. 8.3 s per approach, 1.28 s per gem
inside a spawn. Pickup speed 7.0, exit speed 6.9 u/s, 12.8 Blasts.
- TreasureBox: planner 181 vs human 207. Human 19 spawns / 153 gems vs 16 / 128. Approach 6.58 vs 7.90 s each
  (21 s a round); in-spawn 0.82 vs 0.92 s per gem (11 s). The human picks up SLOWER (5.0 u/s) and exits slower
  (4.7 u/s): braking into gems is not the gap.
- Approach legs (scratchpad approach_legs.py): the planner's pace on a leg is 10.3 u/s (TreasureBox) / 8.3 (Concentric)
  against the human's 13.4 / 12.7, with the same path/straight ratio (1.46 vs 1.29; 2.04 vs 1.96). Same routes,
  30 percent slower. Vortex and Bowl have 110 s a round in legs over 12 s (wandering: path 300-400 u for 30-95 u
  straight).
- Moving AWAY from every gem (nearest-gem distance up > 3 u over 1 s): Bowl 45 s a round, KOTM 32, Concentric 29,
  Vortex 22, Duplex 18; TreasureBox 5.
- Powerup goals (choice.goal_item_id is a powerup): 5 to 35 s a round per map (mean about 23 s); 114 of 1571 fetches
  were farther than the nearest gem (336 s over 80 rounds). The worst: Bowl, right after a pickup while dropping
  into the bowl, the search found nothing useful toward the next gem (32 u away) and the nothing-found fallback
  fetched a Helicopter 161 u away, 10 s out and 10 s back, in every Bowl round (both seeds).
- Stalls are small overall (3.3 s a round) except Concentric 11.6 s (direct_recharge_* plans: waiting under a ledge
  for the Blast to charge, 4-7 s each) and Cragmire 7 s (direct_jump_9 into the same wall at (0,13,13)).
- Human references available: TreasureBox 207 (demo_20261021_151105), Concentric demo rounds 44/62/99 (incomplete
  rounds in demo_20261021_084144; the clock column obs_raw[:,31] splits them), KOTM 153 (demo_20261003_211632).

First change from the ledger (11:45): powerup fetch cap. reliability_eval RESOURCE_FETCH_MAX_U=25 (nothing-found
path: the item must also be no farther than the gem), RESOURCE_STALL_FETCH_MAX_U=40 (stall path), resources.
choose_resource(max_distance). Control flag --no-resource-cap. Pair rescap1 launched 11:47 (baseline = cap ON =
default, candidate = --no-resource-cap control). Expected about 12:35.

Next from the ledger, in order: (a) the nothing-found situation itself (momentum carrying the marble away: no plan
makes progress within 3 s, so the fallbacks pick a goal switch; the right answer is the plan that loses least, i.e.
turn and brake toward the gem); (b) approach pace (10 vs 13 u/s on the same routes: END_VELOCITY_WEIGHT, the
'fast' families, braking at gems the human does not do); (c) Concentric recharge waits; (d) the Vortex / Bowl
wandering legs (leg_trace.py in the scratchpad prints one leg per second).

### More ledger numbers (12:25, same 80 rounds; scratchpad pace_profile.py / fast_reject.py)

- Speed after leaving a spawn (legs over 35 u): planner 7.8 u/s at 1 s, 10.6 at 2 s, about 11 from then on;
  human 9.0 at 1 s, 14.7 at 2 s, 18.6 at 3 s (TreasureBox), 13-15 on Concentric. Time on legs above 13 u/s:
  planner 21 percent, human 46 percent. TARGET_SPEED_CAP is 11 (TRAVEL_TOP 13.5 ramps in beyond 12-20 u); the fast
  families (16/20) exist and do win sometimes (2 of 8 probed travel states on TreasureBox), direct_fast is mostly
  rejected (no_supported_continuation: it leaves the floor), fast_16/20 rank 7th-18th when a steering or Blast plan
  wins on route distance. This is step 2 material (time-scored plans), not a knob.
- Fallback seconds a round: frozen navigator 10-36 s (Bowl 36, KOTM 36, Fractured 33, TreasureBox 32, Vortex 30);
  opportunistic powerup goals 5-28 s; recharge waits Concentric 10 s; safe hold / A2 under 6 s.
- Opportunistic detours: 6-30 episodes a round, most ABANDONED (Cragmire 22 of 28, Fractured 29 of 30, Sprawl 18 of
  30, TreasureBox 11 of 19); the collected powerup is used within 2-7 s. Cheap pair: --opportunistic-powerups 0
  (control) against the default 4, then if the feature pays, make a started detour commit instead of flip-flopping.

### Pair log
- rescap1 (12:30): powerup fetch cap. cap 130.60 vs no-cap 131.10 equal-map (-0.5, noise). Kept: the nothing-found cap
  (25 u, no farther than the gem). Dropped: the stall-path cap (Cragmire -10 twice). Lesson for every pair: a single
  round diverges by +-10 points from one changed decision; twenty rounds an arm resolve about +-1.5 a round. A
  change worth under 2 points a round needs two pairs (four seeds) before a verdict.
- oppo1 (launched 12:33): default detours (4 u) vs --opportunistic-powerups 0.

### Step 2 spec (written 12:45 while oppo1 runs; not implemented yet)

Today's cost (gem_demo.candidate_cost): a plan without a pickup is scored by where it ends, route distance + 0.2 x
straight distance + 0.12 x end speed; a pickup is -100 + 4 u per second of pickup delay (PICKUP_TIME_WEIGHT) + the
post-pickup progress term read at reply 24. Only ballistic projections pay for time beyond the horizon (8 u/s).
Two consequences the ledger shows: time is priced at 4 u/s while the marble travels 10-13 u/s, so a slower plan
that ends a little nearer wins; and extended forecasts (landing continuation 40 replies, crossing phase) are judged
on their end point after 5-6 s against 3 s plans.

Spec: price every plan in SECONDS of estimated time-to-go, same for all families and lengths:
  cost_s = t_end + D_route(end) / V_REF - V_SPEED_S x (v_end - V_REF)_+   (no pickup)
  cost_s = t_pickup + post_term / V_REF - BONUS_S                          (pickup; post_term as today, in u)
with V_REF = 10 u/s (the measured cruise; the human's 13 is the target once the gates allow it), V_SPEED_S about
0.05 s per u/s of end speed above V_REF (carried speed is future progress; v17's 0.3 u credit raised falls, so keep
it small), BONUS_S = 10 s (the old -100 u at 10 u/s). Everything downstream that compares costs keeps working
(useful_candidate, order_choice totals, chain_score bonus, WALL_COST_MARGIN_U, time_penalty_u) once their units are
seconds; audit each constant (grep "cost" in search.py, landing.py, spawn_plans.py, edge_hits.py, crossings.py).
Screen: sim_bench --whole-group over the recorded starts (30-60 min, NOT while a pool runs: the pool's 2 s budget is
wall clock and a CPU-heavy sim batch would slow one arm), then one ten-map pair, then a second pair with two new
seeds if the first is within +-2 a round.

Rule learned 12:30: never run a sim batch or anything CPU-heavy while a pair runs; never edit under
nav/marble_dynamics while a pair runs (a second session did at 12:32 and both arms went invalid in 13 s).
Step 2a (minimal version, same effect in u instead of s, to run first): keep units in u, price time at V_REF = 10 u/s
everywhere: PICKUP_TIME_WEIGHT 4 -> 10 (v51 tried 8 once: -2, noise), and every forecast longer than the horizon
(landing continuation, crossing phase, pickup landing) pays (replies - steps) x 0.064 x 10 u like the ballistic
projection already pays 8 (search.py near 'time_penalty_u' and the landing-extension block; the wall-time penalty
added 2026-10-22 becomes a special case of it). END_VELOCITY_WEIGHT stays 0.12 for this pair. Cost sites to keep
consistent (all in u today): search.py 314 JUMP_COST_U, 476 (-150 pickup special), 841 ballistic penalty, 1038/1040/
1055/1384 BACKUP_PENALTY_U 150, 1111 penalty, 1216 blast_charge_u, 1355 WALL_COST_MARGIN_U; spawn_plans chain bonus;
gem_demo 168-173.

### Offline findings from the second session (platinumquest-dev-2e, 12:55; scripts in its scratchpad
C:/Users/doug/AppData/Local/Temp/claude/c--Users-doug-OneDrive-Documents-GitHub-PlatinumQuest-Dev/90fdb797-5714-4662-b63f-fabc633d35e8/scratchpad/)

- frozen_navigator is NOT a fallback: it is the learned navigator's rollout, physics-checked and chosen on cost like
  any family. 24 s a round in three situations: (A) in-spawn with a predicted pickup 8.9 s (KOTM 33 s!), slower
  than the others (3.9 vs 5.4 u/s progress); (B) long travel between spawns 7.5 s (Bowl 29, Vortex 18), where it is
  the FASTEST family (10.7 vs 8.7 u/s): the hand-built families cruise slower than the network; (C) the last metres
  to the first gem of a spawn 7.2 s, 2-decision flickers, slower than the others. Step 2 must not push it out of (B)
  without the fast families taking over; (A) and (C) are where a time-priced cost should drop it.
- Concentric recharge waits (10.9 s a round, 22 episodes in 8 rounds): the wait plan targets a FULL meter (planned
  21 s), the stall rule cuts every wait at 5 s, the Blast fires inside the wait 0 of 22 times; the meter was already
  at or above the required level in 9 of 22 at the start, and after the cut a Blast fires within 3.8 s in 13 of 22
  and the gem is taken in 21 of 22 (median 7 s after the wait began). Nothing else collectable within 15 u in 19 of
  22. Fix (small, about +0.5 a round): size the wait to the meter the Blast plan needs (several meter levels, the
  shortest wait whose forecast collects) or let ordinary Blast plans replace the wait once value >= required
  (recharge.choose_recharge: wait=(1-value)*charge_ms/64+2; the retained wait is never re-searched because it is a
  pickup plan). Queue: after step 2a.

### Second session, 13:05: wandering legs and abandoned detours (scripts wander_legs.py, wander_switches.py,
detour_ends.py, detour_last_plan.py in its scratchpad)

- Vortex long legs (110 s a round): flip-flopping between targets 28 percent (30 s a round), lost 9 percent,
  toward a powerup 10 percent, falls under 1 percent. 48 target switches a round inside long legs; stage->gem 9.8 a
  round with 83 percent returning to the previous target within 5 s (crossing staging flicker), gem->gem 8.0 (52
  percent return), gem->powerup 7.9, powerup->gem 7.5 (67 percent return). One leg: 40 s, 95 u straight, 320 u of
  path, 35 switches.
- Bowl long legs (104 s a round): 73 percent genuine long travel (targets 95 u median), the Helicopter fetch (gone
  with the cap), and the powerup<->gem detour flip-flop (12.5 percent).
- Opportunistic detours: 17.8 a round, 59 percent abandoned after a median 0.19 s (3 decisions); 76 percent of the
  abandoned ones still predicted the powerup pickup in their last plan and 62 percent re-target the same item within
  5 s: the detour budget test in pick() (extra walk <= 4 u, median 0.66 at the drop) or its search flips between
  decisions. Fix if detours are worth keeping (pair oppo1 decides): hysteresis, a started detour keeps its item while
  its plan predicts the pickup, budget 8 u once started.
- New step for the plan (between 4 and 5): TARGET COMMITMENT. A target switch (gem, crossing stage, powerup) needs a
  margin over the current target and no switch back within 5 s; the crossing staging must not alternate with the raw
  gem goal between decisions. Vortex alone has 30 s a round in flip-flop.
- oppo1 (13:17): detours on 129.60 vs off 129.30 (tie). Detours stay. Both ledger knobs were ties: the per-round noise
  (+-10 from one changed decision, +-30 on Concentric/Cragmire) means only changes touching MANY decisions can show
  in a pair. Next: step 2a (time pricing), which touches every decision.

### Session 2e (operator 17:00: "do a, then b, then c")
- tp1 (13:56, evaluated 17:00): time price ON 127.45 vs OFF 130.10 (-2.65), one TreasureBox round 113 vs 187 is the
  whole loss (the other 19 rounds +1.1 a round for time price); falls 5 vs 11.
- A: tp2 launched 18:33, same arms, seeds 4420117 / 9083321. Keep time price only if tp1 + tp2 together are positive.
- B next: target commitment (offline flip-flop cost 6-12 s a round, two thirds general, one third crossing staging).
- C after: travel speed cap (TARGET_SPEED_CAP 11; TRAVEL_TOP 13.5; human legs 13-18 u/s).
- A done 19:15: tp2 time price ON 129.35 vs OFF 128.20 (+1.15); tp1+tp2 40 rounds an arm 128.40 vs 129.15 (-0.75;
  TreasureBox -22.5, Sprawl +12.8). Not kept: --time-price is opt-in, default off. Step 6 (Blast priced by time) is
  therefore not "free" any more; revisit after B and C.
- B (target commitment) screened offline 19:30-19:44: bundle 1.015 slower, each part alone 1.003-1.009, more episodes
  worse than better. Not pursued; no pair. The flip-flops measured in the journals are not a cause of lost time.
- C screened offline: cap 13.5 + TRAVEL_TOP 16 = 2.6 percent faster per group + trip (cap alone 0.8 percent). Second
  round with higher travel tops running; then one 4-seed pair (40 rounds an arm).
- C pair speed1 launched 19:55 (four seeds, 40 rounds an arm): cap 13.5 + TRAVEL_TOP 16 (new defaults) vs --speed-cap 11
  --travel-top 13.5. Higher tops screened offline and not better (TRAVEL_TOP 20: 0.982; cap 15 + 18: 0.987).
- C done 21:07: pair speed1 (40 rounds an arm) 128.65 new vs 128.80 old (-0.15, se 2.3). Not kept, old speeds restored.
  Same map pattern as time pricing: Sprawl up, TreasureBox / Duplex down. Open question: speed by context (open
  ground vs clusters / edges), and an evaluation that can see 2-point changes (a 40-round pair cannot).
