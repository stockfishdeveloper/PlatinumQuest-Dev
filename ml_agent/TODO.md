# Todo list (started 2026-10-19)

## 1. Wall rebounds: keep momentum through gems by using walls to turn

Status: built, switched off (rolled back 2026-10-19 21:47). The code stays in place behind two switches:
`search.WALL_BOUNCE` and `reliability_eval.WALL_TOUR` (both `False`).

What exists:
- Bounce plans (`nav/marble_dynamics/wall_bounce.py`): carry full speed into a wall aimed so the rebound heads for the next
  gem, then roll straight at it. Used when the marble moves fast away from its target, and for a row that ends at a wall
  with the next gem behind.
- Wall-aware gem order (`reliability_eval.wall_turn_cost`): a turn at a gem with a wall close behind it costs only the
  turn left after the rebound, so the order can be built around using a wall.
- Exact pickup-box switch for the ordinary through plans (`search.THROUGH_BOX`, compiled controller ABI 9), also off.

What we know:
- Offline (120 recorded starts, 10 maps): 0.07 s faster per gem group, 6 better / 2 worse (Sprawl -0.35 s, Cragmire
  -0.22 s).
- Live real-time rounds with it on: Cragmire 94 (4 stuck respawns), TreasureBox 126, 165, 157 (1-2 stuck respawns each),
  below the pools (TreasureBox 174-196).
- Found along the way: plans count a gem as passed only within 0.6 u of its centre, but the game collects it ~1 u out,
  so the marble can circle a gem it already has.

Next:
- Find out whether the walls cause the stuck respawns: live rounds with walls off vs on, same maps.
- Fix the circling-after-pickup problem in the ordinary plans (the pickup-box switch helped TreasureBox and Vortex but
  hurt Cragmire offline).
- If it holds up, a 30-round in-game test before turning it back on.

2026-10-19 follow-up: the box switch was repaired to use actual 16 ms native tick sweeps, with 48/48 compiled/per-reply checks. The ten-map native screen nevertheless lost points (121.4 versus 123.7; 3 versus 1 falls), so it stays off. Bounded generic trajectory refinement also lost its native screen and stays off. See `docs/HANDOFF_REALTIME_PLANNER_2026-10-19.md` for the full per-map results. Do not treat the successful pickup-mechanics check as a score improvement.

## 2. Mega Marble usage

Status: measured and modeled experimentally on 2026-10-19; production Mega proposals remain disabled. See `docs/HANDOFF_REALTIME_PLANNER_2026-10-19.md`.

Why: the planner only considers using powerups 1-5 (Mega is type 6), and its physics forecast refuses Mega because the
game grows the marble through a server script (about 3.5x the radius, after a short delay, for 10 s; 5 s in competitive
Hunt) and that change is not modelled.

Completed evidence:
- Observed activation/expiry, radius, evaluated normal/Mega physics profiles, and grounded/airborne impulses on Bowl and Concentric. The model wraps compiled physics with measured script events.
- Fresh five-decision replay: 150/155 windows within declared practical tolerances, zero pickup-points mismatches. Five Bowl contact divergences remain; do not enable proposals on that evidence alone.
- Requested competitive mode in the local test still produced a 10-second effect. Multiplayer 5-second behavior has not been validated.

Remaining before enabling:
- Resolve chained contact divergence and validate the actual multiplayer duration.
- Verify radius-dependent traversal/clearance and resource accounting, then add delayed/current use proposals behind a flag and require native score evidence.

Possible gain: a bigger marble collects gems from ~0.5 u further away and rolls over small gaps. Risks: narrow ledges.

2026-10-20 13:15 follow-up (flag `--mega-proposals`, default off; pools, real-time runner, realtime_bench and sim_bench
all take it):
- Built: a held Mega gets the same use families as powerups 1-5 (search.py), and any plan that fires it, or runs while
  it is active, is forecast with the measured script-event model (bridge.mega_relevant -> script_forecast.predict_events:
  growth +16 ms, +6 u/s hop +32 ms, 10 s expiry, shrink). Plans that do not fire it keep the exact compiled forecast.
  Unit test: test_search.MegaProposals.
- Measured facts from the pools' inventory telemetry: the Mega profile is SLOWER (max roll 12 vs 15 u/s, angular accel
  60 vs 75, braking 25 vs 30, max_dot_slide 0.3 vs 0.5); radius 0.667 vs 0.190. The planner holds a Mega unused for
  ~900 decisions (~58 s) a round on Sprawl and Concentric, 130-340 on KOTM/Cragmire/Vortex/Bowl, 0 on Cube/Duplex/Fractured.
- Offline (sim_bench, 8 Sprawl held-Mega whole-group starts): the use plans forecast fine (radius grows, pickups
  predicted, cost about -94) but mostly fail the stability gate because the big marble is still bouncing 1.5 s after
  the activation hop, and the one that passes loses on cost to the ordinary plan. No use plan was ever selected.
- Cost: the Python per-tick event forecast is ~10x slower than the compiled one (20 s vs 2 s CPU per episode when
  every forecast went through it); with the narrowed routing only use plans pay it.
- Next (if picked up again): decide what a Mega is FOR. On these maps the physics say reach/gap tolerance, not speed;
  a use only pays on a leg with a gap the small marble cannot roll over or a dense cluster where +0.48 u reach saves a
  turn. That needs a Mega-specific candidate (use + straight line through a cluster) judged by the whole-group sim, not
  the generic use families. Also the inventory slot: holding a Mega blocks every other powerup for the rest of the round.

## 3. Real-time continuation and benchmark startup

Status2026-10-20: core event-order, delayed inventory restoration, exact forecast resume and bounded worker ownership fixes retained. Optional checked continuation remains off. Quiet same-layout repeat scored149 versus159 mean points; fresh-layout screen scored153.667 versus149 but doubled accepted rescues2to4 and had a TreasureBox startup-state mismatch. Full dynamics suite240passed. See `docs/HANDOFF_REALTIME_PLANNER_2026-10-19.md` and its per-map evidence.

Before another promotion test, synchronize the benchmark's neutral/countdown startup and record/compare the native state at GO. Gem RNG equality alone does not guarantee the same physical state or pre-spin; asynchronous capture differed by one observation on TreasureBox. The benchmark now matches menu-play's1ms Python switch interval, requires native real-time pacing, archives source/binaries and checks independent rescue counts.

If revisiting checked continuation, first explain the TreasureBox points loss and extra Cragmire rescues. Keep it opt-in until a reproducible multi-map benefit survives confirmation. No training or repeated recipe search is running.

## 4. A2 fallback falls (2026-10-20 afternoon)

Status: mechanism found and a mitigation built behind `--pickup-stop` (default off). See the 14:20-17:35 log entries.
- Cause: fast approach to an edge gem -> every candidate collects it and runs over the edge within the 48-reply horizon
  -> all rejected -> unchecked actor drives 1-2 s -> fall. A full brake from the same state is safe but was never a
  candidate (BrakeContinuation existed only behind --brake-continuation 0).
- Flag: stop_now / stop_short last-resort candidates when nothing passes and the free roll leaves the floor (30 u ray,
  1.5 s); parked-streak cap (24 decisions at rest); held stop plans released when the edge test clears.
- Five ten-map screens: points a wash (+1.55 / +0.15 / -2.35 / +0.05 / +0.80 vs a stable 128 baseline), falls 7-10 ->
  3-5 except with a speed floor (7 -> 9). Cube and Duplex edge falls removed every time (+10..+41 a round); Cragmire
  loses 10-40 a round because the all-rejected state is common beside ledges the actor handles, and a stopped marble
  there gets stuck (8 s rescues).
- Next ideas: (a) when a stop plan is held and the next search still fails at rest, prefer a jump/route candidate over
  the actor (Cragmire's ledges are gaps); (b) a Cragmire-free assessment is not allowed (no map hacks), so the fix must
  be in why the planner rejects everything beside those ledges in the first place (likely the same 'collect then over
  the edge' pattern at low speed, where a slow approach plan should exist).

## 5. Edge hits (2026-10-20 evening to 2026-10-21 05:35): DONE, on by default

Status: on by default since 2026-10-21 05:30 (`--no-edge-hits` = control). Eight ten-map native pairs: v2 +1.10,
v4 +2.55, v5 +0.20, v6 +2.95, v7 +1.85, v7 fresh seeds +1.35, v8 +0.60 (reverted), v9 +2.15 equal-map points
(about +2, 1.6 percent). Blast use unchanged; falls vary without a trend.
- Physics: the engine bounce on a ledge lip (restitution along the tilted normal + spin-friction kick) sends the marble
  up while keeping most forward speed; the compiled simulator reproduced 114 of 118 natural edge hits exactly.
- Recipe: nav/marble_dynamics/edge_hits.py (real lips: wall-or-nothing other side AND a drop beyond; candidates 0.4-2.6 u
  above the marble toward a climb target; jump timing from distance/speed/rise, apex timing for wall hits) + search.py
  (lip-aimed plans with own route and through box, timed jumps on route/direct lines, landing extension, climb gate
  edge_climb, evaluated after the navigator, retry guard EDGE_RETRY_DECISIONS) + reliability_eval (edge_recent).
- Where it still loses: Concentric 7347019, Cragmire 12091843, Vortex 12091843 lose on most pairs (path divergence after
  an early edge run, plus some wasted climbs toward 4-5 u targets no edge hit can reach). Ideas not yet tried:
  (a) corner/drop-pop hits (fall onto the opposite lip to recover height during descents: the biggest natural kicks,
  20-33 u/s, were of this kind); (b) a round-level Blast valuation so 'edge hit now, Blast later' can win more often;
  (c) a cheaper lip candidate set (the climb gate rejects 60 percent of edge forecasts).

## Edge hits: push further (operator 2026-10-21 08:20: "they can be pushed much further")
- Edge hits are ON and worth ~+2 equal-map points; the operator knows they can do far more (longer climbs, corners of two ledges, chained hits).
- Ideas in item 5 above; wall rebounds (edge_hits.wall_candidates, 2026-10-21) are the first extension.
