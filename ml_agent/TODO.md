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

## 3. Real-time continuation and benchmark startup

Status2026-10-20: core event-order, delayed inventory restoration, exact forecast resume and bounded worker ownership fixes retained. Optional checked continuation remains off. Quiet same-layout repeat scored149 versus159 mean points; fresh-layout screen scored153.667 versus149 but doubled accepted rescues2to4 and had a TreasureBox startup-state mismatch. Full dynamics suite240passed. See `docs/HANDOFF_REALTIME_PLANNER_2026-10-19.md` and its per-map evidence.

Before another promotion test, synchronize the benchmark's neutral/countdown startup and record/compare the native state at GO. Gem RNG equality alone does not guarantee the same physical state or pre-spin; asynchronous capture differed by one observation on TreasureBox. The benchmark now matches menu-play's1ms Python switch interval, requires native real-time pacing, archives source/binaries and checks independent rescue counts.

If revisiting checked continuation, first explain the TreasureBox points loss and extra Cragmire rescues. Keep it opt-in until a reproducible multi-map benefit survives confirmation. No training or repeated recipe search is running.
