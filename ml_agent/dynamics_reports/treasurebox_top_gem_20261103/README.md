# TreasureBox: the gem on the box top without Blast or powerups (2026-11-03)

Operator project. Situation: marble at rest at (86.87, 28.31, 4.19), the five-gem group of the operator's 14:03 round, target the
2-point gem at (89, 15, 8.5) on the raised pad of the box (box top 8.0, pad 8.5, floor 4.0). No Blast, no powerup
(`PQ_NO_BLAST=1 PQ_NO_POWERUPS=1`: nav/marble_dynamics/blast.py, nav/protocol.py). Everything measured on the production simulator
(tools/sim_bench, exact engine port) and, where stated, in the game at 1x.

## Part 1: the planner gets up there (wall kick off the deck, onto the step)

Baseline: the planner collected the four floor gems and then dithered below the box for the rest of the segment (0 of 1).

Mechanism found by probing the engine: a full-throttle run at a vertical face with a jump 0.6 to 1.3 u before contact adds the
spin-friction kick to the jump; the marble rises about 3.5 u and rebounds 2 to 2.5 u back from the face. Speed above 12 u/s kills
the kick. A straight run at the box's own face therefore never stands on it (apex 8.0, 2.5 u back). What works: the 5.6 u lane
in the slot west of the 7.5 u step, angled 10 deg at the deck's south face (y 22); the rebound lands on the step (3.5 u rise),
then a hop onto the box top and the pad.

Planner changes (nav/marble_dynamics/climbs.py, crossings.py, reliability_eval.py), all generic:
- wall-climb run lanes may be angled up to 20 deg off the face normal; the landing is sought along the mirrored line
  (`WALL_LANE_ANGLES`); the lane test uses the mesh (`top_at`) because the walk grid keeps a level under a solid block;
- `WALL_RUN_MIN_U` 6 -> 5; `WALL_LATERAL_U` 2.5 -> 1.2; `WALL_REACH_U` min 1.5 -> 2.0; landing cells inside the place
  (`WALL_LAND_EDGE_U`); `WALL_TRIG_U` gains 1.3; `wall_meter` float-safe (a 3.5 u rise needs no Blast);
- a climb's approach phase also searches the gem itself (`CLIMB_MIX_APPROACH`); PQ_NO_BLAST reports an empty meter to the staging.

Results: sim 5 of 5 nearby starts (actual spot, 6 u NW, channel, east, mid), the top gem 5 to 9 s after the last floor gem.
Game at 1x, run 1 (before `WALL_REACH_U` 2.0): 4 of 8 tries within 15 s; run 2: 9 of 9 tries, top gem at 8.1 to 13.1 s.
Viewer: `tb_top_gem_climb.html` (recorded game positions of run 2 over the mesh).

## Part 2: as fast as possible from the start (the lip pop)

The operator's edge hit: run south at full throttle along x 87.4 to 88.3 (just west of the box), jump early (about 0.6 to 0.8 s
after the start), so the marble COMES DOWN onto a low lip with its bottom at the lip's height while still carrying 12 to 15 u/s:
the lip of the 0.5 u raised platform at y 18 (z 4.5) or the 1 u step at y 16 (z 5.0). The lip's reflection of the forward-and-down
velocity plus the spin kick turn 14.5 u/s forward into 14 u/s up with 5 u/s kept forward (tick trace: velocity (1.1, -14.5, -3.8)
-> (0.7, -5.1, 14.2) at (87.77, 16.07, 5.15)); the marble peaks at 9.1 to 9.4 u over the pad and takes the gem. A rising clip of
the same lip (the existing edge-hit timing) only reaches 5.5 u.

Optimizer (tb_opt3s: programs of up to 3 phases, 230,000 evaluations incl. the game's one-reply lag):

| | best | typical |
|---|---:|---:|
| nominal start, no lag | 2.05 s | 2.24 s |
| with the game's reply lag | 2.05 s | 2.18 s |
| physical floor (13.4 u at full acceleration + the pop) | about 2.0 s | |

Open-loop programs are knife-edge: at most 6 of 30 perturbed starts (+-0.3 u, 0 to 0.9 u/s initial speed, lag). The game
confirmed it: three scripted replays matched the simulator to the millimetre until the lip and missed it by 0.2 u.
Closed loop (tb_mpc: every decision forecast 5 lanes x 5 speed caps x 6 jump delays with the exact simulator, commit the
earliest plan that collects): 28 of 28 unperturbed, 23 of 28 with the lag, 2.24 to 2.75 s (mean 2.42). Speed modulation is what
makes it robust: the pop needs the landing within about 0.1 u of the lip height.

Planner changes so far (nav/marble_dynamics/edge_hits.py, search.py): descending-timing lip plans (`descent_jump_replies`,
`FastLipProposal` at `LIP_FAST_U_S` 14, first in the family), arrival from the acceleration profile (`arrival_time`), lips from
0.25 u (`ABOVE_MIN_U`), one candidate per ledge level (`LIP_SAME_LEVEL_U`), lip points off segment ends, `MAX_LIPS` 3. At the first
decision from the start the planner now forecasts the fast lip plans (one peaks at 8.45 u, 0.25 u short of the pad) but still
chooses another plan; the speed-cap and jump-delay spread of the closed-loop controller is not in the family yet. To be
integrated together with the operator.

Viewer: `tb_lip_pop_routes.html` (simulator trajectories of the five fastest routes).

Scratch tools (tb_bed.py sim test bed from a situation, wall_probe.py, tb_search3s.py, tb_opt3s.py, tb_closed.py, tb_mpc.py,
tb_viewer.py, tb_live*.py, the situation json): archived in C:/Users/doug/pq_archive/treasurebox_top_gem_tools_2026-11-03.zip.

## Part 3: the lip pop in the game at 1x (17:00 to 17:07, operator accepted: "that's the correct answer")

tb_live.py (archive zip): the closed-loop controller driven from the game's own dynamics message every decision (candidate sweep
5 lanes x 5 speed caps x 6 jump delays on 16 exact-simulator workers, ~300 ms a sweep under the planner's load; the first
collecting plan is committed and swapped for any faster one found before the jump). It was hooked into realtime.py through a PQ_SCRIPT={"live": ...} env switch for the
run; that experiment hook was removed from realtime.py on 11-03 evening (the planner integration replaces it). Run 2 (live_lip_pop_run2_17h03.txt): 10 of 11 tries took the top gem first,
2.18 to 2.43 s after the start (2.18, 2.30 x4, 2.37 x3, 2.43 x2); try 1 missed (closest 1.25 u). The misses are start
states where the 1 u/s speed-cap grid straddles the landing height: on the try 11 state a 13.5 u/s cap collects (probe,
tb_live_probe.py in the zip), so the cap grid needs 0.5 u/s steps. The sweep is deterministic (pool and serial agree).
