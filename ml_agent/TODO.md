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

## 2. Mega Marble usage

Status: not started. The planner never uses Mega Marble today.

Why: the planner only considers using powerups 1-5 (Mega is type 6), and its physics forecast refuses Mega because the
game grows the marble through a server script (about 3.5x the radius, after a short delay, for 10 s; 5 s in competitive
Hunt) and that change is not modelled.

Plan (about 2 hours):
- Measure Mega in the real game: delay from use to growth, the new radius, any position or speed jump, the duration.
- Model it in the compiled forecast and check it against recorded Mega uses.
- Add "use Mega now / a bit later" moves; offline batch, then live rounds.

Possible gain: a bigger marble collects gems from ~0.5 u further away and rolls over small gaps. Risks: narrow ledges.
