# PlatinumQuest RL Training

Reinforcement learning system for training an AI agent to play Hunt mode in PlatinumQuest.

> **Where the live work is.** There are two agents in this repo. The current one is the
> **navigator** in `nav/`, a recurrent PPO policy trained across maps. Its authoritative status
> document is **`HANDOFF_NAV_TRAINING.md`**, and the CURRENT STATE block at the top of that file
> overrides everything else including this README. The older per-map MLP (`train_ppo.py`) is kept
> as the King-of-the-Marble baseline and is not being developed.
>
> **Verified as of 2026-09-21:** 98.4 points on KingOfTheMarble (human 143.5) and 96.8 gems on
> FlatGemTraining (human 104), 8 real rounds each.

## Quick start (navigator)

```powershell
cd ml_agent
pip install -r requirements.txt

.\start_training.ps1                      # trainer first, waits for ports, then 8 game instances
.\eval_both.ps1 -Tag mytest -Rounds 8      # stop training, play 8 real rounds on 2 maps, leave down
.\real_run.ps1 -Map FlatGemTraining_Hunt -Rounds 1    # one real round
```

Watch a round in real time at 1x:

```powershell
$env:NAV_WATCH="1"; $env:NAV_SPEED="1"; $env:NAV_VIEW_SUBSTEPS="4"; $env:NAV_MARK="1"
.\real_run.ps1 -Map FlatGemTraining_Hunt -Rounds 1
```

Two rules that are easy to get wrong:

* **Python is the SERVER.** It binds and listens; the game dials it 100 ms after "GO!". Every
  launcher therefore starts Python and waits for the port before launching the game. Never launch
  a game first.
* **A real-round evaluation requires stopping training.** The 8 GB GPU has no room for a 9th game
  instance beside a PPO update. `real_run.ps1` refuses to start while the trainer is up.

## Architecture

```
+-----------------+         +------------------+
|   Game (C++)    |  TCP    |  Python          |
|                 | <-----> |  (the SERVER)    |
| - Observations  |         | - Recurrent PPO  |
| - Actions       |         | - Reward + goals |
+-----------------+         +------------------+
```

One decision every 64 ms, delivered under engine fixed-step + lockstep so the sim advances only
when Python replies. Training runs 8 game instances against one trainer process
(`nav/train_nav.py` plus one torch-free `nav/vec_worker.py` per instance).

Observation: a local terrain height crop plus a 53-dim feature vector (goal bearing and distance,
next goal, velocity, edge ray-marches, frame history). Action: a 2D direction, a throttle, a jump
and a brake. The game-side observer is
`../Marble Blast Platinum/platinum/client/scripts/ai/observer.cs`.

## Console commands

- `MLAgent::start()` / `MLAgent::stop()` - connect to Python and start/stop AI control
- `$MLAgent::AutoStart = true` - auto-start when entering Hunt mode
- `AIObserver::collectState()` - test observation collection
- `$AIObserver::GemSource` - `"server"` (correct) or `"itemarray"` (reproduces a fixed bug that
  made the agent blind for up to 1.9 s after every pickup; see HANDOFF section 22)
- `$AIObserver::GemDiag` - remaining budget of gem-source disagreement log lines

The game is normally launched headlessly for training instead:
`marbleblast_mbx.exe -autotrain <MissionFileBaseName> -aiport <port>`.

## Current status

**Working and verified**
- Socket bridge, 8-instance multi-process training, engine fixed-step + lockstep, render-only
  view yaw, headless `-autotrain` launch
- Navigator PPO trained across FlatGemTraining / KingOfTheMarble / FlatIslands (4/3/1 split)
- Real-round transfer test on the game's own gems (`nav/real_run.py`), which is the ONLY
  trustworthy metric
- Human demo recording and comparison tooling (`plot_approach_profile.py`, demo `.npz` files)

**Known open items**
- Route length: 30.4 u per gem against the human's 28.3 (the larger remaining gap)
- Mid-leg peak speed: 12.8 u/s against the human's 14.94, mechanism measured in HANDOFF
  section 23
- FlatIslands has never had a real-round evaluation
- `Sprawl` is generated but not in the rotation, and its walk grid is unverified

## Hard-won lessons (do not re-discover these)

- **Judge only by 8-round `nav/real_run.py` evaluations.** Training curves have misled repeatedly,
  most recently when a 15 % training-speed gain converted to 5 % of real score.
- **Training cannot see bugs in the gem pipeline.** Training uses synthetic teleport-to-waypoint
  segments; real rounds use the game's own gems. A defect in gem reporting is invisible to every
  training metric by construction.
- **Verify both ends of any data contract.** Read the game file and the Python file; never assume.
- **The walk grid can have phantom holes** (solid ground marked non-walkable). Verify any new map
  by teleporting the marble into the "holes" before training on it.
- **Waypoints may only ever appear inside a real gem.** No routing corner points, no marker on
  empty ground.
- **No Unicode in Python log output.** Windows cp1252 crashes on characters like arrows. Use ASCII.
- **Delete the compiled `.dso`** after editing any `.cs` or `.mcs`, or the engine keeps running the
  old version.

## Documents

- `HANDOFF_NAV_TRAINING.md` - authoritative status, every experiment and its result. Start here.
- `ROADMAP_NAVIGATOR_PLANNER.md` - why a navigator plus a planner
- `IMPLEMENTATION_PLAN_NAVIGATOR.md` - work packages WP1-WP10
- `CHAT_CONTEXT.md` - system prompt for the dashboard chat assistant
- `logs/nav/overnight_notes.txt` - dated running log of runs and evaluations

## Troubleshooting

**The game connects but the marble does not move.** Python was not listening when the round
reached GO. Start Python first and wait for the port (`nav_ready.ps1` does this).

**`real_run.ps1` refuses to start.** The trainer is running. Stop it, or pass `-Force` and risk a
GPU out-of-memory.

**A script edit had no effect.** Delete the matching `.dso`.

**The marble looks slow at 1x.** Use `NAV_VIEW_SUBSTEPS=4`. Without it the sim advances in one
64 ms jump per decision and the render just redraws a frozen state. Do not measure with substeps
on; the integration step differs from training.
