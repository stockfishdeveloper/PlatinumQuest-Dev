# PlatinumQuest RL Training

Reinforcement learning system for training an AI agent to play Hunt mode in PlatinumQuest.

> **Where the live work is.** The current agent is the **navigator** in `nav/`, a recurrent PPO
> policy. Its status document is **`HANDOFF_NAV_TRAINING.md`**, which overrides this README. The older
> per-map MLP (`train_ppo.py`, `dashboard.py`, `analyze_log.py`) is the old King-of-the-Marble baseline
> and is not being developed.
>
> **As of 2026-09-26:** about 153 points on KingOfTheMarble (3x, 8 deterministic rounds), against human
> rounds of 143-177. The next piece of work is jump physics: start at `KOTMJUMP_START_HERE.md`.

## Quick start (navigator)

```powershell
cd ml_agent
pip install -r requirements.txt

Start-Process powershell -ArgumentList '-File','start_training.ps1'   # trainer first, waits for ports, then 8 games
.\eval_cycle.ps1 -Tag mytest -Rounds 8     # stop training, play 8 real rounds, restart training
.\real_run.ps1 -Rounds 1                   # one real KOTM round with nav_latest (training must be stopped)
```

Watch a round in real time at 1x:

```powershell
$env:NAV_WATCH="1"; $env:NAV_SPEED="1"; $env:NAV_VIEW_SUBSTEPS="4"
.\real_run.ps1 -Rounds 1
```

Two rules that are easy to get wrong:

* **Python is the SERVER.** It binds and listens; the game dials it. Every launcher therefore starts
  Python and waits for the port before launching the game. Never launch a game first.
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
(`nav/train_nav.py` plus one torch-free `nav/vec_worker.py` per instance), on the game's own gems.

Observation (NAV_OBS_V6): two local terrain height crops plus a 61-number vector (goal bearing and
distance, velocity, edge rays, the next gem, the gap ahead with landing-predictor verdicts, and the
marble's spin). Action: a 2D direction, a throttle, a jump and a brake. The game-side observer is
`../Marble Blast Platinum/platinum/client/scripts/ai/observer.cs`. Details: `HANDOFF_NAV_TRAINING.md`
section 5.

## Console commands

- `MLAgent::start()` / `MLAgent::stop()` - connect to Python and start/stop AI control
- `$MLAgent::AutoStart = true` - auto-start when entering Hunt mode
- `AIObserver::collectState()` - test observation collection
- `$AIObserver::GemSource` - `"server"` (correct) or `"itemarray"` (reproduces a fixed bug that
  made the agent blind for up to 1.9 s after every pickup; log section 22)
- `$AIObserver::GemDiag` - remaining budget of gem-source disagreement log lines

Controls the Python side sends over the socket include `OOBCLICK`, the legal quick respawn a human
gets by clicking the left mouse while out of bounds. It fires only while the game's own
`%client.isOOB` is true, which is the same gate a human click passes, so it can never respawn
earlier than a real competitive Hunt round allows. Disable with `NAV_OOB_CLICK=0`.

The game is normally launched headlessly for training instead:
`marbleblast_mbx.exe -autotrain <MissionFileBaseName> -aiport <port>`.

## Hard-won lessons (do not re-discover these)

- **Judge only by 8-round `nav/real_run.py` evaluations.** Training curves have misled repeatedly,
  most recently when a 15 % training-speed gain converted to 5 % of real score.
- **Synthetic training cannot see bugs in the gem pipeline.** Until 2026-09-23 training used
  synthetic teleport-to-waypoint segments, and a defect in gem reporting was invisible to every
  training metric. Training now uses the game's own gems for this reason.
- **Verify both ends of any data contract.** Read the game file and the Python file; never assume.
- **The walk grid can have phantom holes** (solid ground marked non-walkable). Verify any new map
  by teleporting the marble into the "holes" before training on it.
- **Waypoints may only ever appear inside a real gem.** No routing corner points, no marker on
  empty ground.
- **No Unicode in Python log output.** Windows cp1252 crashes on characters like arrows. Use ASCII.
- **Delete the compiled `.dso`** after editing any `.cs` or `.mcs`, or the engine keeps running the
  old version.
- **PowerShell `$env:X = ""` deletes X**, so an "off" setting silently becomes the code default.
- **Watch a real round at 1x when stuck.** Several of the biggest gains started with a human
  watching play and describing something that looked wrong, invisible in every tracked metric.

## Documents

- `HANDOFF_NAV_TRAINING.md` - the current state: scores, the system, how to run it, traps, rules. Start here.
- `docs/HANDOFF_NAV_TRAINING_LOG.md` - the dated history of every experiment (sections 1-28.x).
- `BIG_QUESTIONS.md` - the big open questions: learning the physics, representing terrain, finding routes, handing plans to the policy.
- `KOTMJUMP_NAVIGATION_DESIGN.md` - the 2026-09-27 design and implementation sequence for learned physics, reliable floating-gem jumps, and useful route discovery on kotmjump; not implemented yet.
- `JUMP_PLAN_OVERVIEW.md` - the jump physics plan in plain words, with the roadmap.
- `KOTMJUMP_START_HERE.md` - the first experiment of the jump physics plan (P0). Start here for that work.
- `PHYSICS_PLANNER_PLAN.md` - the first jump-predictor draft (2026-09-26), superseded by the design; history.
- `PHYSICS_SKILLS_DESIGN.md` - background on the engine's physics (`marble.cc`).
- `docs/archive/` - superseded documents (the 2026-09-16 roadmap and implementation plan, the old
  dashboard's chat context, earlier designs).

## Troubleshooting

**The game connects but the marble does not move.** Python was not listening when the round
reached GO. Start Python first and wait for the port (`nav_ready.ps1` does this).

**Python stops with "the game sends 35 observation numbers but NAV_OBS_V6 needs 38".** The game is
running an old compiled observer. Delete `client/scripts/ai/observer.cs.dso`.

**`real_run.ps1` refuses to start.** The trainer is running. Stop it, or pass `-Force` and risk a
GPU out-of-memory.

**A script edit had no effect.** Delete the matching `.dso`.

**The marble looks slow at 1x.** Use `NAV_VIEW_SUBSTEPS=4`. Without it the sim advances in one
64 ms jump per decision and the render just redraws a frozen state. Do not measure with substeps
on; the integration step differs from training.
