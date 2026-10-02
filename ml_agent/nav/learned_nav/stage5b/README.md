# Stage 5b check scripts (2026-09-29)

One-off analysis and in-game test scripts behind log section 33 (docs/HANDOFF_NAV_TRAINING_LOG.md), kept as they
were run. They use absolute paths to this checkout; run them from `ml_agent/`.

In-game tests (start the script first, then the game: `marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport <port>`;
kotmjump_p0 has KOTM's exact geometry):

| script | what it checks | output |
|---|---|---|
| diagjump3.py | open-loop centre-gem -> corner jumps at 9-11 u/s with four air inputs; records every state and reply | logs/learned_nav/diagjump3.jsonl |
| cornerdrill.py | the planner (shortcut settings) closed loop on the same crossing from pickup-like states (22/24) | logs/learned_nav/cornerdrill.jsonl |

Offline (no game):

| script | what it shows |
|---|---|
| replay_prior.py <jsonl> | the step model replayed on recorded game jumps, with and without planner.JUMP_PRIOR |
| takeoff.py <map_phys> | the model's jump apex against the stage 3 recordings, by speed and edge distance |
| afterdbg2.py | after-landing inputs on programs that land through the corner gem |
| jumpleg.py | the planner's first plan for the crossing from a centre-gem state |
| human_legs.py, voidentry.py, order.py | human (demos/) vs navigator gem-to-gem legs, hole crossings, gem order |
| walkfit.py, turnfit.py | the navigator's leg-time fit (route.py constants) |
| route_check.py | route.py's overrides on the human's recorded states |
| dry.py <tag> | consult-only shortcut probes against the navigator's actual time |

run_maps.ps1: the launcher used for all runs (a learned_nav module per map, N games at a time), e.g.
`powershell -File nav\learned_nav\stage5b\run_maps.ps1 -Module nav.learned_nav.hybrid -Maps KingOfTheMarble_Hunt
-Batch 1 -Port0 9511 -Tag s6_nav -ArgStr "--rounds 8 --memory current --shortcuts 0 --rescue 0 --tag s6_nav"`.
