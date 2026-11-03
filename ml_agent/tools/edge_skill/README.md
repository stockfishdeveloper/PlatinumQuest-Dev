# Standalone edge-hit skill

The campaign is complete: see [REPORT.md](../../dynamics_reports/edge_skill_20261103/REPORT.md) (416/416 accepted native successes on 1,044 sealed starts, 11-03). Production planner integration is deliberately absent and awaits review.

Current evidence and protocol: [PROTOCOL.md](../../dynamics_reports/edge_skill_20261103/PROTOCOL.md). The previous experiment and its videos remain in `dynamics_reports/edge_skill_20261102`. Its implementation is archived in `edge_skill_20261103/previous_implementation.zip`; obsolete entry points were removed from this directory.

## Components

- `maps`, `native_lab`: procedural native geometry, real-map fixture selection, hidden game collection.
- `execution`, `audit`: contact-conditioned feedback, targeted search, terminal contract and independent contact verification.
- `compiled`, `edge_execution.cpp`: a private C++ controller DLL using the unchanged existing physics and input bridge.
- `learn`: geometry-local multi-proposal learning and a separately measured retrieval baseline.
- `experiment`: one CLI for collection, anchors, perturbations, evaluation and native confirmation.
- `review`: self-contained playback of recorded native trajectories at 1x.
- `test_skill`: mechanism, terminal-state, control encoding, queue continuation and C++/Python conformance.

## Environment

Run from the repository root:

```powershell
$env:PYTHONPATH = 'ml_agent'
$edgePython = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
& $edgePython -m tools.edge_skill.compiled
& $edgePython -m unittest tools.edge_skill.test_skill -v
& $edgePython -m tools.edge_skill.experiment --help
```

The compiler uses the installed Visual Studio Build Tools. Binaries are hashed and cached under TEMP, not written over the production simulator DLL. No production planner or game script imports this skill.

The current campaign directory is `ml_agent/dynamics_reports/edge_skill_20261103`. Output names are resumable: use a new name when changing the controller, budget, checkpoint or experimental settings. Frozen suites must not be silently replaced.

## Contract

Given the full native/control state, an exact lip segment, a landing surface and a desired exit velocity, the solver returns a checked ordinary-movement plan or a decline. Blast and power activation are excluded. The maintained feedback representation supports horizontal and tilted lips, independently rotated/sloped landing platforms, and a flat approach floor.

A success requires an independently audited lower-half lip impulse, no wall assist, supported arrival near the requested goal, approximately the requested planar exit velocity for 0.512 seconds, and a supported braking continuation for 0.384 seconds. Passing the nominated lip without contact ends that attempt. Low rolling lips are diagnostic, separate from the jump-based headline results.

`stable_inputs=True` encodes the existing 1/16 native input levels away from rounding boundaries. Actual native axes remain clamped to their ordinary range. Training labels from this encoding must not be mixed with labels from the legacy encoding.

The queued-state guidance family forecasts the already committed input interval before choosing its next correction. Forecasts do not substitute for future native observations. Native runs record actual per-substep contact evidence and compare exact-action replay against the simulator.

## Standalone API

`learn.Proposer.propose(engine, opportunity, dynamics, budget=48)` returns parameters, the forecast, the bound opportunity, the exact planning clock and a checked result. The checkpoint supplies its input-encoding contract when the opportunity does not specify one.

For an accepted result, `execution.make_executor(candidate, engine, dynamics)` creates a fresh feedback executor using the correct bound contract. It rejects a stale planning clock. Call it once per actual 64 ms observation; the laboratory's native runner demonstrates the observation hook and outcome audit. The simulator's already-advanced `_forecast_policy` is internal and is not an executor.

The optional native `--execution retained` mode is an experimental ablation, not the default: it retains checked commands across tiny feedback discrepancies and can refresh before contact. The default is `policy`.

## Evaluation boundaries

Generated geometry is split before optimization. Final synthetic and real-map test locations remain sealed until the controller and selection rules are frozen. Report candidate coverage separately from execution reliability; arbitrary random airborne states are not guaranteed reachable. Known successful continuations and state perturbations measure different things.

Real-map fixtures are synthetic resets on actual collision geometry, not collected live route opportunities. A drill pass does not establish points improvement. Native lockstep results do not establish real-time deadline compliance; proposal and execution latency must be measured separately.

All owned validation games launch hidden and clean up on exit. The trace viewer is recorded native motion, clearly distinguished from game footage. Planner integration and full-round scoring remain the next reviewed stage.
