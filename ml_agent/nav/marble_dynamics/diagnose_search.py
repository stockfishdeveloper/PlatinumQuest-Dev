"""Reconstruct past policy state and diagnose a saved failed search offline.

Only past actual observations restore the recurrence. Candidate futures use
the independent predictor from the terminal observation, never future game data.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json
from nav.gems import visible_gems
from nav.routing import RoutePlanner, snap_to_walkable
from nav.terrain import TerrainGrid, TerrainMap
from nav.marble_dynamics.record import ML, sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.search import search, assess, BrakeContinuation
from nav.marble_dynamics.bridge import predict_replies


def reconstruct_navigator(terrain, checkpoint, logs, terminal_raw):
    nav = FrozenNavigator(terrain, checkpoint)
    for index, log in enumerate(logs):
        raw = np.asarray(log['raw_observation'], dtype=np.float32)
        gem = np.asarray(log['gem'][:3])
        others = [g for g in visible_gems(raw) if np.linalg.norm(np.asarray(g[:3])-gem) > .1]
        near = min(others, key=lambda g: np.linalg.norm(np.asarray(g[:3])-gem)) if others else None
        next_goal = snap_to_walkable(terrain, *near[:3])[0] if near else None
        nav.act(raw, log['waypoint'], gem, next_goal)
        nav.sent_action(log['action'], raw)
        after = np.asarray(logs[index+1]['raw_observation'], dtype=np.float32) if index+1 < len(logs) else terminal_raw
        nav.outcome(raw, after)
    return nav


def tail_geometry(mesh, rollout):
    result = []
    for state in rollout['states'][-4:]:
        p = np.asarray(state['position']); radius = state['radius']
        hit = mesh.raycast(p+[0,0,.05], p-[0,0,radius+.10], cull_backfaces=True)
        result.append(dict(position=state['position'], velocity=state['velocity'],
            support_normal=list(hit['normal']) if hit else None,
            support_normal_velocity=float(np.dot(state['velocity'], hit['normal'])) if hit else None))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('terminal', type=Path)
    p.add_argument('--horizons', type=int, nargs='+', default=[40,80,120])
    p.add_argument('--brake-tail',type=int,default=0,help='Try this many braking replies after otherwise unambiguous unsupported candidates')
    p.add_argument('--checkpoint', type=Path, default=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    case = json.loads(a.terminal.read_text()); name = a.terminal.parents[2].name
    logs = json.loads(a.terminal.with_name('decisions.json').read_text())
    terrain = TerrainGrid(TerrainMap.resolve(name)); route = RoutePlanner(terrain)
    engine = GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    raw = np.asarray(case['raw_observation'], dtype=np.float32)
    nav = reconstruct_navigator(terrain, a.checkpoint, logs, raw)
    report = dict(map=name, terminal=str(a.terminal), terminal_sha256=sha(a.terminal),
        checkpoint_sha256=sha(a.checkpoint), replayed_past_decisions=nav.decision,
        source_hashes={str(f):sha(f) for f in Path(__file__).parent.glob('*.py')},
        control_basis='Past observations restore recurrence; all future states predicted from terminal snapshot.',
        policy_training_started=False, horizons=[])
    for horizon in a.horizons:
        started = time.perf_counter()
        chosen, pairs = search(engine, route, case['dynamics'], raw, case['target'][:3], nav, horizon)
        entries = []
        for info, rollout in pairs:
            entry = dict(info)
            if rollout:
                entry['tail'] = tail_geometry(engine.mesh, rollout)
                entry['forecast_s'] = (len(rollout['states'])-1)*.064
                ambiguities=[event for events in rollout['events'] for event in events if event['kind']=='ambiguous_contact']
                reasons={event['reason'] for event in ambiguities}
                entry['ambiguity_reason_counts']={reason:sum(event['reason']==reason for event in ambiguities) for reason in reasons}
                entry['first_ambiguity_by_reason']=[next(event for event in ambiguities if event['reason']==reason) for reason in sorted(reasons)]
                if info['rejected'] is None:
                    atomic_json(a.output/(str(horizon)+'_'+info['label']+'.rollout.json'), rollout)
            entries.append(entry)
            if a.brake_tail and rollout and info['rejected']=='no_supported_continuation':
                extended=predict_replies(engine,case['dynamics'],BrakeContinuation(rollout['proposed'],case['target'][:3],a.brake_tail))
                extra=assess(engine,route,case['dynamics'],np.asarray(case['target'][:3]),extended,info['label']+'_brake_tail')
                extra['tail']=tail_geometry(engine.mesh,extended);extra['forecast_s']=(len(extended['states'])-1)*.064
                entries.append(extra)
                atomic_json(a.output/(str(horizon)+'_'+extra['label']+'.rollout.json'),extended)
        result = dict(horizon=horizon, elapsed_s=time.perf_counter()-started,
            chosen=chosen[0] if chosen else None, candidates=entries)
        report['horizons'].append(result); atomic_json(a.output/'report.json', report)
        print(json.dumps(dict(horizon=horizon, elapsed_s=result['elapsed_s'], chosen=result['chosen'],
            rejection_counts={str(k):sum(c['rejected']==k for c in entries) for k in {c['rejected'] for c in entries}})), flush=True)


if __name__ == '__main__':main()
