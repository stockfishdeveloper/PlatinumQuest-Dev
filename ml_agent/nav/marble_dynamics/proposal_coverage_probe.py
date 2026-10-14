"""Measure bounded general proposal coverage with full stopping/landing checks.

More samples optimize ordinary controls, not model weights. Full static collision
geometry and the same independent physics/ambiguity gates decide each candidate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.collision.mesh import atomic_json
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.global_goals import observed_gems
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.landing import alternatives
from nav.marble_dynamics.phased_search import explore
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.search import useful_candidate
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/CubeIsle_Hunt/151113/physics_plan/decisions.json'
    initial=json.loads(source.read_text())[503];dynamics=initial['dynamics'];raw=np.asarray(initial['raw_observation'],np.float32)
    mesh=CachedCompiledSweepMap(ML/'collision_maps/CubeIsle_Hunt.npz',a.dll);engine=GeneralPhysics(mesh)
    engine.terminal_stop_replies=48
    terrain=TerrainGrid(TerrainMap.resolve('CubeIsle_Hunt'));geometric=walking_terrain(terrain,mesh)
    geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance);route=RoutePlanner(geometric)
    goals=observed_gems(dynamics,raw);checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth'
    checkpoint_hash=sha(checkpoint)
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/experimental_sweep.py',a.dll]
    atomic_json(a.output/'prepared.json',dict(map='CubeIsle_Hunt',decision=503,source=str(source),source_sha256=sha(source),
        source_hashes={str(p):sha(p) for p in sources},checkpoint_sha256=checkpoint_hash,
        budgets=[40,160,400],horizon=80,braking=48,landing_replies=80,landing_points=8,
        landing_prefix_limit=12,proposal_seed=997,goals=goals,training=False,
        limitation='Archived development state, action search only; not native success or policy training.'))
    results=[];started=time.perf_counter();selected=None
    for samples in [40,160,400]:
        for goal in goals:
            choice,pairs=explore(engine,route,dynamics,goal[:3],horizon=80,samples=samples,seed=997,brake_replies=48)
            landing=[]
            if choice is None:
                eligible=[pair for pair in pairs if pair[1] and pair[0].get('possible_pickup_interval') is not None
                    and not pair[1]['failure'] and not any(i<pair[0]['possible_pickup_interval'] for i in pair[1]['ambiguous_intervals'])]
                eligible.sort(key=lambda pair:(pair[0]['minimum_gem_distance_u'],pair[0]['possible_pickup_interval']))
                unique=[];seen=set()
                for pair in eligible:
                    key=tuple(tuple(a) for a in pair[1]['proposed'][:pair[0]['possible_pickup_interval']])
                    if key in seen:continue
                    seen.add(key);unique.append(pair)
                    if len(unique)>=12:break
                landing=alternatives(engine,route,dynamics,goal[:3],unique,80,8)
                route.navigate(dynamics['native']['state']['position'],goal[:3])
                choice=useful_candidate(landing,route,route.distance(dynamics['native']['state']['position']),True)
            entry=dict(samples=samples,goal=goal,selected=choice[0] if choice else None,
                sampled_candidates=len(pairs),landing_candidates=len(landing),
                rejections=dict(Counter(str(info['rejected']) for info,_ in pairs+landing)))
            results.append(entry);print(json.dumps(entry),flush=True)
            atomic_json(a.output/'progress.json',dict(attempts=results,training=False))
            if choice:
                selected=dict(samples=samples,goal=goal,info=choice[0])
                atomic_json(a.output/'selected_forecast.json',choice[1]);break
        if selected:break
    atomic_json(a.output/'report.json',dict(attempts=results,selected=selected,elapsed_s=time.perf_counter()-started,
        checkpoint_unchanged=sha(checkpoint)==checkpoint_hash,training=False))


if __name__=='__main__':main()
