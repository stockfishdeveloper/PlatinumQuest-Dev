"""Check whether terminal braking viability redirects an archived gem choice."""
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
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.search import search,useful_candidate
from nav.marble_dynamics.viability import braking_viability
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/CubeIsle_Hunt/151113/physics_plan/decisions.json'
    initial=json.loads(source.read_text())[503];dynamics=initial['dynamics'];raw=np.asarray(initial['raw_observation'],np.float32)
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth';fingerprint=sha(checkpoint)
    mesh=CachedCompiledSweepMap(ML/'collision_maps/CubeIsle_Hunt.npz',a.dll);engine=GeneralPhysics(mesh)
    terrain=TerrainGrid(TerrainMap.resolve('CubeIsle_Hunt'));geometric=walking_terrain(terrain,mesh)
    geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance)
    route=RoutePlanner(geometric);nav=FrozenNavigator(terrain,checkpoint)
    goals=observed_gems(dynamics,raw)
    atomic_json(a.output/'prepared.json',dict(map='CubeIsle_Hunt',decision=503,source=str(source),
        source_sha256=sha(source),checkpoint_sha256=fingerprint,dll_sha256=sha(a.dll),
        diagnostic_source_sha256=sha(__file__),horizons=[40,80],braking=48,goals=goals,
        training=False,limitation='Development state; frozen neural proposal history reset; no execution.'))
    results=[];selected=None;started=time.perf_counter()
    for horizon in [40,80]:
        for number,gem in enumerate(goals):
            _,pairs=search(engine,route,dynamics,raw,gem[:3],nav,horizon,brake_replies=48)
            checked=[]
            for info,rollout in pairs:
                if info['rejected'] or rollout is None:continue
                check,joined=braking_viability(engine,dynamics,rollout,gem[:3],48)
                clone=dict(info,stopping_check=check)
                if not check['valid']:clone['rejected']='no_stopping_continuation:'+check['rejected']
                checked.append((clone,joined))
            route.navigate(dynamics['native']['state']['position'],gem[:3])
            choice=useful_candidate(checked,route,route.distance(dynamics['native']['state']['position']),True)
            entry=dict(horizon=horizon,goal=gem,selected=choice[0] if choice else None,
                rejections=dict(Counter(str(i['rejected']) for i,_ in checked)),candidates=[i for i,_ in checked])
            results.append(entry)
            print(json.dumps({k:entry[k] for k in ['horizon','goal','selected','rejections']}),flush=True)
            if choice:
                selected=dict(goal=gem,horizon=horizon,info=choice[0])
                atomic_json(a.output/'selected_forecast.json',choice[1]);break
        if selected:break
    atomic_json(a.output/'report.json',dict(attempts=results,selected=selected,elapsed_s=time.perf_counter()-started,
        checkpoint_unchanged=sha(checkpoint)==fingerprint,training=False))


if __name__=='__main__':main()
