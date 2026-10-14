"""Check all currently observed goals at archived planner failure states.

No native execution, hidden mission items, training or invented inventory. These
are development diagnostics: no outcome-based result is called fresh validation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import atomic_json
from nav.routing import RoutePlanner
from nav.terrain import TerrainMap,TerrainGrid
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.global_goals import observed_gems
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.query_cache import CachedCollisionMap
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.reliability_eval import pick
from nav.marble_dynamics.search import PlanMemory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[('reliability_landing_validation_20261012','CubeIsle_Hunt',151113),
           ('reliability_geometry_pool_20261012','FracturedIslands_Hunt',111108),
           ('reliability_geometry_pool_20261012','TreasureBox_Hunt',111108)]
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth'
    fingerprint=sha(checkpoint);sources={str(path):sha(path) for path in Path(__file__).parent.glob('*.py')}
    atomic_json(a.output/'prepared.json',dict(cases=cases,horizon=40,recovery_horizon=80,
        braking_replies=48,preserve_resources=True,source_hashes=sources,checkpoint_sha256=fingerprint,
        training=False,selection='Archived terminal states; development diagnostics only'))
    results=[]
    for directory,name,seed in cases:
        source=ML/'dynamics_reports'/directory/name/str(seed)/'physics_plan/terminal_search.json'
        terminal=json.loads(source.read_text());dynamics=terminal['dynamics'];raw=np.asarray(terminal['raw_observation'],dtype=np.float32)
        all_goals=observed_gems(dynamics,raw)
        # Current nearest-five information is already archived in the raw block.
        from nav.gems import visible_gems
        nearby=visible_gems(raw)
        extra=[g for g in all_goals if not any(np.linalg.norm(np.asarray(g[:3])-np.asarray(n[:3]))<.1 for n in nearby)]
        terrain=TerrainGrid(TerrainMap.resolve(name));engine=GeneralPhysics(CachedCollisionMap(ML/'collision_maps'/(name+'.npz')))
        geometric=walking_terrain(terrain,engine.mesh)
        geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance)
        nav=FrozenNavigator(terrain,checkpoint);route=RoutePlanner(geometric)
        started=time.perf_counter()
        if extra:
            chosen,pairs,current=pick(engine,route,dynamics,raw,extra,extra[0],nav,40,PlanMemory(),48,80,preserve_resources=True)
        else:chosen=None;pairs=[];current=None
        result=dict(map=name,seed=seed,source=str(source),source_sha256=sha(source),
            geometry_sha256=engine.mesh.info['mesh_sha256'],observed_goals=len(all_goals),nearby_goals=len(nearby),
            extra_goals=[list(g) for g in extra],selected=chosen[0] if chosen else None,
            selected_goal=list(current) if chosen else None,candidates=pairs,elapsed_s=time.perf_counter()-started)
        results.append(result);atomic_json(a.output/(name+'.json'),result)
        atomic_json(a.output/'summary.json',dict(cases=results,checkpoint_unchanged=sha(checkpoint)==fingerprint,training=False))
        print(json.dumps({k:result[k] for k in ('map','observed_goals','nearby_goals','selected','selected_goal','elapsed_s')}),flush=True)
    assert sha(checkpoint)==fingerprint


if __name__=='__main__':main()
