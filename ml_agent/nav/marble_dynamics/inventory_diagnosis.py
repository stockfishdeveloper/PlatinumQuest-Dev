"""Explicit conditional-inventory hypotheses for a saved failed gem approach.

These are NOT native restorations or claimed attainable actions: ask what each
held item would enable at the recorded pose. Actual resource acquisition and
native verification remain separate. No game state or checkpoint is modified.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.terrain import TerrainGrid,TerrainMap
from nav.routing import RoutePlanner
from nav.protocol import RAW_POW_HELD
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.diagnose_search import reconstruct_navigator
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.search import search


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('terminal',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--horizon',type=int,default=80)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    case=json.loads(a.terminal.read_text());name=a.terminal.parents[2].name
    original=TerrainGrid(TerrainMap.resolve(name));engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
    terrain=walking_terrain(original,engine.mesh)
    raw=np.asarray(case['raw_observation'],dtype=np.float32)
    checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth'
    nav=reconstruct_navigator(original,checkpoint,json.loads(a.terminal.with_name('decisions.json').read_text()),raw)
    report=dict(map=name,terminal_sha256=sha(a.terminal),checkpoint_sha256=sha(checkpoint),
        actual_held=case['dynamics']['native']['state']['held'],cases=[],policy_training_started=False,
        source_hashes={str(path):sha(path) for path in Path(__file__).parent.glob('*.py')},
        contract='Conditional hypotheses: same recorded pose/pending controls, inventory deliberately replaced. No actual pickup, teleport, powerup grant or attainable multi-stage plan claimed.')
    for powerup in range(6):
        hypothetical=copy.deepcopy(case['dynamics']);hypothetical['native']['state']['held']=powerup
        hypothesis_raw=raw.copy();hypothesis_raw[RAW_POW_HELD]=powerup
        route=RoutePlanner(terrain);started=time.perf_counter()
        chosen,pairs=search(engine,route,hypothetical,hypothesis_raw,case['target'][:3],nav,a.horizon,brake_replies=48)
        accepted=[info for info,rollout in pairs if info['rejected'] is None]
        pickups=[info for info in accepted if info['possible_pickup_interval'] is not None]
        record=dict(hypothetical_held=powerup,elapsed_s=time.perf_counter()-started,
            accepted=len(accepted),pickup_candidates=pickups,chosen=chosen[0] if chosen else None,
            candidates=[info for info,rollout in pairs])
        report['cases'].append(record)
        if chosen:atomic_json(a.output/(str(powerup)+'.selected_rollout.json'),chosen[1])
        atomic_json(a.output/'report.json',report)
        print(json.dumps({key:record[key] for key in ('hypothetical_held','elapsed_s','accepted','pickup_candidates','chosen')}),flush=True)


if __name__=='__main__':main()
