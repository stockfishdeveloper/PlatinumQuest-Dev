"""Paired failed-search timing with identical accepted futures and selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.terrain import TerrainMap,TerrainGrid
from nav.routing import RoutePlanner
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.diagnose_search import reconstruct_navigator
from nav.marble_dynamics import search as current


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ML/'dynamics_reports/rejection_speed_20261012/search.py')
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/rejection_speed_20261012/equivalence.json')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('before_rejection_speed',a.reference)
    before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
    paths=[ML/'dynamics_reports/reliability_optimized_20261012/Sprawl_Hunt/61103/physics_plan/terminal_search.json',
           ML/'dynamics_reports/reliability_reuse_20261012/Bowl_Hunt/71104/physics_plan/terminal_search.json',
           ML/'dynamics_reports/reliability_braking_regression_20261012/Sprawl_Hunt/61103/physics_plan/terminal_search.json']
    cases=[]
    for path in paths:
        name=path.parents[2].name;case=json.loads(path.read_text())
        terrain=TerrainGrid(TerrainMap.resolve(name));engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        raw=np.asarray(case['raw_observation'],dtype=np.float32)
        nav=reconstruct_navigator(terrain,ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',
                                  json.loads(path.with_name('decisions.json').read_text()),raw)
        for horizon in (40,80):
            times=[[],[]];result=[]
            for index,module in enumerate((before,current)):
                for _ in range(3):
                    route=RoutePlanner(terrain);start=time.perf_counter()
                    chosen,pairs=module.search(engine,route,case['dynamics'],raw,case['target'][:3],nav,horizon,brake_replies=48)
                    times[index].append(time.perf_counter()-start)
                result.append((chosen,pairs))
            old,new=result
            self_valid=lambda pairs:{info['label']:rollout for info,rollout in pairs if info['rejected'] is None}
            old_valid,new_valid=self_valid(old[1]),self_valid(new[1])
            assert old_valid.keys()==new_valid.keys(),(path,horizon,'acceptance changed')
            for label in old_valid:
                for key in ('states','controls','events','proposed'):
                    assert old_valid[label][key]==new_valid[label][key],(path,horizon,label,key)
            old_label=old[0][0]['label'] if old[0] else None;new_label=new[0][0]['label'] if new[0] else None
            assert old_label==new_label,(path,horizon,'selected candidate changed')
            t0,t1=[float(np.median(values)) for values in times]
            record=dict(map=name,source=str(path),source_sha256=sha(path),horizon=horizon,
                before_s=t0,after_s=t1,speedup=t0/t1,selected_label=new_label,
                identical_accepted_futures=True,accepted_labels=list(new_valid),
                before_simulated_replies=sum(len(r['controls']) for _,r in old[1] if r),
                after_simulated_replies=sum(len(r['controls']) for _,r in new[1] if r))
            cases.append(record);print(json.dumps(record),flush=True)
            atomic_json(a.output,dict(cases=cases,reference_sha256=sha(a.reference),current_sha256=sha(Path(current.__file__)),
                contract='Three-repeat median on saved development failures; rejected futures may truncate at first ambiguity. Accepted futures/selection identical.',policy_training_started=False))


if __name__=='__main__':main()
