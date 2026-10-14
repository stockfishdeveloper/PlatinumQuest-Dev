"""Contact sensitivity experiment, not a calibrated probability or safety bound.

All ensemble members run open loop from microscopically perturbed initial states.
Actual future motion is used only for scoring. No member is selected for control.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import load_rows,continuous_prefix,errors,stats
from nav.marble_dynamics.record import ML


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ML/'datasets/marble_dynamics/contact_diagnostics_20261012')
    p.add_argument('--map',default='CubeIsle_Hunt');p.add_argument('--trial',type=int,default=33)
    p.add_argument('--start',type=int,default=40);p.add_argument('--horizon',type=float,default=1.)
    p.add_argument('--members',type=int,default=24)
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/contact_sensitivity_20261012')
    args=p.parse_args();folder=args.data/args.map
    manifest=json.loads((folder/'manifest.json').read_text());trial=next(t for t in manifest['trials'] if t['id']==args.trial)
    source=load_rows(folder/trial['file']);source,removed=continuous_prefix(trial,source)
    steps=int(np.ceil(args.horizon/(source[args.start]['dt_ms']*.001)))
    rows=source[args.start:args.start+steps]
    mesh=CollisionMap(ML/'collision_maps'/(args.map+'.npz'));engine=GeneralPhysics(mesh)
    actual=[rows[0]['before']]+[r['after'] for r in rows]
    times=np.concatenate([[0.],np.cumsum([r['dt_ms']*.001 for r in rows])]);rng=np.random.RandomState(612)
    cases=[];full_members=[];failed=[]
    for member in range(args.members):
        state=copy.deepcopy(rows[0]['before'])
        if member:
            for key,sigma in [('position',1e-8),('velocity',1e-6),('spin',1e-6)]:
                state[key]=(np.asarray(state[key])+rng.normal(0,sigma,3)).tolist()
        predicted=[state];ambiguities=[]
        for i,row in enumerate(rows):
            state=predicted[-1].copy();state['clock_ms']=row['sim_ms']
            for key in ('yaw','next_yaw','pitch','next_pitch','smooth_camera'):state[key]=row['before'][key]
            try:state,events=engine.step(state,row['move'],row['dt_ms'])
            except Exception as exc:failed.append(dict(member=member,sample=i,reason=str(exc)));break
            predicted.append(state)
            if any(e['kind']=='ambiguous_contact' for e in events):ambiguities.append(i)
        count=len(predicted);errs=[errors(a,b) for a,b in zip(predicted,actual)]
        case=dict(id=1000+member,family=trial['family']+'_hypothesis',powerup=trial['spec']['powerup'],split=manifest['split'],
                  time_s=times[:count].tolist(),position_error=[e[0] for e in errs],final_errors=errs[-1],
                  max_position_error_u=max(e[0] for e in errs),truncated_external_samples=removed,
                  rollout_failure=next((f for f in failed if f['member']==member),None),
                  ambiguous_contact_intervals=ambiguities,native_use_events=[],
                  control_basis='Recorded applied controls/camera; open-loop state; initial microscopic perturbation only')
        for basis,states in [('actual',actual[:count]),('predicted',predicted)]:
            for key in ('position','velocity','spin'):case[basis+'_'+key]=[s[key] for s in states]
        cases.append(case)
        if count==len(actual):full_members.append(case['predicted_position'])
    array=np.asarray(full_members);truth=np.asarray([s['position'] for s in actual])
    if not len(array):raise RuntimeError('No complete hypotheses')
    nearest=np.linalg.norm(array-truth[None,:,:],axis=2).min(axis=0)
    extent=(array.max(axis=0)-array.min(axis=0))
    report=dict(map=args.map,source_trial=args.trial,start_sample=args.start,horizon_s=float(times[-1]),
                source_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
                predictor_source_sha256=hashlib.sha256(Path(__file__).with_name('physics.py').read_bytes()).hexdigest(),
                contact_order_source_sha256=hashlib.sha256(Path(__file__).with_name('contact_order.py').read_bytes()).hexdigest(),
                members=args.members,complete_members=len(array),failures=failed,
                initial_sigma=dict(position_u=1e-8,velocity_u_per_s=1e-6,spin_rad_per_s=1e-6),
                unperturbed_final_error=cases[0]['final_errors'],nearest_member_final_position_error_u=float(nearest[-1]),
                maximum_nearest_member_position_error_u=float(nearest.max()),
                final_coordinate_spread_u=extent[-1].tolist(),
                maximum_coordinate_spread_u=extent.max(axis=0).tolist(),
                final_member_errors=stats([c['final_errors'] for c in cases if c['rollout_failure'] is None]),
                calibrated=False,planner_approved=False,
                caveat='Sensitivity demonstration on a selected dev fixture; neither coverage probability nor a safety certification.')
    atomic_json(args.output/(args.map+'.sensitivity.json'),report)
    atomic_json(args.output/(args.map+'.trajectories.json'),dict(map=args.map,geometry_sha256=mesh.info['mesh_sha256'],trajectories=cases))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
