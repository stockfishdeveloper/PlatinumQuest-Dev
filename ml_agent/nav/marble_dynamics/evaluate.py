"""One-step and open-loop general-physics checks, with viewer-ready trajectories.

Open-loop position, velocity and spin are never replaced with observations. Inputs
are recorded actually applied moves and camera settings, not hypothetical policy commands.
Server warps delimit recordings; they are not labeled as continuous physical motion.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.physics import GeneralPhysics,vector_state
from nav.marble_dynamics.record import DEFAULT_OUTPUT,ML
from nav.marble_dynamics.outcomes import compare as compare_outcomes


def load_rows(path):
    return [r for r in map(json.loads,gzip.decompress(path.read_bytes()).decode().splitlines()) if r['kind']=='transition']


def continuous_prefix(trial,rows):
    warps=[e['index'] for e in trial.get('external_state_changes',[]) if e['field']=='position' and e['error']>.001]
    stop=min(warps) if warps else len(rows)
    # Script-driven size/datablock/force changes are discrete events. A physics
    # rollout must not secretly receive their future parameter values.
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        if i+1>=stop:break
        keys=('radius','datablock','parameters','gravity_direction','held')
        if any(a['after'][k]!=b['before'][k] for k in keys):stop=i+1;break
        if any(np.linalg.norm(np.asarray(a['after'][k])-b['before'][k])>1e-5 for k in ('velocity','spin')):
            stop=i+1;break
    for i,r in enumerate(rows[:stop]):
        if r['before']['mode']!=0 or r['after']['mode']!=0 or r['before']['oob']:
            stop=i;break
        p=np.asarray(r['after']['position'])-r['before']['position']
        speed=max(np.linalg.norm(r['before']['velocity']),np.linalg.norm(r['after']['velocity']))
        if np.linalg.norm(p)>speed*r['dt_ms']*.001+.3:
            stop=i;break
    return rows[:stop],len(rows)-stop


def stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(count=0)
    return dict(count=len(x),median=np.percentile(x,50,axis=0).tolist(),p95=np.percentile(x,95,axis=0).tolist(),max=x.max(axis=0).tolist(),
                units=['position_u','velocity_u_per_s','spin_rad_per_s'])


def errors(pred,actual):
    delta=vector_state(pred)-vector_state(actual)
    return [float(np.linalg.norm(delta[k:k+3])) for k in (0,3,6)]


def evaluate_map(name,args):
    folder=args.data/name
    manifest=json.loads((folder/'manifest.json').read_text())
    mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
    if mesh.info['mesh_sha256']!=manifest['geometry_sha256']:raise RuntimeError('Geometry hash changed')
    engine=GeneralPhysics(mesh)
    step_errors=[];by_family=defaultdict(list);cases=[];failures=[];worst=[];empty_prefix=[]
    coverage=Counter();skipped=0;started=time.perf_counter()
    source_files=[Path(__file__),Path(__file__).with_name('physics.py'),Path(__file__).with_name('contact_order.py'),Path(__file__).with_name('outcomes.py'),
                  ML/'nav/collision/mesh.py',ML/'hxDif.py']
    source_hashes={str(p.relative_to(ML)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    version=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    for trial in manifest['trials']:
        if not trial['valid']:continue
        native_rows=load_rows(folder/trial['file'])
        rows,removed=continuous_prefix(trial,native_rows)
        skipped+=removed
        if not rows:
            empty_prefix.append(dict(trial=trial['id'],family=trial['family'],removed_samples=removed,
                                     initial_mode=native_rows[0]['before']['mode'] if native_rows else None,
                                     initial_oob=native_rows[0]['before']['oob'] if native_rows else None))
            continue
        sample_indices=np.arange(len(rows)) if not args.stride else np.arange(0,len(rows),args.stride)
        for j in sample_indices:
            row=rows[j]
            try:
                before=row['before'].copy();before['clock_ms']=row['sim_ms']
                state,event=engine.step(before,row['move'],row['dt_ms'])
                error=errors(state,row['after']);step_errors.append(error);by_family[trial['family']].append(error)
                coverage['contact_steps']+=int(row['telemetry']['contact_steps']>0)
                coverage['collision_steps']+=int(row['telemetry']['collisions']>0)
                coverage['jump_steps']+=int(row['move']['jump'])
                coverage['powerup_use_steps']+=int(row['move']['use'])
                if error[0]>.01 or error[1]>.5 or error[2]>5:
                    worst.append(dict(trial=trial['id'],sample=int(j),family=trial['family'],errors=error,
                                      position=row['before']['position'],native_telemetry=row['telemetry']))
            except Exception as exc:
                failures.append(dict(trial=trial['id'],sample=int(j),reason=str(exc)))
        actual=[rows[0]['before']]+[r['after'] for r in rows]
        predicted=[rows[0]['before']];predicted_events=[];rollout_failure=None
        for i,row in enumerate(rows):
            # Camera aim is an applied control in this bridge. Physics state stays predicted.
            state=predicted[-1].copy()
            state['clock_ms']=row['sim_ms']
            for key in ('yaw','next_yaw','pitch','next_pitch','smooth_camera'):
                if key in row['before']:state[key]=row['before'][key]
            try:
                result,event=engine.step(state,row['move'],row['dt_ms'])
                predicted.append(result);predicted_events.append(event)
            except Exception as exc:
                rollout_failure=dict(sample=i,reason=str(exc));break
        count=min(len(actual),len(predicted))
        times=np.concatenate([[0.],np.cumsum([r['dt_ms']*.001 for r in rows])])[:count]
        per_step=[errors(p,a) for p,a in zip(predicted[:count],actual[:count])]
        p_error=np.asarray(per_step)[:,0]
        case=dict(id=trial['id'],family=trial['family'],powerup=trial['spec']['powerup'],split=manifest['split'],
                  source_file=trial['file'],samples=count-1,time_s=times.tolist(),
                  actual_position=[a['position'] for a in actual[:count]],
                  predicted_position=[p['position'] for p in predicted[:count]],
                  actual_velocity=[a['velocity'] for a in actual[:count]],
                  predicted_velocity=[p['velocity'] for p in predicted[:count]],
                  actual_spin=[a['spin'] for a in actual[:count]],
                  predicted_spin=[p['spin'] for p in predicted[:count]],
                  position_error=p_error.tolist(),errors=stats(per_step),
                  max_position_error_u=float(p_error.max()),final_errors=per_step[-1],
                  native_contacts=[r['telemetry'] for r in rows[:count-1]],
                  truncated_external_samples=removed,rollout_failure=rollout_failure,
                  control_basis='recorded applied moves and absolute camera settings; open-loop p/v/spin',
                  predicted_model='independent engine-mechanics baseline; uncalibrated uncertainty')
        case['native_use_events']=[dict(sample=i,time_s=float(times[i]),held_id=int(r['before']['held']))
                                   for i,r in enumerate(rows[:count-1]) if r['move']['use'] and r['before']['held']]
        case['pair_group']=trial.get('pair_group')
        case['intervention']=trial.get('intervention')
        case['ambiguous_contact_intervals']=[i for i,ev in enumerate(predicted_events) if any(e['kind']=='ambiguous_contact' for e in ev)]
        # Duration-specific errors expose drift hidden by aggregate one-step accuracy.
        case['horizon_errors']={str(h):per_step[int(np.searchsorted(times,h))] for h in (.064,.25,.5,1.,2.,3.,5.) if times[-1]>=h}
        case['outcomes']=compare_outcomes(rows,actual,predicted,predicted_events,times)
        case['rollout_complete']=rollout_failure is None
        case['native_state_fields']=dict(previous_best_contact='best_contact_normal' in rows[0]['before'],
                                        exact_effect_clock='effect_end_clock_ms' in rows[0]['before'])
        cases.append(case)
    replay=manifest['replay_checks']
    report=dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],split=manifest['split'],one_step=stats(step_errors),
                model_version_sha256=version,source_hashes=source_hashes,
                dataset_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),one_step_stride=args.stride,
                one_step_by_family={k:stats(v) for k,v in by_family.items()},coverage=dict(coverage),
                skipped_external_or_invalid_samples=skipped,failures=failures,worst_samples=sorted(worst,key=lambda r:r['errors'][0],reverse=True)[:50],
                accepted_native_fixtures=sum(t['valid'] for t in manifest['trials']),zero_length_continuous_prefix_trials=empty_prefix,
                replay_checks=len(replay),replay_max_errors={k:max((r.get('max_errors',{}).get(k,0) for r in replay),default=0) for k in ('position','velocity','spin')},
                replay_length_differences=sum(not r.get('same_length',False) for r in replay),
                rollout_final=stats([c['final_errors'] for c in cases if c['rollout_complete']]),
                rollout_failures=[dict(trial=c['id'],**c['rollout_failure']) for c in cases if c['rollout_failure']],
                outcomes=[dict(trial=c['id'],family=c['family'],**c['outcomes']) for c in cases],
                native_source_order_checks=engine.contact_order.mapping_checks,
                native_source_order_unranked_contacts=engine.contact_order.unranked_contacts,
                rollout_horizons={str(h):stats([c['horizon_errors'][str(h)] for c in cases if str(h) in c['horizon_errors']]) for h in (.064,.25,.5,1.,2.,3.,5.)},
                trajectories=len(cases),elapsed_s=time.perf_counter()-started,
                planner_approved=False,uncertainty_calibrated=False)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/(name+'.report.json'),report)
    atomic_json(args.output/(name+'.trajectories.json'),dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],model_version_sha256=version,trajectories=cases))
    print(json.dumps(dict(map=name,one_step=report['one_step'],rollout_final=report['rollout_final'],
                         failures=len(failures),external_skipped=skipped,coverage=dict(coverage),elapsed_s=round(report['elapsed_s'],1))),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/pilot_20261011')
    p.add_argument('--maps',nargs='+')
    p.add_argument('--stride',type=int,default=0,help='0 checks every native transition')
    args=p.parse_args()
    names=args.maps or [p.parent.name for p in sorted(args.data.glob('*/manifest.json'))]
    for name in names:
        report=evaluate_map(name,args)
        all_reports=[json.loads(p.read_text()) for p in sorted(args.output.glob('*.report.json'))]
        current=[r for r in all_reports if r.get('model_version_sha256')==report['model_version_sha256']]
        stale=[r['map'] for r in all_reports if r.get('model_version_sha256')!=report['model_version_sha256']]
        atomic_json(args.output/'summary.json',dict(maps=current,stale_reports=stale,policy_training_started=False,
                    predictor='independent native-mechanics port',inputs='applied moves; command-delay migration remains separate'))


if __name__=='__main__':main()
