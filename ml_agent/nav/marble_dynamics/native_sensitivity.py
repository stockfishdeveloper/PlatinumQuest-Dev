"""Repeat and perturb a disclosed development contact fixture in the real game."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import atomic_json,CollisionMap
from nav.marble_dynamics.record import ML,RUNTIME,game,run_trial,archive,sha
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.rollout import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=9054)
    p.add_argument('--output',type=Path,default=ML/'datasets/marble_dynamics/native_sensitivity_20261012')
    a=p.parse_args();name='Concentric_Hunt';dest=a.output/name
    if (dest/'manifest.json').exists():raise RuntimeError('Use new output; retain previous evidence')
    dest.mkdir(parents=True,exist_ok=True)
    source=ML/'datasets/marble_dynamics/timers_20261012'/name/'manifest.json'
    spec=next(t['spec'] for t in json.loads(source.read_text())['trials'] if t['id']==8)
    spec['frames']=90
    variants=[('baseline',None,0.,0),('baseline_repeat',None,0.,0)]
    variants += [('velocity_'+str(axis)+'_'+str(sign),'velocity',sign*2e-5,axis) for axis in range(3) for sign in (-1,1)]
    variants += [('position_x_'+str(sign),'position',sign*2e-5,0) for sign in (-1,1)]
    mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));engine=GeneralPhysics(mesh);trials=[]
    with game(name,a.port) as (env,proc):
        for i,(label,key,delta,axis) in enumerate(variants):
            changed=copy.deepcopy(spec)
            if key:changed[key][axis]+=delta
            raw=RUNTIME/(name+'.native_sensitivity_%02d.jsonl'%i)
            result=run_trial(env,changed,raw);trial=dict(id=i,label=label,spec=changed,**result)
            if result['valid']:
                file=dest/('trial_%02d.jsonl.gz'%i);archive(raw,file);rows=load_rows(file)
                trial.update(file=file.name,raw_sha256=sha(raw),initial=rows[0]['before'],endpoint=rows[-1]['after'],
                             native_peak_z=max(r['after']['position'][2] for r in rows))
                controls=[dict(move=r['move'],events=r['control_events']) for r in rows]
                model=predict(engine,rows[0]['before'],controls)
                trial.update(prediction_failure=model['failure'],ambiguous_intervals=model['ambiguous_intervals'],
                             nominal_endpoint_errors=errors(model['states'][-1],rows[len(model['states'])-2]['after']))
            trials.append(trial)
            atomic_json(dest/'manifest.json',dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],trials=trials,
                        source_fixture=str(source),source_trial=8,split='dev',owned_game_pid=proc.pid,
                        probe_executable_sha256=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe')))
            print(json.dumps(dict(label=label,valid=result['valid'],peak=trial.get('native_peak_z'),errors=trial.get('nominal_endpoint_errors'))),flush=True)
    valid=[t for t in trials if t['valid']];base=valid[0]
    report=dict(map=name,cases=len(trials),valid=len(valid),
                actual_initial_position_max_difference_u=max(float(np.linalg.norm(np.asarray(t['initial']['position'])-base['initial']['position'])) for t in valid),
                actual_initial_velocity_max_difference_u_per_s=max(float(np.linalg.norm(np.asarray(t['initial']['velocity'])-base['initial']['velocity'])) for t in valid),
                native_peak_height_range_u=[min(t['native_peak_z'] for t in valid),max(t['native_peak_z'] for t in valid)],
                maximum_native_endpoint_difference_u=max(float(np.linalg.norm(np.asarray(t['endpoint']['position'])-base['endpoint']['position'])) for t in valid),
                warning='Selected development sensitivity, not calibrated population coverage or a general landing guarantee',
                policy_training_started=False)
    atomic_json(a.output/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
