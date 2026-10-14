"""Revalidate spin/rebound/jump/powerup native archives without future-state input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.evaluate import load_rows,continuous_prefix,errors
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.outcomes import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=['controls_20261012','wall_jumps_20261012'])
    p.add_argument('--maps',nargs='+',default=['Concentric_Hunt','CubeIsle_Hunt','Sprawl_Hunt','Bowl_Hunt'])
    p.add_argument('--output',type=Path,default=ML/'dynamics_reports/mechanics_recheck_20261012')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for name in a.maps:
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));engine=GeneralPhysics(mesh);trajectories=[]
        for dataset in a.datasets:
            folder=ML/'datasets/marble_dynamics'/dataset/name
            manifest=json.loads((folder/'manifest.json').read_text())
            for trial in manifest['trials']:
                if not trial['valid']:continue
                file=folder/trial['file'];native=load_rows(file);rows,excluded=continuous_prefix(trial,native)
                if not rows:
                    cases.append(dict(map=name,dataset=dataset,trial=trial['id'],excluded_native_samples=excluded,reason='no_continuous_playable_prefix'));continue
                if any('control_events' not in r for r in rows):
                    raise ValueError('Legacy recording lacks complete camera command inputs: '+str(file))
                controls=[dict(move=r['move'],events=r['control_events']) for r in rows]
                result=predict(engine,rows[0]['before'],controls,dt_ms=rows[0]['dt_ms'])
                count=len(result['states']);actual=[rows[0]['before']]+[r['after'] for r in rows[:count-1]]
                per=[errors(p,q) for p,q in zip(result['states'],actual)]
                outcomes=compare(rows,actual,result['states'],result['events'],np.asarray(result['time_s']))
                record=dict(map=name,dataset=dataset,trial=trial['id'],family=trial['family'],intervention=trial.get('intervention'),
                            simulated_s=result['time_s'][-1],initial_only=True,endpoint_errors=per[-1],
                            maximum_position_error_u=max(e[0] for e in per),failure=result['failure'],
                            ambiguous_intervals=result['ambiguous_intervals'],excluded_native_samples=excluded,
                            outcomes=outcomes,native_archive_sha256=sha(file))
                cases.append(record)
                trajectories.append(dict(id=trial['id'],source_dataset=dataset,family=trial['family'],powerup=trial['spec']['powerup'],split='revalidation',
                    time_s=result['time_s'],actual_position=[s['position'] for s in actual],predicted_position=[s['position'] for s in result['states']],
                    actual_velocity=[s['velocity'] for s in actual],predicted_velocity=[s['velocity'] for s in result['states']],
                    actual_spin=[s['spin'] for s in actual],predicted_spin=[s['spin'] for s in result['states']],
                    position_error=[e[0] for e in per],max_position_error_u=record['maximum_position_error_u'],final_errors=per[-1],
                    truncated_external_samples=excluded,rollout_failure=result['failure'],native_use_events=[],
                    ambiguous_contact_intervals=result['ambiguous_intervals'],outcomes=outcomes,
                    control_basis='One initial full physical state plus recorded exogenous Move/camera commands; no future physical/camera state'))
                print(json.dumps({k:record[k] for k in ('map','dataset','trial','maximum_position_error_u','failure')}),flush=True)
        atomic_json(a.output/(name+'.trajectories.json'),dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],trajectories=trajectories))
        atomic_json(a.output/'cases.json',cases)
    valid=[r for r in cases if 'endpoint_errors' in r]
    landed=[r['outcomes']['landing'] for r in valid if r['outcomes']['landing']['both_detected']]
    summary=dict(cases=len(cases),continuous_cases=len(valid),failures=sum(r['failure'] is not None for r in valid),
                 endpoint_error_p95=np.percentile([r['endpoint_errors'] for r in valid],95,axis=0).tolist(),
                 endpoint_error_max=np.max([r['endpoint_errors'] for r in valid],axis=0).tolist(),
                 first_landing_compared=len(landed),max_first_landing_error_u=max((r['position_error_u'] for r in landed),default=None),
                 max_one_second_continuation_error_u=max((r.get('one_second_continuation_position_error_u',0.) for r in landed),default=None),
                 ambiguity_cases=sum(bool(r['ambiguous_intervals']) for r in valid),
                 source_hashes={p.name:sha(p) for p in [Path(__file__).with_name(name+'.py') for name in
                    ['mechanics_check','physics','state_copy','rollout','evaluate','outcomes','contact_order']]+[ML/'nav/collision/mesh.py']},
                 warning='Existing native intervention revalidation, not newly sampled native trials or calibrated safety; event-excluded prefixes disclosed',policy_training_started=False)
    atomic_json(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
