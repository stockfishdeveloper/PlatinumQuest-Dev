"""Separate per-step native physics mismatch from long-forecast amplification."""
import json
from pathlib import Path

import numpy as np
from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.evaluate import load_rows,continuous_prefix,errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.state_copy import clone
from nav.marble_dynamics.record import ML,sha


def main():
    output=ML/'dynamics_reports/rich_mechanics_error_diagnosis_20261012'
    output.mkdir(exist_ok=False);results=[];sources=[]
    for split in ('train','holdout'):
        report=ML/'dynamics_reports'/('rich_mechanics_'+split+'_check_20261012')
        cases=[c for c in json.loads((report/'cases.json').read_text()) if 'endpoint_errors' in c]
        unflagged=sorted((c for c in cases if not c['ambiguous_intervals']),
            key=lambda c:c['maximum_position_error_u'],reverse=True)[:3]
        worst=max(cases,key=lambda c:c['maximum_position_error_u'])
        for case in unflagged+[worst]:
            name=case['map'];dataset=case['dataset'];root=ML/'datasets/marble_dynamics'/dataset/name
            manifest=json.loads((root/'manifest.json').read_text())
            trial=next(t for t in manifest['trials'] if t['id']==case['trial'])
            path=root/trial['file'];rows,excluded=continuous_prefix(trial,load_rows(path))
            engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
            measured=[];flags=[];worst_step=None;worst_error=-1.
            for index,row in enumerate(rows):
                before=clone(row['before']);before['clock_ms']=row['sim_ms']
                move=apply_control(before,dict(move=row['move'],events=row['control_events']))
                after,events=engine.step(before,move,row['dt_ms'])
                error=errors(after,row['after']);measured.append(error)
                if any(e['kind']=='ambiguous_contact' for e in events):flags.append(index)
                if error[0]>worst_error:
                    worst_error=error[0];worst_step=dict(index=index,errors=error,
                        native=row,predicted_after=after,predicted_events=events)
            trace=json.loads((report/(name+'.trajectories.json')).read_text())['trajectories']
            trajectory=next(t for t in trace if t['id']==case['trial'] and t['source_dataset']==dataset)
            first_large=next((i for i,v in enumerate(trajectory['position_error']) if v>.001),None)
            result=dict(map=name,dataset=dataset,trial=case['trial'],family=case['family'],
                one_step_error_max=np.max(measured,axis=0).tolist(),one_step_error_p95=np.percentile(measured,95,axis=0).tolist(),
                one_step_ambiguities=flags,whole_forecast_ambiguities=case['ambiguous_intervals'],
                whole_forecast_max_position_error_u=case['maximum_position_error_u'],
                first_whole_forecast_position_error_over_001u_state=first_large,
                worst_step=worst_step,excluded_native_samples=excluded,native_sha256=sha(path),
                contract='One-step uses each native BEFORE state deliberately as a separate test; NOT an open-loop forecast. Whole forecast remains one initial state plus recorded external controls. No fitting or data correction.')
            results.append(result);sources.append(dict(path=str(path),sha256=sha(path)))
            atomic_json(output/(name+'.'+str(case['trial'])+'.json'),result)
            print(json.dumps({k:result[k] for k in ('map','trial','one_step_error_max',
                'whole_forecast_max_position_error_u','first_whole_forecast_position_error_over_001u_state',
                'whole_forecast_ambiguities')}),flush=True)
    atomic_json(output/'summary.json',dict(cases=[{k:v for k,v in r.items() if k!='worst_step'} for r in results],
        sources=sources,source_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in
            ('diagnose_rich_errors','physics','rollout','evaluate')},training=False))


if __name__=='__main__':main()
