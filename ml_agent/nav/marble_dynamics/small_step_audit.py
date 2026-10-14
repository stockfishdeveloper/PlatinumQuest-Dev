"""Audit smaller-frame mechanics and measured queue timing independently.

Recorded future native moves are exogenous audit inputs here. They are NEVER
supplied to the proposed-action predictor or used to replace predicted states.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.bridge import translated
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.rollout import apply_control


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);a=p.parse_args()
    summary=json.loads((a.data/'summary.json').read_text());reports=[]
    for case in summary['cases']:
        name=case['map'];folder=a.data/name;logs=json.loads((folder/'decisions.json').read_text())
        first=logs[0]['dynamics']['native']['frame_clock_ms'];last=logs[-1]['next_frame_clock_ms']
        rows=[r for r in load_rows(folder/'native.jsonl.gz') if first<=r['sim_ms']<last]
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        state=copy.deepcopy(logs[0]['dynamics']['native']['state']);measurements=[];events=[];moves=[];cameras=[]
        for row in rows:
            state['clock_ms']=row['sim_ms']
            control=dict(move=row['move'],events=row['control_events'])
            apply_control(state,control);state,ev=engine.step(state,row['move'],row['dt_ms'])
            measurements.append(errors(state,row['after']));events.append(ev)
        # The prior macro reply updates camera at the first16ms frame; its
        # translated move starts at the second frame. Current proposed reply
        # waits for the next64ms observation/parse, never arrives mid-macro.
        for log in logs:
            dynamics=log['dynamics'];begin=dynamics['native']['frame_clock_ms']
            macro=[r for r in rows if begin<=r['sim_ms']<log['next_frame_clock_ms']]
            next_move,camera,_,_=translated(dynamics['bridge_action'],dynamics['pow_yaw_hold'],dynamics['pow_yaw'])
            for i,row in enumerate(macro):
                expected=dynamics['native']['pending_move'] if i==0 else next_move
                if expected!=row['move']:moves.append(dict(reply=log['index'],frame=i,expected=expected,actual=row['move']))
                actual=[dict(kind=e['kind'],value=e['value']) for e in row['control_events']]
                expected_camera=camera if i==0 else []
                if len(actual)!=len(expected_camera) or any(a['kind']!=b['kind'] or abs(a['value']-b['value'])>1e-7 for a,b in zip(actual,expected_camera)):
                    cameras.append(dict(reply=log['index'],frame=i,expected=expected_camera,actual=actual))
        report=dict(map=name,native_frames=len(rows),whole_open_loop_maximum_errors=np.max(measurements,axis=0).tolist(),
            whole_open_loop_endpoint_errors=measurements[-1],queue_move_discrepancies=moves,
            camera_discrepancies=cameras,ambiguity_frames=[i for i,e in enumerate(events) if any(v['kind']=='ambiguous_contact' for v in e)],
            native_sha256=sha(folder/'native.jsonl.gz'),audit_source_sha256=sha(__file__),training=False,
            caveat='Native future controls are audit only; open-loop states never corrected. This does not validate future proposed-action forecasting yet.')
        reports.append(report);atomic_json(a.data/'small_step_audit.json',dict(cases=reports,training=False))
        print(json.dumps({k:report[k] for k in ['map','native_frames','whole_open_loop_maximum_errors','whole_open_loop_endpoint_errors','queue_move_discrepancies','camera_discrepancies','ambiguity_frames']}),flush=True)


if __name__=='__main__':main()
