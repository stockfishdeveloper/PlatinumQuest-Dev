"""Audit fixed-body Mega physics; explicitly exclude its script transitions.

This uses recorded exogenous native controls, not proposed policy inputs. Every
constant-body interval starts from one measured state. No later measured state
or parameter is supplied to that interval's prediction.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json
from nav.marble_dynamics.evaluate import load_rows, errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML, sha
from nav.marble_dynamics.rollout import predict


def signature(state):
    return json.dumps({k:state[k] for k in ('radius','parameters','datablock','mode','oob')},sort_keys=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path)
    a=p.parse_args();prepared=json.loads((a.data/'prepared.json').read_text())
    observed=json.loads((a.data/'observations.json').read_text())
    first=observed[0]['dynamics']['native']['frame_clock_ms']
    last=observed[-1]['actual_next']['native']['frame_clock_ms']
    rows=[r for r in load_rows(a.data/'native.jsonl.gz') if first<=r['sim_ms']<last]
    assert len(rows)==4*len(observed), 'Incomplete16ms recording'
    segments=[];cuts=[];start=0
    for i in range(1,len(rows)):
        previous=rows[i-1]['after'];current=rows[i]['before']
        discontinuity=errors(previous,current)
        if signature(current)!=signature(rows[start]['before']) or max(discontinuity)>1e-5:
            segments.append(rows[start:i]);start=i
            cuts.append(dict(native_frame=i,clock_ms=rows[i]['sim_ms'],
                physical_boundary_errors=discontinuity,before_radius=previous['radius'],
                after_radius=current['radius'],reason='observed script/body transition; not predicted'))
    segments.append(rows[start:])
    engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(prepared['map']+'.npz')))
    checks=[]
    for segment in segments:
        controls=[dict(move=r['move'],events=r['control_events']) for r in segment]
        forecast=predict(engine,segment[0]['before'],controls,16)
        assert forecast['failure'] is None, forecast['failure']
        assert len(forecast['states'])==len(segment)+1
        actual=[segment[0]['before']]+[r['after'] for r in segment]
        measures=[errors(pred,native) for pred,native in zip(forecast['states'],actual)]
        checks.append(dict(start_clock_ms=segment[0]['sim_ms'],native_frames=len(segment),
            radius=segment[0]['before']['radius'],initial_gravity=segment[0]['before']['parameters']['gravity'],
            maximum_errors=np.max(measures,axis=0).tolist(),endpoint_errors=measures[-1],
            ambiguous_intervals=forecast['ambiguous_intervals']))
    report=dict(map=prepared['map'],native_frames=len(rows),segments=checks,
        excluded_script_transitions=cuts,source_hashes={str(path):sha(path) for path in
            [Path(__file__),Path(__file__).with_name('physics.py'),Path(__file__).with_name('rollout.py'),
             a.data/'native.jsonl.gz',a.data/'observations.json']},training=False,
        mega_activation_forecast_verified=False,
        contract='Observed constant-body intervals only. Native exogenous controls are audit inputs. Script growth, impulses and shrink are explicitly excluded and are not predicted; this does not enable Mega in planning.')
    atomic_json(a.data/'fixed_body_audit.json',report)
    print(json.dumps(report['segments']),flush=True)


if __name__=='__main__':main()
