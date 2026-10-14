"""Diagnostic future sensitivity, never a calibrated uncertainty/safety bound.

Perturb only the INITIAL position/velocity/spin; all future controls and complete
map geometry remain fixed. No future measured state is injected into a variant.
"""
import json
from pathlib import Path

import numpy as np
from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.evaluate import load_rows,continuous_prefix
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.rollout import predict
from nav.marble_dynamics.state_copy import clone
from nav.marble_dynamics.record import ML,sha


def main():
    output=ML/'dynamics_reports/rich_mechanics_initial_sensitivity_20261012'
    output.mkdir(exist_ok=False);results=[]
    selections=[('rich_mechanics_train_20261012','Sprawl_Hunt',11),
        ('rich_mechanics_holdout_20261012','Bowl_Hunt',13),
        ('rich_mechanics_train_20261012','TreasureBox_Hunt',26),
        ('rich_mechanics_train_20261012','Cragmire_Hunt',25)]
    for dataset,name,ident in selections:
        root=ML/'datasets/marble_dynamics'/dataset/name
        trial=next(t for t in json.loads((root/'manifest.json').read_text())['trials'] if t['id']==ident)
        native=root/trial['file'];rows,excluded=continuous_prefix(trial,load_rows(native))
        initial=rows[0]['before'];controls=[dict(move=r['move'],events=r['control_events']) for r in rows]
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        nominal=predict(engine,initial,controls,dt_ms=16)
        positions=np.asarray([s['position'] for s in nominal['states']]);variants=[];paths=[]
        for field,epsilon in [('position',1e-8),('velocity',1e-7),('spin',1e-6)]:
            for axis in range(3):
                for sign in (-1,1):
                    start=clone(initial);start[field][axis]+=sign*epsilon
                    result=predict(engine,start,controls,dt_ms=16)
                    count=len(result['states']);path=np.asarray([s['position'] for s in result['states']])
                    distances=np.linalg.norm(path-positions[:count],axis=1)
                    first=next((i for i,d in enumerate(distances) if d>.05),None)
                    variants.append(dict(field=field,axis=axis,delta=sign*epsilon,
                        maximum_position_difference_from_nominal_u=float(np.max(distances)),
                        terminal_position_difference_from_nominal_u=float(distances[-1]),
                        first_state_over_005u=first,ambiguous_intervals=result['ambiguous_intervals'],failure=result['failure']))
                    paths.append(path.tolist())
        value=dict(map=name,dataset=dataset,trial=ident,initial_only=True,steps=len(rows),
            nominal_ambiguities=nominal['ambiguous_intervals'],excluded_native_samples=excluded,
            maximum_variant_position_difference_u=max(v['maximum_position_difference_from_nominal_u'] for v in variants),
            variants=variants,native_sha256=sha(native),geometry_sha256=engine.mesh.info['mesh_sha256'],
            contract='Sensitivity diagnosis with arbitrary disclosed tiny initial perturbations, not statistical confidence, coverage guarantee, native replay or planner-approved action. Future inputs are recorded external controls; no future native state enters any forecast.')
        atomic_json(output/(name+'.'+str(ident)+'.json'),dict(**value,
            nominal_positions=positions.tolist(),variant_positions=paths))
        results.append(value);print(json.dumps(dict(map=name,trial=ident,
            max_variant_difference_u=value['maximum_variant_position_difference_u'])),flush=True)
    atomic_json(output/'summary.json',dict(cases=results,training=False,planner_integrated=False,
        source_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in
            ('sensitivity_probe','physics','rollout','state_copy')}))


if __name__=='__main__':main()
