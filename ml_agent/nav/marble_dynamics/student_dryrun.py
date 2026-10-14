"""Verify complete-map motion preparation and time unfitted forward passes only.

No optimizer, gradients, weight saving, native gameplay or policy integration.
This reports feasibility, never random-network accuracy or planner speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.evaluate import load_rows
from nav.marble_dynamics.physics_student import MotionStudent
from nav.marble_dynamics.student_data import geometry,state_features,motion,MOTION_SCALES
from nav.marble_dynamics.record import ML,sha


def weight_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args()
    if a.batch_size<1 or a.repeats<1:raise ValueError('Positive bounded counts required')
    if a.output.exists():raise ValueError('Keep prior evidence; choose a new report')
    manifest=json.loads((a.data/'manifest.json').read_text())
    if not manifest['preparation_complete']:raise ValueError('Incomplete preparation')
    for name,digest in manifest['source_hashes'].items():
        if sha(Path(__file__).with_name(name+'.py'))!=digest:
            raise ValueError('Preparation source changed; use archived source or prepare a new bundle: '+name)
    if sha(ML/'nav/collision/mesh.py')!=manifest['geometry_reader_sha256']:
        raise ValueError('Geometry reader changed after preparation')
    for file,digest in manifest['output_hashes'].items():
        if sha(a.data/file)!=digest:raise ValueError('Prepared array changed: '+file)
    for source in manifest['sources']:
        if sha(source['path'])!=source['sha256'] or sha(source['manifest'])!=source['manifest_sha256']:
            raise ValueError('Native source changed after preparation')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(121012);model=MotionStudent().to(device).eval()
    digest=weight_digest(model);results=[]
    protected={str(path):sha(path) for path in [ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']}
    for record in manifest['map_records']:
        name=record['map'];mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
        surface=np.load(a.data/(name+'.surfaces.npz'),allow_pickle=False)
        steps=np.load(a.data/(name+'.steps.npz'),allow_pickle=False)
        exact=geometry(mesh)
        for key in exact:
            if not np.array_equal(surface[key],exact[key]):raise ValueError('Surface input changed: '+name+'/'+key)
        if not len(surface['triangle_ids'])==record['surfaces']==len(mesh.triangles):
            raise ValueError('Missing collision surface')
        # Independently recover labels from raw native BEFORE/AFTER states.
        checks=min(24,len(steps['x']));raw_cache={}
        for index in np.linspace(0,len(steps['x'])-1,checks,dtype=int):
            source_id=int(steps['source_id'][index]);source=manifest['sources'][source_id]
            if source['map']!=name:raise ValueError('Mixed map provenance')
            if source_id not in raw_cache:raw_cache[source_id]=load_rows(Path(source['path']))
            row=raw_cache[source_id][int(steps['native_frame'][index])]
            x,canonical=state_features(row['before'],dict(move=row['move'],events=row['control_events']),
                row['dt_ms'],surface['center'],float(surface['scale']))
            y=((motion(row['after'])-motion(canonical))/MOTION_SCALES).astype(np.float32)
            if not np.array_equal(x,steps['x'][index]) or not np.array_equal(y,steps['y'][index]):
                raise ValueError('Input/label provenance mismatch')
        batch=min(a.batch_size,len(steps['x']))
        indexes=np.linspace(0,len(steps['x'])-1,batch,dtype=int)
        inputs=[torch.as_tensor(value,dtype=torch.float32,device=device) for value in
            [steps['x'][indexes],surface['features'],surface['bbox_lo'],surface['bbox_hi']]]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        times=[]
        with torch.inference_mode():
            for i in range(a.repeats+2):
                start=time.perf_counter();pred=model(*inputs)
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                if i>=2:times.append(elapsed)
                if pred.shape!=(batch,9) or not bool(torch.all(torch.isfinite(pred))):
                    raise ValueError('Invalid unfitted motion output')
        results.append(dict(map=name,role=record['role'],surfaces=record['surfaces'],steps=record['steps'],
            batch=batch,raw_reconstruction_checks=checks,all_surfaces_exact_preparation=True,
            median_forward_s=float(np.median(times)),max_forward_s=max(times),
            peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None))
        print(json.dumps(results[-1]),flush=True)
        del inputs,pred;surface.close();steps.close()
    if digest!=weight_digest(model):raise ValueError('Unexpected model-weight change')
    if any(sha(path)!=before for path,before in protected.items()):raise ValueError('Protected artifact changed')
    report=dict(schema='unfitted_motion_feasibility_v1',device=str(device),
        device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',torch_version=torch.__version__,
        parameter_count=sum(p.numel() for p in model.parameters()),seed=121012,random_weights_sha256=digest,
        weights_unchanged=True,training_started=False,backward_calls=0,optimizer_steps=0,
        preparation_manifest=str(a.data/'manifest.json'),manifest_sha256=sha(a.data/'manifest.json'),
        source_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in
            ('student_dryrun','student_data','physics_student')},protected_artifacts=protected,maps=results,
        warning='Forward feasibility only. No accuracy claim, autoregressive rollout, planner speedup, model promotion or saved weights. Native one-step BEFORE state includes camera/effect/contact metadata whose multi-step evolution is not learned by this motion-only student.')
    atomic_json(a.output,report);print(a.output,flush=True)


if __name__=='__main__':main()
