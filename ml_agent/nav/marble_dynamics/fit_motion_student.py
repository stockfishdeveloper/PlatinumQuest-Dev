"""Prepared bounded motion benchmark; fitting requires the explicit --train flag.

The default only writes an inspectable protocol. No native games, A2 modification,
PPO, autoplay, checkpoint promotion or long-rollout accuracy claim.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.physics_student import MotionStudent
from nav.marble_dynamics.student_data import STATE_FEATURES,SURFACE_FEATURES,MOTION_SCALES
from nav.marble_dynamics.record import ML,sha
from nav.source_archive import snapshot_sources


class KinematicResidualStudent(MotionStudent):
    """Learn corrections to v*dt; velocity/spin changes still come from data.

    No power/trick recipes. Complete geometry/features are unchanged. This has
    different forward semantics from MotionStudent even with identical keys.
    """
    def __init__(self):
        super().__init__()
        torch.nn.init.zeros_(self.motion[-1].weight)
        torch.nn.init.zeros_(self.motion[-1].bias)

    def forward(self,states,surfaces,bbox_lo,bbox_hi):
        residual=super().forward(states,surfaces,bbox_lo,bbox_hi)
        baseline=torch.zeros_like(residual)
        baseline[:,:3]=states[:,3:6]*25.*states[:,STATE_FEATURES.index('dt_s'):STATE_FEATURES.index('dt_s')+1]
        return baseline+residual


class RelativeGeometryResidualStudent(KinematicResidualStudent):
    """Encode every face relative to the marble, without a distance cutoff.

    Continuous atan coordinates preserve near-contact scale while keeping distant
    geometry finite. Native normals/materials/boundaries remain present. No
    measured within-step contact, map ID, reward or future state is supplied.
    """
    def relative_features(self,states,surfaces):
        p=states[:,:3];radius=states[:,STATE_FEATURES.index('radius_map')]
        if bool(torch.any(radius<=0)):raise ValueError('Positive marble radius required')
        relative=surfaces[None].expand(len(states),-1,-1).clone()
        for start in (0,3,6,9,12,15,SURFACE_FEATURES.index('center_x')):
            relative[:,:,start:start+3]=torch.atan((relative[:,:,start:start+3]-p[:,None])/radius[:,None,None])
        ix=SURFACE_FEATURES.index('native_plane_d')
        signed=surfaces[None,:,ix]+torch.sum(surfaces[None,:,ix-3:ix]*p[:,None],dim=-1)
        relative[:,:,ix]=torch.atan(signed/radius[:,None])
        ix=SURFACE_FEATURES.index('extent_x')
        relative[:,:,ix:ix+3]=torch.atan(relative[:,:,ix:ix+3]/radius[:,None,None])
        ix=SURFACE_FEATURES.index('area')
        relative[:,:,ix]=torch.atan(torch.sqrt(torch.clamp(relative[:,:,ix],min=0.))/radius[:,None])
        return relative

    def forward(self,states,surfaces,bbox_lo,bbox_hi):
        if not len(surfaces) or len(surfaces)!=len(bbox_lo) or len(surfaces)!=len(bbox_hi):
            raise ValueError('Complete nonempty surface/bounds alignment required')
        relative=self.relative_features(states,surfaces)
        p=states[:,:3];radius=states[:,STATE_FEATURES.index('radius_map')]
        # Global origin is irrelevant to this physical transition; its position
        # still determines all relative geometry and the known drift below.
        canonical=states.clone();canonical[:,:3]=0.
        state=self.state(canonical);encoded=self.surface(relative)
        logits=torch.sum(self.query(state)[:,None]*self.key(encoded),dim=-1)/math.sqrt(self.width)
        scale=states[:,STATE_FEATURES.index('map_scale')]*100.
        dt=states[:,STATE_FEATURES.index('dt_s')]
        reach=radius+(torch.linalg.vector_norm(states[:,3:6],dim=1)*25.+45.)*dt/scale+.05/scale
        distance=torch.maximum(torch.maximum(bbox_lo[None]-p[:,None],p[:,None]-bbox_hi[None]),
            torch.zeros((),device=states.device,dtype=states.dtype))
        prior=-torch.clamp(torch.sum(distance**2,dim=-1)/reach[:,None]**2,max=40.)
        attention=torch.softmax(logits+prior,dim=-1)
        context=torch.sum(attention[:,:,None]*self.value(encoded),dim=1)
        residual=self.motion(torch.cat([state,context],dim=1))
        baseline=torch.zeros_like(residual);baseline[:,:3]=states[:,3:6]*25.*dt[:,None]
        return baseline+residual


def closest_triangle_context(points, triangles):
    """Exact face/segment/corner geometry, including zero-area fan triangles.

    Returns all B-by-face closest points and all three segment projections.
    No nearest-K/range filter or future-contact annotation is involved.
    """
    a,b,c=triangles.unbind(dim=1);ab=b-a;ac=c-a
    normal=torch.cross(ab,ac,dim=-1);nn=torch.sum(normal*normal,dim=-1)
    denominator=torch.where(nn>0,nn,torch.ones_like(nn))
    height=torch.sum((points[:,None]-a[None])*normal[None],dim=-1)/denominator[None]
    projected=points[:,None]-normal[None]*height[:,:,None]
    rel=projected-a[None]
    u=torch.sum(torch.cross(rel,ac[None].expand_as(rel),dim=-1)*normal[None],dim=-1)/denominator[None]
    v=torch.sum(torch.cross(ab[None].expand_as(rel),rel,dim=-1)*normal[None],dim=-1)/denominator[None]
    inside=(nn[None]>0)&(u>=0)&(v>=0)&(u+v<=1)
    edges=torch.roll(triangles,-1,dims=1)-triangles
    ee=torch.sum(edges*edges,dim=-1)
    ee=torch.where(ee>0,ee,torch.ones_like(ee))
    t=torch.sum((points[:,None,None]-triangles[None])*edges[None],dim=-1)/ee[None]
    segments=triangles[None]+edges[None]*torch.clamp(t,0.,1.)[:,:,:,None]
    distances=torch.sum((segments-points[:,None,None])**2,dim=-1)
    index=torch.argmin(distances,dim=-1)
    nearest=torch.gather(segments,2,index[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
    return torch.where(inside[:,:,None],projected,nearest),segments


def contact_geometry_features(states,surfaces):
    """Nine continuous before-state features, retaining every operative face."""
    points=states[:,:3];radius=states[:,STATE_FEATURES.index('radius_map')]
    if bool(torch.any(radius<=0)):raise ValueError('Positive marble radius required')
    closest,_=closest_triangle_context(points,surfaces[:,:9].reshape(-1,3,3))
    _,segments=closest_triangle_context(points,surfaces[:,9:18].reshape(-1,3,3))
    flags=surfaces[:,-3:]>0
    available=flags.any(dim=-1)
    distances=torch.sum((segments-points[:,None,None])**2,dim=-1)
    index=torch.argmin(torch.where(flags[None],distances,torch.full_like(distances,float('inf'))),dim=-1)
    edge=torch.gather(segments,2,index[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
    edge_relative=torch.where(available[None,:,None],edge-points[:,None],torch.zeros_like(edge))
    relative=closest-points[:,None];distance=torch.linalg.vector_norm(relative,dim=-1)
    extra=torch.cat([torch.atan(relative/radius[:,None,None]),torch.atan(distance/radius[:,None])[:,:,None],
        torch.atan(edge_relative/radius[:,None,None]),
        torch.atan(torch.linalg.vector_norm(edge_relative,dim=-1)/radius[:,None])[:,:,None],
        available[None,:,None].expand(len(states),-1,-1).to(states.dtype)],dim=-1)
    return extra,distance


class ContactContextResidualStudent(RelativeGeometryResidualStudent):
    """Four independent contact summaries over all complete-map faces."""
    def __init__(self):
        MotionStudent.__init__(self,surface_features=len(SURFACE_FEATURES)+9,width=64)
        self.heads=4;self.head_width=32
        self.query=torch.nn.Linear(128,self.heads*self.head_width)
        self.key=torch.nn.Linear(64,self.heads*self.head_width)
        self.value=torch.nn.Linear(64,self.heads*self.head_width)
        self.motion=torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.SiLU(),torch.nn.Linear(128,9))
        torch.nn.init.zeros_(self.motion[-1].weight);torch.nn.init.zeros_(self.motion[-1].bias)

    def forward(self,states,surfaces,bbox_lo,bbox_hi):
        if not len(surfaces) or len(surfaces)!=len(bbox_lo) or len(surfaces)!=len(bbox_hi):
            raise ValueError('Complete nonempty surface/bounds alignment required')
        relative=self.relative_features(states,surfaces);extra,distance=contact_geometry_features(states,surfaces)
        canonical=states.clone();canonical[:,:3]=0.
        state=self.state(canonical);encoded=self.surface(torch.cat([relative,extra],dim=-1))
        query=self.query(state).reshape(len(states),self.heads,self.head_width)
        key=self.key(encoded).reshape(len(states),len(surfaces),self.heads,self.head_width)
        value=self.value(encoded).reshape(len(states),len(surfaces),self.heads,self.head_width)
        logits=torch.sum(query[:,None]*key,dim=-1)/math.sqrt(self.head_width)
        scale=states[:,STATE_FEATURES.index('map_scale')]*100.
        dt=states[:,STATE_FEATURES.index('dt_s')];radius=states[:,STATE_FEATURES.index('radius_map')]
        reach=radius+(torch.linalg.vector_norm(states[:,3:6],dim=1)*25.+45.)*dt/scale+.05/scale
        prior=-torch.clamp(distance**2/reach[:,None]**2,max=40.)
        attention=torch.softmax(logits+prior[:,:,None],dim=1)
        context=torch.sum(attention[:,:,:,None]*value,dim=1).reshape(len(states),-1)
        residual=self.motion(torch.cat([state,context],dim=1))
        baseline=torch.zeros_like(residual);baseline[:,:3]=states[:,3:6]*25.*dt[:,None]
        return baseline+residual


def ordinary_motion_eligible(development):
    """Every component on each dev map must preserve the declared baseline."""
    for record in development['maps']:
        ordinary=record['by_stratum'].get('0')
        if ordinary is None:return False
        if np.any(np.asarray(ordinary['mean_error'])>1.05*np.asarray(ordinary['constant_velocity_mean_error'])+1e-6):return False
    return bool(development['maps'])


def load_data(root,manifest):
    data={}
    for record in manifest['map_records']:
        name=record['map']
        with np.load(root/(name+'.steps.npz'),allow_pickle=False) as z:
            steps={k:z[k].copy() for k in z.files}
        with np.load(root/(name+'.surfaces.npz'),allow_pickle=False) as z:
            surfaces={k:z[k].copy() for k in ('features','bbox_lo','bbox_hi')}
        data[name]=dict(record=record,steps=steps,surfaces=surfaces,
            by_stratum={i:np.flatnonzero(steps['stratum']==i) for i in range(3)})
    return data


def evaluate(model,data,role,device,batch_size):
    """Native BEFORE one-step evaluation, explicitly not autoregressive rollout."""
    results=[];model.eval()
    with torch.inference_mode():
        for name,item in data.items():
            if item['record']['role']!=role:continue
            z=item['steps'];surface=[torch.as_tensor(item['surfaces'][k],device=device) for k in
                ('features','bbox_lo','bbox_hi')];predictions=[]
            for first in range(0,len(z['x']),batch_size):
                x=torch.as_tensor(z['x'][first:first+batch_size],device=device)
                predictions.append(model(x,*surface).cpu().numpy())
            pred=np.concatenate(predictions);delta=(pred-z['y'])*MOTION_SCALES
            # A disclosed constant-velocity baseline: dt16ms, unchanged velocity/spin.
            baseline=np.zeros_like(z['y']);baseline[:,:3]=z['x'][:,3:6]*25.*z['x'][:,STATE_FEATURES.index('dt_s'):STATE_FEATURES.index('dt_s')+1]
            baseline_delta=(baseline-z['y'])*MOTION_SCALES
            norms=lambda a:np.column_stack([np.linalg.norm(a[:,i:i+3],axis=1) for i in (0,3,6)])
            error=norms(delta);base_error=norms(baseline_delta);classes={}
            for s in range(3):
                keep=z['stratum']==s
                if np.any(keep):classes[str(s)]=dict(count=int(keep.sum()),mean_error=error[keep].mean(axis=0).tolist(),
                    p95_error=np.percentile(error[keep],95,axis=0).tolist(),max_error=error[keep].max(axis=0).tolist(),
                    constant_velocity_mean_error=base_error[keep].mean(axis=0).tolist(),
                    normalized_motion_mse=float(np.mean((pred[keep]-z['y'][keep])**2)))
            powers={}
            for power in range(1,6):
                keep=(z['stratum']==2)&(z['x'][:,STATE_FEATURES.index('held_'+str(power))]==1)
                if np.any(keep):powers[str(power)]=dict(count=int(keep.sum()),
                    mean_error=error[keep].mean(axis=0).tolist(),p95_error=np.percentile(error[keep],95,axis=0).tolist(),max_error=error[keep].max(axis=0).tolist())
            results.append(dict(map=name,count=len(error),mean_error=error.mean(axis=0).tolist(),
                p95_error=np.percentile(error,95,axis=0).tolist(),max_error=error.max(axis=0).tolist(),
                constant_velocity_mean_error=base_error.mean(axis=0).tolist(),by_stratum=classes,
                activation_by_held_power=powers,
                stratum_balanced_selection_mse=float(np.mean([r['normalized_motion_mse'] for r in classes.values()])),
                normalized_motion_mse=float(np.mean((pred-z['y'])**2))))
    model.train()
    return dict(role=role,maps=results,units=['position_u','velocity_u_per_s','spin_rad_per_s'],
        warning='Each sample uses its measured native BEFORE state. This is a teacher-forced one-step benchmark, not a whole-trajectory forecast or safety evaluation.')


def native_motion_loss(prediction,target,paired_prediction=None,paired_target=None):
    """Marginal motion plus optional measured action effect; no reward target."""
    loss=torch.mean((prediction-target)**2)
    if (paired_prediction is None)!=(paired_target is None):
        raise ValueError('Both paired predictions and targets are required')
    if paired_prediction is not None:
        if paired_prediction.ndim!=3 or paired_prediction.shape[1:]!=(2,9) or paired_prediction.shape!=paired_target.shape:
            raise ValueError('Paired native motion must have shape N,2,9')
        effect=paired_prediction[:,1]-paired_prediction[:,0]
        measured=paired_target[:,1]-paired_target[:,0]
        loss=loss+torch.mean((effect-measured)**2)
    return loss


def load_action_pairs(root,data):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!='native_action_difference_v1' or not manifest['preparation_complete']:
        raise ValueError('Incomplete or unsupported action-pair preparation')
    if manifest['state_features']!=STATE_FEATURES or manifest['motion_scales']!=MOTION_SCALES.tolist():
        raise ValueError('Paired feature/label schema differs')
    for path,digest in manifest['sources'].items():
        if sha(path)!=digest:raise ValueError('Paired native source changed')
    for file,digest in manifest['output_hashes'].items():
        if sha(root/file)!=digest:raise ValueError('Paired arrays changed')
    pairs={}
    for record in manifest['map_records']:
        name=record['map']
        if record['role']!='train' or name not in data or data[name]['record']['role']!='train':
            raise ValueError('Action pairs must be from the existing training maps only')
        if record['geometry_file_sha256']!=data[name]['record']['geometry_file_sha256']:
            raise ValueError('Paired source geometry differs')
        with np.load(root/(name+'.pairs.npz'),allow_pickle=False) as z:
            x,y=z['x'].copy(),z['y'].copy()
        if x.shape!=(record['pairs'],2,len(STATE_FEATURES)) or y.shape!=(record['pairs'],2,9) or not len(x):
            raise ValueError('Malformed paired motion tensors')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Nonfinite paired motion')
        pairs[name]=dict(x=x,y=y)
    if set(pairs)!={n for n,d in data.items() if d['record']['role']=='train'}:
        raise ValueError('Every declared training map needs action pairs')
    return pairs,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train',action='store_true',help='Run only after operator review; otherwise prepare protocol only')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--max-wall-s',type=float,default=300.);p.add_argument('--seed',type=int,default=121013)
    p.add_argument('--kinematic-residual',action='store_true',help='Separate variant: learn corrections to known v*dt position drift')
    p.add_argument('--relative-geometry',action='store_true',help='Separate residual variant: encode every face relative to current marble at contact scale')
    p.add_argument('--contact-context',action='store_true',help='Development-only four-head exact face/real-edge context; ordinary-motion eligibility gate')
    p.add_argument('--dev-only',action='store_true',help='Do not reopen the already inspected test maps during development')
    p.add_argument('--paired-data',type=Path,help='Optional verified train-only native action pairs; four pairs/update, auxiliary weight1')
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1 or a.max_wall_s<=0:raise ValueError('Positive bounded limits required')
    if sum((a.kinematic_residual,a.relative_geometry,a.contact_context))>1:raise ValueError('Choose one residual architecture')
    if a.contact_context and not a.dev_only:raise ValueError('Contact-context development protocol keeps neural test maps closed')
    if a.paired_data and not (a.contact_context and a.dev_only):raise ValueError('Paired-loss experiment requires contact-context/dev-only')
    if a.output.exists():raise ValueError('Preserve previous run/protocol; choose a fresh output')
    manifest=json.loads((a.data/'manifest.json').read_text())
    if not manifest['preparation_complete']:raise ValueError('Incomplete data preparation')
    if manifest['state_features']!=STATE_FEATURES or manifest['surface_features']!=SURFACE_FEATURES:
        raise ValueError('Feature-schema mismatch')
    for name,digest in manifest['source_hashes'].items():
        if sha(Path(__file__).with_name(name+'.py'))!=digest:raise ValueError('Prepared source changed: '+name)
    if sha(ML/'nav/collision/mesh.py')!=manifest['geometry_reader_sha256']:raise ValueError('Geometry reader changed')
    for file,digest in manifest['output_hashes'].items():
        if sha(a.data/file)!=digest:raise ValueError('Prepared arrays changed')
    for source in manifest['sources']:
        if sha(source['path'])!=source['sha256'] or sha(source['manifest'])!=source['manifest_sha256']:
            raise ValueError('Raw source changed after preparation')
    protected={str(path):sha(path) for path in [ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']}
    protocol=dict(schema='bounded_motion_fit_protocol_v1',data=str(a.data),
        manifest_sha256=sha(a.data/'manifest.json'),steps=a.steps,batch_size=a.batch_size,max_wall_s=a.max_wall_s,
        wall_budget_scope='Checked between training updates; scheduled/final validation finishes its bounded passes and may extend total wall time beyond this training-loop limit.',
        seed=a.seed,optimizer='AdamW',learning_rate=.0003,weight_decay=.0001,
        loss='Mean squared normalized native16ms motion delta, no reward',
        sampler='Uniform training map;60% ordinary,20% jump,20% held-power use, fallback only if a map lacks a stratum. Classes balance a supervised physics regression, never force a gameplay action.',
        validation='All development rows every200 updates and final; residual variant also checks update0. Map/stratum-balanced normalized dev MSE selects best checkpoint; per-power activation errors reported separately. Test maps evaluated once at the selected final checkpoint unless dev-only; never used for selection.',
        checkpoint_scope='New directory only, isolated83k-parameter random motion student; never A2 or a gameplay controller',
        fitting_started=bool(a.train),policy_training=False,promotion=False,
        architecture='contact_context_residual_motion_v1' if a.contact_context else 'relative_geometry_residual_motion_v1' if a.relative_geometry else 'kinematic_residual_motion_v1' if a.kinematic_residual else 'raw_delta_motion_v1',
        test_evaluation=not a.dev_only,
        autoregressive_validation=False,protected_artifacts=protected,
        source_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in
            ('fit_motion_student','physics_student','student_data')})
    if a.contact_context:
        protocol.update(sampler='Uniform train map;90%ordinary/5%jump/5%held-use; fallback only for absent strata',
            validation='Update0/every200/final; eligibility requires ordinary p/v/spin means on EACH development map <=1.05*constant velocity+1e-6. Lowest balanced dev MSE among eligible updates, with update0 baseline explicit. Test stays closed.',
            checkpoint_scope='New isolated contact-context student only; no policy modification',
            geometry_context='All38surface features plus9before-state operative-face/real-source-edge features;4heads of32dimensions. No contact/distance/top-k cutoff.')
    data=None;pairs=None;pair_manifest=None
    if a.paired_data:
        data=load_data(a.data,manifest);pairs,pair_manifest=load_action_pairs(a.paired_data,data)
        protocol.update(paired_data=str(a.paired_data),paired_manifest_sha256=sha(a.paired_data/'manifest.json'),
            paired_maps=pair_manifest['map_records'],paired_samples_per_update=4,paired_auxiliary_weight=1.,
            paired_sampler_seed=a.seed+1,marginal_sampler_sequence_preserved=True,
            loss='Marginal normalized-motion MSE plus MSE of paired predicted versus native motion-delta differences; no reward')
    a.output.mkdir(parents=True);atomic_json(a.output/'protocol.json',protocol)
    snapshot_sources(a.output,[Path(__file__).with_name(name+'.py')
        for name in protocol['source_hashes']]+[ML/'nav/source_archive.py'])
    if not a.train:
        print('PREPARATION ONLY: '+str(a.output/'protocol.json'),flush=True);return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed);rng=np.random.RandomState(a.seed)
    pair_rng=np.random.RandomState(a.seed+1) if pairs is not None else None
    model=(ContactContextResidualStudent() if a.contact_context else RelativeGeometryResidualStudent() if a.relative_geometry else KinematicResidualStudent() if a.kinematic_residual else MotionStudent()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    data=data if data is not None else load_data(a.data,manifest);names=[n for n,d in data.items() if d['record']['role']=='train']
    if not names:raise ValueError('No training maps')
    history=[];best_score=float('inf');best_update=0;completed=0;reason='step_limit';start=time.perf_counter()
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if a.kinematic_residual or a.relative_geometry or a.contact_context:
        initial_dev=evaluate(model,data,'dev',device,a.batch_size)
        best_score=float(np.mean([r['stratum_balanced_selection_mse'] for r in initial_dev['maps']]))
        history.append(dict(update=0,elapsed_wall_s=time.perf_counter()-start,dev=initial_dev,
            ordinary_motion_eligible=ordinary_motion_eligible(initial_dev) if a.contact_context else None))
    # Native gameplay never runs; states are individual archived regression inputs.
    for update in range(a.steps):
        if time.perf_counter()-start>=a.max_wall_s:reason='wall_limit';break
        name=names[rng.randint(len(names))];item=data[name];indexes=[]
        for s in rng.choice([0,1,2],size=a.batch_size,p=[.9,.05,.05] if a.contact_context else [.6,.2,.2]):
            available=item['by_stratum'][int(s)]
            indexes.append(int(rng.choice(available)) if len(available) else int(rng.randint(len(item['steps']['x']))))
        x=torch.as_tensor(item['steps']['x'][indexes],device=device)
        y=torch.as_tensor(item['steps']['y'][indexes],device=device)
        surfaces=[torch.as_tensor(item['surfaces'][k],device=device) for k in ('features','bbox_lo','bbox_hi')]
        optimizer.zero_grad(set_to_none=True);pred=model(x,*surfaces)
        paired_prediction=paired_target=None
        if pairs is not None:
            pair=pairs[name];selected=pair_rng.randint(len(pair['x']),size=4)
            pair_x=torch.as_tensor(pair['x'][selected].reshape(-1,len(STATE_FEATURES)),device=device)
            paired_target=torch.as_tensor(pair['y'][selected],device=device)
            paired_prediction=model(pair_x,*surfaces).reshape(4,2,9)
        loss=native_motion_loss(pred,y,paired_prediction,paired_target)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite fitting loss; no promotion')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();completed=update+1
        if completed%200==0 or completed==a.steps:
            dev=evaluate(model,data,'dev',device,a.batch_size)
            score=float(np.mean([r['stratum_balanced_selection_mse'] for r in dev['maps']]))
            eligible=ordinary_motion_eligible(dev) if a.contact_context else True
            history.append(dict(update=completed,elapsed_wall_s=time.perf_counter()-start,train_loss=float(loss),dev=dev,ordinary_motion_eligible=eligible))
            if eligible and score<best_score:best_score=score;best_update=completed;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_json(a.output/'history.json',history)
    # If the wall bound interrupts before a scheduled dev check, evaluate the actual final state.
    if not history or history[-1]['update']!=completed:
        dev=evaluate(model,data,'dev',device,a.batch_size)
        score=float(np.mean([r['stratum_balanced_selection_mse'] for r in dev['maps']]))
        eligible=ordinary_motion_eligible(dev) if a.contact_context else True
        history.append(dict(update=completed,elapsed_wall_s=time.perf_counter()-start,dev=dev,ordinary_motion_eligible=eligible))
        if eligible and score<best_score:best_score=score;best_update=completed;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    selected_dev=evaluate(model,data,'dev',device,a.batch_size)
    test=dict(skipped=True,reason='Development-only variant; do not reopen inspected test maps') if a.dev_only else evaluate(model,data,'test',device,a.batch_size)
    if any(sha(path)!=digest for path,digest in protected.items()):raise ValueError('Protected artifact changed')
    if pair_manifest and (sha(a.paired_data/'manifest.json')!=protocol['paired_manifest_sha256'] or
        any(sha(path)!=digest for path,digest in pair_manifest['sources'].items()) or
        any(sha(a.paired_data/file)!=digest for file,digest in pair_manifest['output_hashes'].items())):
        raise ValueError('Paired supervision changed during fit')
    # NEW student artifact only; optimizer is not resumed and A2 is never opened for writing.
    torch.save(dict(model=best,protocol=protocol,completed_updates=completed,best_update=best_update,best_dev_score=best_score),a.output/'motion_student.pth')
    atomic_json(a.output/'history.json',history)
    atomic_json(a.output/'result.json',dict(completed_updates=completed,stop_reason=reason,
        elapsed_wall_s=time.perf_counter()-start,best_dev_score=best_score,test=test,
        best_update=best_update,
        selection_status='baseline_only_no_eligible_improvement' if a.contact_context and best_update==0 else 'dev_selected',
        selected_dev=selected_dev,
        checkpoint_sha256=sha(a.output/'motion_student.pth'),protected_artifacts_unchanged=True,
        policy_training=False,promotion=False,autoregressive_validation=False))
    print(a.output/'result.json',flush=True)


if __name__=='__main__':main()
