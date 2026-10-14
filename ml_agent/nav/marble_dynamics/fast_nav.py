"""Train and run a cached complete-map imitation navigator, separate from A2.

Current observations/destinations are inputs. Recorded future outcomes are
supervision only. Every static triangle participates in four spatial queries.
The native evaluator, not imitation loss, establishes gameplay performance.
"""
import argparse
import copy
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nav.collision.mesh import CollisionMap, atomic_json
from nav.marble_dynamics.record import ML, ROOT, sha
from nav.marble_dynamics.student_data import geometry, PARAMETERS, PARAMETER_SCALES
from nav.source_archive import snapshot_sources

SCHEMA = 'fast_complete_map_navigator_v1'
GOAL_KINDS = ('gem', 'powerup', 'geometric_landing', 'physical_recovery')


def features(dynamics, raw, destination, waypoint, center, scale, kind='gem', base_action=None):
    """Pure current-observation features: no future state, identity or score."""
    state = dynamics['native']['state']
    raw = np.asarray(raw, dtype=float)
    position = np.asarray(state['position'], float)
    destination = np.asarray(destination, float)
    waypoint = np.asarray(waypoint, float)
    values = list((position-center)/scale)
    values += list((destination-position)/scale) + list((waypoint-position)/scale)
    values += list(np.clip((destination-position)/20., -4, 4))
    values += list(np.clip((waypoint-position)/20., -4, 4))
    values += list(np.asarray(state['velocity'])/25.)
    values += list(np.asarray(state['spin'])/100.)
    values += [state['radius']/.18975, state['ground_ms']/1000.]
    values += list(state['gravity_direction']) + list(state['best_contact_normal'])
    values += list(state['camera_offset'])
    for key in ('yaw', 'next_yaw', 'pitch', 'next_pitch'):
        values += [math.sin(state[key]), math.cos(state[key])]
    held = int(raw[38])
    if not 0 <= held <= 6:
        raise ValueError('Unknown held power type')
    values += [float(held == i) for i in range(7)]
    values += list(raw[39:46]/np.asarray([1, 1, 1, 10, 10, 10, 10]))
    for active, left in state['effects']:
        values += [float(active), float(left)/10000. if active else 0.]
    values += list(np.asarray([state['parameters'][p] for p in PARAMETERS])/PARAMETER_SCALES)
    action = list(dynamics['bridge_action'])
    values += [(action[3]-action[2])/math.sqrt(2), (action[0]-action[1])/math.sqrt(2),
               action[4], math.sin(action[5]), math.cos(action[5]), action[6],
               math.sin(action[7]), math.cos(action[7]), action[8]]
    values += list(dynamics['input_move'])
    values += [dynamics['pow_yaw_hold']/12., math.sin(dynamics['pow_yaw']),
               math.cos(dynamics['pow_yaw']), scale/100.]
    values += [float(kind == k) for k in GOAL_KINDS]
    if base_action is not None:values += list(action_target(base_action))
    result = np.asarray(values, np.float32)
    queries = np.asarray([position, destination, waypoint,
                          position + np.asarray(state['velocity'])*.6], float)
    queries = ((queries-center)/scale).astype(np.float32)
    if result.ndim != 1 or not np.isfinite(result).all() or not np.isfinite(queries).all():
        raise ValueError('Invalid/nonfinite current navigator observation')
    return result, queries


def action_target(action):
    if len(action) != 9 or not np.isfinite(action).all():
        raise ValueError('Expected finite nine-field applied reply')
    return np.asarray([(action[3]-action[2])/math.sqrt(2),
                       (action[0]-action[1])/math.sqrt(2),
                       math.sin(action[5]), math.cos(action[5]),
                       math.sin(action[7]), math.cos(action[7]),
                       action[4], action[6], action[8]], np.float32)


def decode_action(output,base_action=None,axis_mode='tanh'):
    output = np.asarray(output, float)
    if output.shape != (9,) or not np.isfinite(output).all():
        raise ValueError('Invalid navigator output')
    if axis_mode not in ('tanh','linear'):raise ValueError('Unknown navigator axis mode')
    axes=np.clip(output[:2],-1,1) if axis_mode=='linear' else np.tanh(output[:2])
    mx, my = axes*math.sqrt(2)
    # Camera rotation and per-axis clamp are the game's existing reply path.
    yaw = math.atan2(output[2], output[3])
    aim = math.atan2(output[4], output[5])
    result=[round(max(my, 0.), 6), round(max(-my, 0.), 6),
            round(max(-mx, 0.), 6), round(max(mx, 0.), 6),
            int(output[6] > 0), yaw, int(output[7] > 0), aim, int(output[8] > 0)]
    if base_action is not None:
        for index in (5,7):
            delta=math.atan2(math.sin(result[index]-base_action[index]),math.cos(result[index]-base_action[index]))
            result[index]=base_action[index]+delta
        # Preserve serialized native controls for the exact zero residual,
        # including camera angles outside[-pi,pi] and Float32 inverse-tanh noise.
        for index in range(9):
            if abs(result[index]-base_action[index])<3e-6:result[index]=base_action[index]
    return result


def legal_action(action,dynamics,raw):
    """Current inventory and game Blast availability, with no maneuver prior."""
    result=list(action);held=int(raw[38])
    if not 0<=held<=6:raise ValueError('Unknown current inventory')
    if held==0:result[6]=0
    context=dynamics.get('blast')
    if context is None:raise ValueError('Observed Blast context required for legal decoding')
    # Fireball has a separate cooldown unavailable in this observation. Preserve
    # its command rather than pretending the ordinary meter is its availability.
    if not context.get('fireball',False):
        required=.33 if context.get('triple',False) else context['required']
        if not context['enabled'] or context['frozen'] or context['value']<required:result[8]=0
    return result


class FastMapPolicy(nn.Module):
    def __init__(self, state_size, surface_size, width=32, residual=False,axis_mode='tanh'):
        super().__init__()
        self.config = dict(state_size=state_size, surface_size=surface_size, width=width,residual=residual)
        if axis_mode not in ('tanh','linear'):raise ValueError('Unknown navigator axis mode')
        if axis_mode=='linear':self.config['axis_mode']=axis_mode
        self.axis_mode=axis_mode
        self.residual=residual
        self.width = width
        self.surface = nn.Sequential(nn.Linear(surface_size, 64), nn.SiLU(),
                                     nn.Linear(64, width), nn.SiLU())
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.state = nn.Sequential(nn.Linear(state_size, 128), nn.SiLU(),
                                   nn.Linear(128, 128), nn.SiLU())
        self.query = nn.Linear(128, 4*width)
        self.head = nn.Sequential(nn.Linear(128+4*width, 128), nn.SiLU(), nn.Linear(128, 9))
        with torch.no_grad():
            if residual:
                self.head[-1].weight.zero_();self.head[-1].bias.zero_()
            else:self.head[-1].bias[6:] = -2.

    def encode_map(self, surfaces, lo, hi):
        if len(surfaces) != len(lo) or len(lo) != len(hi) or not len(lo):
            raise ValueError('A complete aligned nonempty surface set is required')
        encoded = self.surface(surfaces)
        return self.key(encoded), self.value(encoded), lo, hi

    def forward_cached(self, x, spatial_queries, encoded):
        keys, values, lo, hi = encoded
        state = self.state(x)
        queries = self.query(state).reshape(-1, 4, self.width)
        logits = torch.matmul(queries, keys.T)/math.sqrt(self.width)
        # No hard cutoff. Body and distant goal/waypoint query different places.
        delta = torch.maximum(torch.maximum(lo[None, None]-spatial_queries[:, :, None],
                              spatial_queries[:, :, None]-hi[None, None]),
                              torch.zeros((), device=x.device))
        distance2 = torch.sum(delta*delta, dim=-1)
        prior = -torch.clamp(distance2/.025**2, max=16.)
        attention = torch.softmax(logits+prior, dim=-1)
        context = torch.matmul(attention, values).reshape(len(x), -1)
        correction=self.head(torch.cat([state, context], dim=-1))
        if not self.residual:return correction
        base=x[:,-9:]
        axes=base[:,:2]+.5*correction[:,:2] if self.axis_mode=='linear' else torch.atanh(base[:,:2].clamp(-.999999,.999999))+.5*correction[:,:2]
        angles=base[:,2:6]+.25*correction[:,2:6]
        buttons=torch.where(base[:,6:]>.5,2.,-2.)+correction[:,6:]
        return torch.cat([axes,angles,buttons],dim=1)

    def forward(self, x, spatial_queries, surfaces, lo, hi):
        return self.forward_cached(x, spatial_queries, self.encode_map(surfaces, lo, hi))


def button_eligibility(x, target):
    """Supervision mask; not an inference exploration rule or reward change."""
    if x.shape[1] not in (100,109):raise ValueError('Unknown navigation feature layout')
    mask=torch.ones_like(target[:,6:])
    mask[:,1]=(x[:,41:47].sum(dim=1)>.5).float()
    mask[:,2]=((x[:,47]>=.2)|(x[:,48]>.5)).float()
    return torch.maximum(mask,(target[:,6:]>.5).float())


def objective(prediction, target, weight, positive_weight=None, eligibility=None,axis_mode='tanh'):
    predicted_axes=prediction[:,:2] if axis_mode=='linear' else torch.tanh(prediction[:,:2])
    axes = ((predicted_axes-target[:, :2])**2).mean(dim=1)
    angles = ((prediction[:, 2:6]-target[:, 2:6])**2).mean(dim=1)
    buttons = F.binary_cross_entropy_with_logits(prediction[:, 6:], target[:, 6:],
                                                pos_weight=positive_weight,reduction='none')
    if eligibility is not None:buttons=buttons*eligibility
    buttons=buttons.mean(dim=1)
    return ((axes + .15*angles + .4*buttons)*weight).sum()/weight.sum()


def base_preservation_objective(prediction, current_base, axis_mode='linear'):
    """Preserve the existing action function, without driving logits to infinity."""
    axes=prediction[:,:2] if axis_mode=='linear' else torch.tanh(prediction[:,:2])
    reference_buttons=torch.where(current_base[:,6:]>.5,2.,-2.)
    return ((axes-current_base[:,:2])**2).mean()+.15*((prediction[:,2:6]-current_base[:,2:6])**2).mean()+.4*((prediction[:,6:]-reference_buttons)**2).mean()


def preservation_fit_objective(prediction,x,target,weight,original_count,axis_mode):
    """Original train rows anchor A2; only checked advice supplies corrections."""
    if not 0<original_count<=len(prediction):raise ValueError('Nonempty original anchor rows required')
    fraction=original_count/len(prediction)
    loss=fraction*base_preservation_objective(prediction[:original_count],x[:original_count,-9:],axis_mode)
    if original_count<len(prediction):
        loss+=(1-fraction)*objective(prediction[original_count:],target[original_count:],weight[original_count:],axis_mode=axis_mode)
    return loss


def configure_activation_only(model):
    """Train existing activation rows; preserve the learned movement function."""
    for parameter in model.parameters():parameter.requires_grad_(False)
    last=model.head[-1]
    if not isinstance(last,nn.Linear) or last.out_features!=9:
        raise ValueError('Expected independent nine-field navigation output')
    def activation_gradient(gradient):
        result=gradient.clone();result[:7]=0
        return result
    for parameter in (last.weight,last.bias):
        parameter.requires_grad_(True)
        parameter.register_hook(activation_gradient)


def activation_objective(prediction,target,x,weight):
    eligible=button_eligibility(x,target)[:,1:]
    weighted=eligible*weight[:,None]
    error=F.binary_cross_entropy_with_logits(prediction[:,7:],target[:,7:],reduction='none')
    return (error*weighted).sum()/weighted.sum().clamp(min=1.)


def assert_movement_preserved(model,original):
    for key,value in model.state_dict().items():
        actual=value[:7] if key in ('head.2.weight','head.2.bias') else value
        expected=original[key][:7] if key in ('head.2.weight','head.2.bias') else original[key]
        if not torch.equal(actual,expected):
            raise ValueError('Activation-only fit changed movement parameter: '+key)
    return True


def protected_hashes():
    paths = [ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',
             ROOT/'Marble Blast Platinum/marbleblast_geometry_probe.exe',
             ROOT/'Marble Blast Platinum/marbleblast_mbx.exe',
             ROOT/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs']
    return {str(path): sha(path) for path in paths}


def prepare(teacher, output, residual=False):
    output.mkdir(parents=True, exist_ok=False)
    maps = json.loads((ML/'configs/real_maps_20261009.json').read_text())['maps']
    sources = {str(teacher/'prepared.json'): sha(teacher/'prepared.json')}
    records = {}; counts = {}; dimension = None
    for map_index, name in enumerate(maps):
        journals = list((teacher/name).glob('*/physics_plan/*/decisions.jsonl.gz'))
        if len(journals) != 1:
            raise ValueError('Exactly one declared teacher round/map required: '+name)
        path = journals[0]; report = json.loads((path.parent/'report.json').read_text())
        if not report['completed'] or not report['score_valid'] or not report['declared_inputs_valid']:
            raise ValueError('Incomplete/input-invalid teacher round')
        sources[str(path)] = sha(path)
        sources[str(path.parent/'report.json')] = sha(path.parent/'report.json')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        mesh = CollisionMap(ML/'collision_maps'/(name+'.npz')); geo = geometry(mesh)
        sources[str(ML/'collision_maps'/(name+'.npz'))] = sha(ML/'collision_maps'/(name+'.npz'))
        xs=[]; qs=[]; ys=[]; ws=[]; splits=[]; decisions=[]; exclusions=Counter()
        for index, row in enumerate(rows):
            goal = row['choice'].get('goal_context')
            state = row['before']['native']['state']
            if not goal or goal['kind'] not in GOAL_KINDS or state['mode'] or state['oob']:
                exclusions['missing_goal_or_nonplayable'] += 1; continue
            if residual and 'actor_action' not in row['choice']:
                exclusions['missing_current_base_action']+=1;continue
            # Disjoint contiguous blocks and a >=1s boundary gap; no row shuffle
            # split that would put neighboring movement into both roles.
            block, offset = divmod(index, 256)
            is_dev = (block+map_index)%5 == 0
            if offset < 16 or offset >= 192:
                exclusions['split_boundary_gap'] += 1; continue
            future = rows[index:min(index+63, len(rows))]
            pickup = sum(max(0., r['outcome']['gem_delta']) for r in future)
            immediate_fall = any(r['outcome']['fell'] for r in future[:8])
            weight = .15 if immediate_fall else .35+min(pickup, 5.)*.3
            x, q = features(row['before'], row['raw_observation'], goal['destination'],
                            goal['waypoint'], geo['center'], geo['scale'], goal['kind'],
                            row['choice']['actor_action'] if residual else None)
            dimension = len(x) if dimension is None else dimension
            if len(x) != dimension:raise ValueError('Feature shape changed across maps')
            xs.append(x);qs.append(q);ys.append(action_target(row['action']));ws.append(weight)
            splits.append(int(is_dev));decisions.append(row['decision'])
        if not xs or not any(splits) or all(splits):raise ValueError('Empty train/dev map role')
        np.savez_compressed(output/(name+'.npz'), x=np.asarray(xs), q=np.asarray(qs),
                            y=np.asarray(ys), weight=np.asarray(ws, np.float32),
                            dev=np.asarray(splits, bool), decision=np.asarray(decisions),
                            **geo)
        records[name] = dict(examples=len(xs), train=sum(not s for s in splits), dev=sum(splits),
                             triangles=len(geo['features']), exclusions=dict(exclusions),
                             teacher_points=report['actual_points'], journal=str(path),
                             tensor_sha256=sha(output/(name+'.npz')))
        counts[name] = [records[name]['train'], records[name]['dev']]
        print(json.dumps(dict(prepared=name, counts=counts[name], triangles=len(geo['features']))), flush=True)
    snapshot_sources(output, [Path(__file__), ML/'nav/marble_dynamics/student_data.py',
                              ML/'nav/collision/mesh.py', ML/'nav/source_archive.py'])
    manifest = dict(schema=SCHEMA, stage='navigation_imitation_preparation', maps=maps,
        teacher=str(teacher), seed=101313, state_size=dimension, records=records,residual=residual,
        source_hashes=sources, protected_hashes=protected_hashes(),
        split='Contiguous256-decision blocks; one in five held out/map;16-decision lower/64-decision upper gaps keep future outcome windows inside their block. Familiar map and one demonstrated seed, no unseen-map claim.',
        future_contract='Only next63 outcome gem deltas/next8 fall flags affect weights. No future physics/score/IDs in current inputs.',
        complete_geometry=True, fitting_started=False)
    atomic_json(output/'manifest.json', manifest)
    return manifest


def load_data(dataset, manifest):
    result = {}
    for name in manifest['maps']:
        path = dataset/(name+'.npz')
        if sha(path) != manifest['records'][name]['tensor_sha256']:
            raise ValueError('Navigator tensor changed: '+name)
        with np.load(path, allow_pickle=False) as archive:
            data = {key:archive[key].copy() for key in archive.files}
        result[name] = {key:torch.from_numpy(data[key]).float() for key in
                        ('x','q','y','weight','features','bbox_lo','bbox_hi')}
        result[name].update(train=np.flatnonzero(~data['dev']), dev=np.flatnonzero(data['dev']))
    return result


def validate(model, data, seed=101314, positive_weight=None, full_development=False):
    rng = np.random.RandomState(seed);metrics={}
    model.eval()
    with torch.inference_mode():
        for name, d in data.items():
            ids = d['dev'] if full_development else rng.choice(d['dev'], min(256, len(d['dev'])), replace=False)
            encoded=model.encode_map(d['features'],d['bbox_lo'],d['bbox_hi'])
            losses=[]; axes=[]; buttons=[];confusion=np.zeros((3,4),dtype=int)
            for start in range(0,len(ids),32):
                ix=ids[start:start+32];p=model.forward_cached(d['x'][ix],d['q'][ix],encoded)
                eligible=button_eligibility(d['x'][ix],d['y'][ix]) if positive_weight is not None else None
                losses.append(float(objective(p,d['y'][ix],d['weight'][ix],positive_weight,eligible,model.axis_mode)))
                predicted_axes=p[:,:2] if model.axis_mode=='linear' else torch.tanh(p[:,:2])
                axes.extend(((predicted_axes-d['y'][ix,:2])**2).mean(dim=1).tolist())
                buttons.extend(((p[:,6:]>0)!=(d['y'][ix,6:]>0.5)).float().tolist())
                actual=(d['y'][ix,6:]>.5).numpy();predicted=(p[:,6:]>0).numpy()
                confusion+=np.stack([(predicted&actual).sum(axis=0),(predicted&~actual).sum(axis=0),
                    (~predicted&actual).sum(axis=0),(~predicted&~actual).sum(axis=0)],axis=1)
            metrics[name]=dict(loss=float(np.mean(losses)),axis_mse=float(np.mean(axes)),
                               button_error=np.asarray(buttons).mean(axis=0).tolist(), examples=len(ids),
                               button_confusion_tp_fp_fn_tn=confusion.tolist())
    model.train()
    return dict(equal_map_loss=float(np.mean([v['loss'] for v in metrics.values()])),per_map=metrics)


def advice_evidence(folder,defer_replay=False):
    manifest_path=folder/'manifest.json'
    evidence=json.loads(manifest_path.read_text())
    path=folder/'independent_audit.json'
    if not path.exists() and defer_replay:path=folder/'collection_integrity.json'
    checked=json.loads(path.read_text())
    if evidence['schema']!='analytical_navigation_advice_v1' or not evidence['complete'] or not checked['passed']:
        raise ValueError('Complete checked advice required')
    if checked['manifest_sha256']!=sha(manifest_path):raise ValueError('Advice check does not match manifest')
    if path.name=='collection_integrity.json' and (not defer_replay or checked.get('independent_replay') is not False):
        raise ValueError('Pending replay must be explicitly authorized for this scaling fit')
    return checked,path


def collection_integrity(folder,manifest):
    atomic_json(folder/'collection_integrity.json',dict(passed=manifest['complete'],
        manifest_sha256=sha(folder/'manifest.json'),cases_sha256=manifest['cases_sha256'],
        independent_replay=False,replay_pending=True))


def advice_context(name,dll,cache_dir=None):
    from nav.terrain import TerrainGrid,TerrainMap
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),dll)
    if cache_dir:
        from nav.marble_dynamics.static_cache import cached_terrain,cached_walking,load_queries
        terrain=cached_terrain(TerrainMap.resolve(name),cache_dir);load_queries(mesh,dll,cache_dir)
        terrain=cached_walking(terrain,mesh,dll,cache_dir)
    else:terrain=walking_terrain(TerrainGrid(TerrainMap.resolve(name)),mesh)
    return mesh,terrain


def persist_advice_context(mesh,terrain,cache_dir):
    if cache_dir:
        from nav.marble_dynamics.static_cache import save_queries,save_clearance
        save_queries(mesh);save_clearance(terrain)


def load_advice(folder, manifest,defer_replay=False):
    evidence=json.loads((folder/'manifest.json').read_text())
    audit,_=advice_evidence(folder,defer_replay)
    result={}
    for name in manifest['maps']:
        path=folder/(name+'.npz')
        if sha(path)!=evidence['maps'][name]['tensor_sha256']:raise ValueError('Advice tensor changed')
        with np.load(path,allow_pickle=False) as z:
            result[name]={key:torch.from_numpy(z[key].copy()).float() for key in ('x','q','y')}
        d=result[name]
        if d['x'].shape!=(len(d['y']),manifest['state_size']) or d['q'].shape!=(len(d['y']),4,3) or d['y'].shape[1:]!=(9,):
            raise ValueError('Advice interface mismatch')
        if not all(torch.isfinite(t).all() for t in d.values()):raise ValueError('Nonfinite advice')
        d['weight']=torch.ones(len(d['y']))
    return result


def advice_indices(data,rng,count,action_balanced=False):
    """Reserve examples of validated uses, never impose a live activation floor."""
    chosen=[]
    if action_balanced:
        for button in (7,8):
            positives=np.flatnonzero(data['y'][:,button].numpy()>.5)
            n=min(2,count-len(chosen)) if len(positives) else 0
            if n:chosen.extend(rng.choice(positives,n,replace=len(positives)<n).tolist())
    rest=count-len(chosen)
    if rest:chosen.extend(rng.choice(len(data['y']),rest,replace=len(data['y'])<rest).tolist())
    rng.shuffle(chosen);return np.asarray(chosen,dtype=np.int64)


def fit(dataset, output, updates=1500, wall_s=900, advice=None, initial=None, balanced_buttons=False,linear_axes=False,advice_actions=False,activation_only=False,preserve_base=False,defer_replay=False):
    if activation_only and (initial is None or advice is None or balanced_buttons):
        raise ValueError('Activation-only fit requires an initial model and audited advice, without global positive weights')
    output.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest['schema']!=SCHEMA:raise ValueError('Unsupported navigator dataset')
    if preserve_base and (not manifest.get('residual') or advice is None or balanced_buttons or activation_only):
        raise ValueError('Base preservation requires residual data and audited causal corrections, without competing training scopes')
    torch.set_num_threads(4);torch.manual_seed(101313);rng=np.random.RandomState(101313)
    data=load_data(dataset,manifest);first=next(iter(data.values()))
    model=FastMapPolicy(manifest['state_size'],first['features'].shape[1],residual=manifest.get('residual',False),axis_mode='linear' if linear_axes else 'tanh')
    converted_axes=False;decoder={}
    if initial is not None:
        saved=torch.load(str(initial),map_location='cpu',weights_only=False)
        decoder=saved.get('decoder',{}).copy()
        converted_axes=linear_axes and model.residual and saved['config'].get('axis_mode','tanh')=='tanh'
        compatible={k:v for k,v in model.config.items() if k!='axis_mode'} if converted_axes else model.config
        if saved['schema']!=SCHEMA or saved['config']!=compatible:raise ValueError('Initial candidate interface mismatch')
        model.load_state_dict(saved['model'])
        if converted_axes:
            with torch.no_grad():model.head[-1].weight[:2].zero_();model.head[-1].bias[:2].zero_()
    advice_data=load_advice(advice,manifest,defer_replay) if advice is not None else None
    if preserve_base:
        if not json.loads((advice/'manifest.json').read_text()).get('first_reply_causal_corrections'):
            raise ValueError('Preservation fit requires matched first-reply advantage evidence')
        decoder['legal_actions']=True
    original_motion=None
    if activation_only:
        if converted_axes:raise ValueError('Activation-only fit cannot convert the movement decoder')
        original_motion={k:v.detach().clone() for k,v in model.state_dict().items()}
        configure_activation_only(model)
    positive_weight=None;button_counts=None
    if balanced_buttons:
        totals=torch.zeros(3);positives=torch.zeros(3)
        for name,d in data.items():
            roles=[(d,d['train'])]
            if advice_data is not None:roles.append((advice_data[name],np.arange(len(advice_data[name]['y']))))
            for role,ix in roles:
                totals+=button_eligibility(role['x'][ix],role['y'][ix]).sum(dim=0)
                positives+=(role['y'][ix,6:]>.5).float().sum(dim=0)
        positive_weight=((totals-positives)/positives.clamp(min=1)).clamp(min=1,max=32)
        positive_weight[0]=positive_weight[0].clamp(max=4)
        button_counts=dict(eligible=totals.tolist(),positive=positives.tolist(),positive_weight=positive_weight.tolist(),development_counts_used=False)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=3e-4)
    sources={**manifest['source_hashes'],**{str(p):sha(p) for p in [Path(__file__),dataset/'manifest.json']}}
    if initial is not None:sources[str(initial)]=sha(initial)
    if advice is not None:
        _,evidence_path=advice_evidence(advice,defer_replay)
        for p in [advice/'manifest.json',evidence_path]+[advice/(name+'.npz') for name in manifest['maps']]:sources[str(p)]=sha(p)
    snapshot_sources(output,[Path(__file__)])
    protected=protected_hashes()
    atomic_json(output/'protocol.json',dict(schema=SCHEMA,dataset=str(dataset),max_updates=updates,
        max_wall_s=wall_s,seed=101313,batch=48,balanced_map_sampling=True,
        sources=sources,protected=protected,policy_training='bounded_supervised_navigation',ppo=False,
        advice=str(advice) if advice else None,advice_replay_deferred=defer_replay,advice_batch_fraction=.25 if advice else 0,
        advice_action_sampling=advice_actions,
        training_scope='a2_preservation_and_causal_corrections' if preserve_base else 'existing_use_blast_rows_only' if activation_only else 'joint_navigation',
        original_training_targets='Current A2 axes/angles and finite button logits; original y/weight/dev tensors untouched' if preserve_base else 'Recorded selected teacher replies',
        initial=str(initial) if initial else None,button_counts=button_counts,
        axis_mode=model.axis_mode,converted_axis_head_reset=converted_axes,
        validation_contract='Original development segments untouched; current-state advice is train only; all are development experiments.'))
    start=time.perf_counter();history=[];best=float('inf');selected=0
    for update in range(updates+1):
        if update%100==0 or update==updates:
            if activation_only:assert_movement_preserved(model,original_motion)
            metrics=validate(model,data,positive_weight=positive_weight,full_development=advice_data is not None);metrics.update(update=update,wall_s=time.perf_counter()-start)
            history.append(metrics);atomic_json(output/'history.json',history)
            if metrics['equal_map_loss']<best:
                best=metrics['equal_map_loss'];selected=update
                torch.save(dict(schema=SCHEMA,config=model.config,model=model.state_dict(),update=update,
                    dataset_sha256=sha(dataset/'manifest.json'),metrics=metrics,decoder=decoder),output/'candidate.pth')
            print(json.dumps(dict(update=update,dev_loss=metrics['equal_map_loss'],selected=selected,
                                  wall_s=metrics['wall_s'])),flush=True)
        if update==updates or time.perf_counter()-start>=wall_s:break
        name=manifest['maps'][update%len(manifest['maps'])];d=data[name]
        extra=advice_data[name] if advice_data is not None else None
        count=12 if extra is not None and len(extra['y']) else 0
        ix=rng.choice(d['train'],48-count,replace=len(d['train'])<48-count)
        batch={key:d[key][ix] for key in ('x','q','y','weight')}
        if count:
            ax=advice_indices(extra,rng,count,advice_actions)
            batch={key:torch.cat([batch[key],extra[key][ax]]) for key in batch}
        pred=model(batch['x'],batch['q'],d['features'],d['bbox_lo'],d['bbox_hi'])
        eligible=button_eligibility(batch['x'],batch['y']) if balanced_buttons else None
        loss=(preservation_fit_objective(pred,batch['x'],batch['y'],batch['weight'],48-count,model.axis_mode) if preserve_base else
              activation_objective(pred,batch['y'],batch['x'],batch['weight']) if activation_only else
              objective(pred,batch['y'],batch['weight'],positive_weight,eligible,model.axis_mode))
        if not torch.isfinite(loss):raise ValueError('Nonfinite navigator loss')
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    final_metrics=history[-1]
    if final_metrics['update']!=update:
        final_metrics=validate(model,data,positive_weight=positive_weight,full_development=advice_data is not None)
        final_metrics.update(update=update,wall_s=time.perf_counter()-start)
        history.append(final_metrics);atomic_json(output/'history.json',history)
    # New-state advice can disagree with old demonstrations. Preserve the
    # bounded final model for native assessment as well as the Dev-best model.
    torch.save(dict(schema=SCHEMA,config=model.config,model=model.state_dict(),update=update,
        dataset_sha256=sha(dataset/'manifest.json'),metrics=final_metrics,
        selection='Final bounded fit; native improvement not established',decoder=decoder),output/'final.pth')
    integrity=all(sha(p)==v for p,v in sources.items()) and protected_hashes()==protected
    result=dict(schema=SCHEMA,complete=True,updates=update,selected_update=selected,
        initial_dev_loss=history[0]['equal_map_loss'],best_dev_loss=best,
        elapsed_wall_s=time.perf_counter()-start,source_and_protected_unchanged=integrity,
        checkpoint_sha256=sha(output/'candidate.pth'),ppo=False,gameplay_improvement_established=False)
    result.update(final_checkpoint_sha256=sha(output/'final.pth'),final_dev_loss=final_metrics['equal_map_loss'])
    if activation_only:result['movement_function_preserved']=assert_movement_preserved(model,original_motion)
    atomic_json(output/'result.json',result)
    if not integrity:raise ValueError('Source/protected artifact changed during fit')
    return result


class FastNavigator:
    def __init__(self, mesh, checkpoint=None, model=None, terrain=None,base_checkpoint=None,cache_dir=None):
        torch.set_num_threads(1)
        self.decoder={}
        if model is None:
            saved=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
            if saved['schema']!=SCHEMA:raise ValueError('Navigator checkpoint schema mismatch')
            self.decoder=saved.get('decoder',{}).copy()
            model=FastMapPolicy(**saved['config']);model.load_state_dict(saved['model'])
        self.model=model.eval();self.model.requires_grad_(False)
        self.mesh=mesh
        self.encoding_cache_hit=False
        if cache_dir and checkpoint:
            from nav.marble_dynamics.static_cache import cached_encoding
            self.geo,self.encoded,self.encoding_cache_hit=cached_encoding(mesh,model,checkpoint,cache_dir)
        self.base=None
        if model.residual:
            if terrain is None or base_checkpoint is None:
                raise ValueError('Residual navigator requires immutable base checkpoint and terrain')
            from nav.marble_dynamics.navigator import FrozenNavigator
            self.base=FrozenNavigator(terrain,base_checkpoint)
        if not (cache_dir and checkpoint):
            self.geo=geometry(mesh)
            with torch.inference_mode():
                self.encoded=model.encode_map(*(torch.from_numpy(self.geo[k]) for k in
                                              ('features','bbox_lo','bbox_hi')))
        self.dynamics=None;self.last_kind='gem'
        self.last_base_action=None
        self.last_unmasked_action=None

    def set_observation(self,dynamics,kind='gem'):
        self.dynamics=dynamics;self.last_kind=kind

    def act(self,raw,waypoint,gem,next_goal=None):
        if self.dynamics is None:raise ValueError('Current native observation required')
        base_action=self.base.act(raw,waypoint,gem,next_goal) if self.base is not None else None
        self.last_base_action=list(base_action) if base_action is not None else None
        x,q=features(self.dynamics,raw,gem,waypoint,self.geo['center'],self.geo['scale'],self.last_kind,base_action)
        with torch.inference_mode():
            output=self.model.forward_cached(torch.from_numpy(x)[None],torch.from_numpy(q)[None],self.encoded)
        action=decode_action(output[0].numpy(),base_action,self.model.axis_mode)
        self.last_unmasked_action=list(action)
        return legal_action(action,self.dynamics,raw) if self.decoder.get('legal_actions',False) else action

    def sent_action(self,action,before):
        if self.base is not None:self.base.sent_action(action,before)

    def outcome(self,before,after):
        if self.base is not None:self.base.outcome(before,after)

    def reset(self):
        self.dynamics=None
        if self.base is not None:
            from nav.marble_dynamics.navigator import FrozenNavigator
            self.base=FrozenNavigator(self.base.terrain,model=self.base.model)


def benchmark(dataset, checkpoint, output):
    manifest=json.loads((dataset/'manifest.json').read_text());metrics={}
    for name in manifest['maps']:
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));t=time.perf_counter()
        from nav.terrain import TerrainMap,TerrainGrid
        nav=FastNavigator(mesh,checkpoint,terrain=TerrainGrid(TerrainMap.resolve(name)),
                          base_checkpoint=ML/'models/nav/real_maps_20261010_A2/nav_latest.pth');load=time.perf_counter()-t
        with np.load(dataset/(name+'.npz'),allow_pickle=False) as d:
            ids=np.flatnonzero(d['dev'])[:128];x=d['x'][ids];q=d['q'][ids]
        times=[]
        with torch.inference_mode():
            for i in range(144):
                j=i%len(x);t=time.perf_counter()
                p=nav.model.forward_cached(torch.from_numpy(x[j])[None],torch.from_numpy(q[j])[None],nav.encoded)
                decode_action(p[0].numpy(),axis_mode=nav.model.axis_mode)
                if i>=16:times.append(time.perf_counter()-t)
        metrics[name]=dict(triangles=len(mesh.triangles),load_and_cache_s=load,
            median_ms=float(np.median(times))*1000,p95_ms=float(np.percentile(times,95))*1000,
            max_ms=max(times)*1000)
    report=dict(schema=SCHEMA,checkpoint_sha256=sha(checkpoint),maps=metrics,
        scope='Cached CPU forward/decode only; map load, routing and observation extraction excluded.',
        gameplay_improvement_established=False)
    atomic_json(output,report);return report


def causal_activation_advantage(info,forecast,baseline_info,baseline):
    """A longer forecast alone cannot justify spending a power or Blast."""
    if baseline is None:return False
    if forecast['requested_replies']!=baseline['requested_replies']:return False
    for prediction in (forecast,baseline):
        if prediction['failure'] or prediction['ambiguous_intervals'] or len(prediction['controls'])!=prediction['requested_replies']:return False
    if info['rejected'] is not None:return False
    if info.get('resources_consumed',0)<=baseline_info.get('resources_consumed',0):return False
    if baseline_info['rejected'] not in (None,'no_supported_continuation'):return False
    if not baseline_info['stable_continuation']:return True
    pickup=info['possible_pickup_interval'];old_pickup=baseline_info['possible_pickup_interval']
    if pickup is not None or old_pickup is not None:
        return pickup is not None and (old_pickup is None or pickup<old_pickup)
    return info['cost']<baseline_info['cost']-.3


def causal_action_advantage(info,forecast,baseline_info,baseline):
    """Compare this decision with current A2, holding the future reply tail fixed."""
    if baseline is None or forecast['requested_replies']!=baseline['requested_replies']:return False
    for prediction in (forecast,baseline):
        if prediction['failure'] or prediction['ambiguous_intervals'] or len(prediction['controls'])!=prediction['requested_replies']:return False
        if len(prediction['proposed'])!=prediction['requested_replies']:return False
    if forecast['proposed'][1:]!=baseline['proposed'][1:]:return False
    if info['rejected'] is not None or not info['stable_continuation']:return False
    if baseline_info['rejected'] not in (None,'no_supported_continuation'):return False
    if not baseline_info['stable_continuation']:return True
    pickup,old=info['possible_pickup_interval'],baseline_info['possible_pickup_interval']
    if pickup is not None or old is not None:return pickup is not None and (old is None or pickup<old)
    return info['cost']<baseline_info['cost']-.3


def ensure_case_journal(folder):
    # Zero accepted candidates are valid evidence, not a missing-file failure.
    path=folder/'cases.jsonl.gz'
    if not path.exists():
        with gzip.open(path,'wt',encoding='utf-8') as stream:pass


def filter_corrections(advice,dataset,output,dll,query_s=1.,maps=None,wall_s=1200,cache_dir=None,defer_replay=False):
    """Keep only audited advice with a matched current-A2 first-reply advantage."""
    from nav.terrain import TerrainGrid,TerrainMap
    from nav.routing import RoutePlanner
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.bridge import predict_replies
    from nav.marble_dynamics.search import assess
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    original=json.loads((advice/'manifest.json').read_text())
    old_protocol=json.loads((advice/'protocol.json').read_text())
    data_manifest=json.loads((dataset/'manifest.json').read_text())
    if maps:data_manifest={**data_manifest,'maps':maps}
    verified=load_advice(advice,data_manifest,defer_replay)
    audited,evidence_path=advice_evidence(advice,defer_replay)
    cases_path=advice/original.get('cases_path','cases.json')
    if sha(cases_path)!=original['cases_sha256'] or sha(cases_path)!=audited['cases_sha256']:
        raise ValueError('Original case evidence changed')
    from nav.marble_dynamics.scaling_loop import read_cases
    sources={str(p):sha(p) for p in list(Path(__file__).parent.glob('*.py'))+list((ML/'nav').glob('*.py'))+list((ML/'nav/collision').glob('*.py'))}
    for path,digest in original['sources'].items():
        if Path(path).suffix not in ('.py','.cpp'):
            if sha(path)!=digest:raise ValueError('Original advice input changed: '+path)
            sources[path]=digest
    for p in [dll,dataset/'manifest.json',advice/'manifest.json',evidence_path,cases_path,ML/'nav/collision/sweep_kernel.cpp']+[advice/(n+'.npz') for n in data_manifest['maps']]:sources[str(p)]=sha(p)
    output.mkdir(parents=True,exist_ok=False)
    ensure_case_journal(output)
    snapshot_sources(output,[Path(p) for p in sources if Path(p).suffix in ('.py','.cpp')])
    protected=protected_hashes();start=time.perf_counter();records={};attempts=0
    atomic_json(output/'protocol.json',dict(schema='analytical_navigation_advice_v1',sources=sources,protected=protected,
        rollouts=old_protocol['rollouts'],dataset=str(dataset),parent_advice=str(advice),query_wall_s=query_s,total_wall_s=wall_s,
        cache_dir=str(cache_dir) if cache_dir else None,first_reply_causal_corrections=True,contract='Identical initial state, queued controls, horizon and all later replies; only first reply replaced by recorded current A2. No invented recurrent A2 future. Analytical targets, not native success.'))
    torch.set_num_threads(1)
    for name in data_manifest['maps']:
        journal=next((Path(old_protocol['rollouts'])/name).glob('*/fast_navigator/*/decisions.jsonl.gz'))
        with gzip.open(journal,'rt') as stream:rows={r['decision']:r for r in map(json.loads,stream)}
        mesh,terrain=advice_context(name,dll,cache_dir)
        route=RoutePlanner(terrain)
        engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
        original_cases=(c for c in read_cases(cases_path) if c['map']==name and c['accepted'])
        extra=verified[name];selected=[];xs=[];qs=[];ys=[];excluded=Counter();map_attempts=0
        for ix,case in enumerate(original_cases):
            if time.perf_counter()-start>=wall_s:raise RuntimeError('Correction filtering wall limit')
            map_attempts+=1
            row=rows[case['decision']];d=row['before'];gem=row['choice']['actor_goal_context']['destination']
            forecast=case['forecast'];baseline_replies=copy.deepcopy(forecast['proposed'])
            baseline_replies[0]=list(row['choice']['base_actor_action'])
            route.reset();route.navigate(d['native']['state']['position'],gem)
            chosen=next(i for i in case['candidates'] if i['label']==case['label'])
            baseline=None;info=None;reason=None;t=time.perf_counter();engine.planning_deadline=t+query_s
            try:
                baseline=predict_replies(engine,d,baseline_replies,stop_on_ambiguity=True)
                info=assess(engine,route,d,gem,baseline,'recorded_current_A2_first_reply')
                if not causal_action_advantage(chosen,forecast,info,baseline):reason='no_verified_first_reply_advantage'
            except PlanningBudgetExceeded:reason='baseline_deadline'
            except ValueError as exc:reason='unsupported_baseline: '+str(exc)
            finally:engine.planning_deadline=None
            attempts+=1
            if reason is None:
                saved=copy.deepcopy(case);saved['forecast']['action_baseline']=baseline
                selected_info=next(i for i in saved['candidates'] if i['label']==saved['label'])
                selected_info['action_comparison']=dict(advantage=True,baseline=info,first_reply_only=True)
                saved['first_reply_baseline_wall_s']=time.perf_counter()-t
                xs.append(extra['x'][ix].numpy());qs.append(extra['q'][ix].numpy());ys.append(extra['y'][ix].numpy());selected.append(case['decision'])
            else:
                excluded[reason]+=1;saved=dict(map=name,decision=case['decision'],accepted=False,
                    source_sha256=case['source_sha256'],parent_label=case['label'],rejection_reason=reason,baseline=info)
            with gzip.open(output/'cases.jsonl.gz','at',encoding='utf-8') as stream:stream.write(json.dumps(saved)+'\n')
        np.savez_compressed(output/(name+'.npz'),x=np.asarray(xs,np.float32).reshape(-1,data_manifest['state_size']),
            q=np.asarray(qs,np.float32).reshape(-1,4,3),y=np.asarray(ys,np.float32).reshape(-1,9),decision=np.asarray(selected,np.int32))
        records[name]=dict(attempts=map_attempts,accepted=len(xs),excluded=dict(excluded),tensor_sha256=sha(output/(name+'.npz')))
        persist_advice_context(mesh,terrain,cache_dir)
        atomic_json(output/'progress.json',dict(maps=records,complete=False));print(json.dumps(dict(filtered_corrections=name,**records[name])),flush=True)
    integrity=all(sha(p)==h for p,h in sources.items()) and protected_hashes()==protected
    result=dict(schema='analytical_navigation_advice_v1',complete=integrity,maps=records,sources=sources,protected=protected,
        elapsed_wall_s=time.perf_counter()-start,native_outcomes=False,cases_path='cases.jsonl.gz',cases_sha256=sha(output/'cases.jsonl.gz'),
        first_reply_causal_corrections=True,parent_advice=str(advice),attempts=attempts)
    atomic_json(output/'manifest.json',result)
    collection_integrity(output,result)
    if not integrity:raise ValueError('Correction filter source/protected artifacts changed')
    return result


def advice_state_key(x,q):
    """Exact model-input identity across journals; signed zero is immaterial."""
    value=np.concatenate((np.asarray(x).ravel(),np.asarray(q).ravel())).astype('<f4')
    if not np.isfinite(value).all():raise ValueError('Nonfinite correction state')
    value[value==0]=0
    return hashlib.sha256(value.tobytes()).hexdigest()


def collect_advice(rollouts,dataset,output,dll,max_per_map=96,query_s=1.,maps=None,wall_s=1200,cache_dir=None,candidate_limit=None,exclude_states=None):
    """Static analytical advice on CURRENT encountered states, not native outcomes."""
    from nav.terrain import TerrainGrid,TerrainMap
    from nav.routing import RoutePlanner
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.bridge import predict_replies
    from nav.marble_dynamics.search import SequenceProposal,assess,useful_candidate
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    output.mkdir(parents=True,exist_ok=False)
    ensure_case_journal(output)
    original=json.loads((dataset/'manifest.json').read_text())
    if not original.get('residual'):raise ValueError('Advice requires recorded-base residual interface')
    summary=json.loads((rollouts/'summary.json').read_text())
    prepared=json.loads((rollouts/'prepared.json').read_text())
    selected=[r for r in summary['branches'] if r['branch']=='fast_navigator']
    names=maps or original['maps']
    if any(len([r for r in selected if r['map']==n])!=1 for n in names) or any(not r.get('score_valid') or not r.get('declared_inputs_valid') for r in selected):
        raise ValueError('Complete declared learned-controller pool required')
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    excluded_states=json.loads(Path(exclude_states).read_text()) if exclude_states else {}
    if exclude_states:sources[str(exclude_states)]=sha(exclude_states)
    for p in list((ML/'nav').glob('*.py'))+list((ML/'nav/collision').glob('*.py'))+[dll,dataset/'manifest.json',rollouts/'prepared.json',rollouts/'summary.json',ML/'nav/collision/sweep_kernel.cpp']:
        sources[str(p)]=sha(p)
    for name in names:
        report=next(r for r in selected if r['map']==name)
        journal=rollouts/name/str(report['seed'])/'fast_navigator'/str(report['round'])/'decisions.jsonl.gz'
        sources[str(journal)]=sha(journal)
        sources[str(ML/'collision_maps'/(name+'.npz'))]=sha(ML/'collision_maps'/(name+'.npz'))
        sources[str(ML/'collision_maps'/(name+'.json'))]=sha(ML/'collision_maps'/(name+'.json'))
        from nav.terrain import TerrainMap
        terrain_source=TerrainMap.resolve(name)
        sources[str(terrain_source)]=sha(terrain_source)
    snapshot_sources(output,[Path(p) for p in sources if Path(p).suffix in ('.py','.cpp')])
    protected=protected_hashes();started=time.perf_counter();records={};case_count=0
    atomic_json(output/'protocol.json',dict(schema='analytical_navigation_advice_v1',rollouts=str(rollouts),dataset=str(dataset),max_per_map=max_per_map,query_wall_s=query_s,total_wall_s=wall_s,seed=101315,candidate_limit=candidate_limit,cache_dir=str(cache_dir) if cache_dir else None,exclude_states=str(exclude_states) if exclude_states else None,sources=sources,protected=protected,
        scope='Current native states; new analytically checked controls are imitation targets, not native pickup/fall outcomes. Normal radius/ordinary gravity only; hidden-world state remains incomplete. No future state enters inputs.',
        activation_contract='Actual activation must beat completed unambiguous same-horizon no-activation feedback controls with identical jump/steering/braking profile. No comparator => no activation label.'))
    torch.set_num_threads(1);rng=np.random.RandomState(101315)
    queried_keys={}
    for name in names:
        if time.perf_counter()-started>=wall_s:raise RuntimeError('Advice collection wall limit')
        report=next(r for r in selected if r['map']==name)
        path=rollouts/name/str(report['seed'])/'fast_navigator'/str(report['round'])/'decisions.jsonl.gz'
        sources[str(path)]=sha(path)
        with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
        mesh,terrain=advice_context(name,dll,cache_dir)
        route=RoutePlanner(terrain);engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
        g=geometry(mesh);unique={};excluded=Counter()
        for row in rows:
            state=row['before']['native']['state'];choice=row['choice'];goal=choice.get('actor_goal_context');base=choice.get('base_actor_action')
            if base is None or goal is None or state['mode'] or state['oob']:continue
            if abs(state['radius']-.18975)>1e-6 or int(row['raw_observation'][38])==6 or not np.allclose(state['gravity_direction'],[0,0,-1],atol=1e-7):
                excluded['unverified_radius_mega_or_gravity_context']+=1;continue
            key=tuple(np.round(np.asarray(state['position'])*2).astype(int))+tuple(np.round(state['velocity']).astype(int))+tuple(np.round(goal['destination']).astype(int))+(int(row['raw_observation'][38]),)
            key+=tuple(np.round(np.asarray(state['spin'])/5).astype(int))
            key+=tuple(np.round(np.asarray(state['best_contact_normal'])*20).astype(int))
            pending=row['before']['native']['pending_move']
            key+=tuple(pending[k] for k in ('x','y','jump','use'))
            key+=tuple(row['before']['bridge_action'][k] for k in (4,6,8))+(row['before']['pow_yaw_hold'],)
            key+=tuple((bool(active),round(left/500) if active else 0) for active,left in state['effects'])
            if key not in unique:unique[key]=row
        seen=set(excluded_states.get(name,[]));pool=[];input_keys={}
        for row in unique.values():
            goal=row['choice']['actor_goal_context']
            x,q=features(row['before'],row['raw_observation'],goal['destination'],goal['waypoint'],g['center'],g['scale'],base_action=row['choice']['base_actor_action'])
            key=advice_state_key(x,q)
            if key in seen:
                excluded['previously_queried_state']+=1;continue
            seen.add(key);pool.append(row);input_keys[row['decision']]=key
        rng.shuffle(pool)
        # Alternate action-disagreement/power states and ordinary coverage;
        # no map-specific launch, landing or maneuver classifier.
        priority=[r for r in pool if r['choice']['base_actor_action'][4]!=r['action'][4] or int(r['raw_observation'][38])>0]
        priority_ids={r['decision'] for r in priority}
        ordinary=[r for r in pool if r['decision'] not in priority_ids]
        chosen_rows=[]
        while len(chosen_rows)<max_per_map and (priority or ordinary):
            if priority:chosen_rows.append(priority.pop())
            if ordinary and len(chosen_rows)<max_per_map:chosen_rows.append(ordinary.pop())
        xs=[];qs=[];ys=[];accepted=[];query_times=[]
        for row in chosen_rows:
            if time.perf_counter()-started>=wall_s:raise RuntimeError('Advice collection wall limit')
            d=row['before'];goal=row['choice']['actor_goal_context'];gem=goal['destination']
            route.reset();route.navigate(d['native']['state']['position'],gem);start_distance=route.distance(d['native']['state']['position'])
            held=int(d['native']['state']['held']);steps=96 if d['native']['state']['effects'][5][0] else 24
            specs=[('jump_now',dict(jump=0)),('route',{}),('direct',dict(direct=True)),('jump_2',dict(jump=2)),('jump_5',dict(jump=5))]
            if held:specs=[('use_now',dict(use=0)),('use_jump_now',dict(use=0,jump=0))]+specs
            if d.get('blast',{}).get('value',0)>=d.get('blast',{}).get('required',.2) or d.get('blast',{}).get('special'):
                specs=[('blast_jump_now',dict(blast=0,jump=0)),('blast_now',dict(blast=0))]+specs
            if candidate_limit is not None:specs=specs[:candidate_limit]
            pairs=[];deadline=False;error=None;t=time.perf_counter();engine.planning_deadline=t+query_s
            try:
                for label,spec in specs:
                    # Actual activations need enough time to land: 3.072s for
                    # ordinary powers, 6.144s for Helicopter. Merely holding a
                    # power still leaves rolling forecasts at the short horizon.
                    horizon=max(steps,96 if held==5 else 48) if 'use' in spec else steps
                    prediction=predict_replies(engine,d,SequenceProposal(route,gem,horizon,**spec),stop_on_ambiguity=True)
                    info=assess(engine,route,d,gem,prediction,label)
                    activation='use' in spec or 'blast' in spec
                    if activation and info['rejected'] is None:
                        baseline_spec={k:v for k,v in spec.items() if k not in ('use','blast')}
                        # The no-activation partner uses the SAME horizon; other
                        # ordinary proposals retain their shorter horizons.
                        baseline=predict_replies(engine,d,SequenceProposal(route,gem,horizon,**baseline_spec),stop_on_ambiguity=True)
                        baseline_info=assess(engine,route,d,gem,baseline,label+'_without_activation')
                        advantage=causal_activation_advantage(info,prediction,baseline_info,baseline)
                        info['activation_comparison']=dict(advantage=advantage,baseline=baseline_info,
                            horizon=horizon,requested_profile=spec,baseline_profile=baseline_spec)
                        if advantage:prediction['activation_baseline']=baseline
                        else:info['rejected']='no_verified_activation_advantage'
                    pairs.append((info,prediction))
            except PlanningBudgetExceeded:deadline=True
            except ValueError as exc:error=str(exc);excluded['unsupported_prediction']+=1
            finally:engine.planning_deadline=None
            choice=useful_candidate(pairs,route,start_distance)
            case=dict(map=name,decision=row['decision'],source_journal=str(path),rollout_seed=report['seed'],rollout_round=report['round'],source_sha256=sources[str(path)],completed_candidates=len(pairs),deadline=deadline,error=error,wall_s=time.perf_counter()-t,accepted=choice is not None,candidates=[info for info,_ in pairs])
            if choice:
                info,prediction=choice;action=prediction['proposed'][0]
                # This pool's fast live wrapper supplied the default 'gem'
                # kind; actor_goal_context deliberately has no kind field.
                x,q=features(d,row['raw_observation'],gem,goal['waypoint'],g['center'],g['scale'],'gem',base_action=row['choice']['base_actor_action'])
                xs.append(x);qs.append(q);ys.append(action_target(action));accepted.append(dict(decision=case['decision']))
                case.update(label=info['label'],action=action,endpoint=prediction['states'][-1]['position'],possible_pickup_interval=info['possible_pickup_interval'],forecast_replies=len(prediction['proposed']),forecast=prediction)
            else:excluded['no_completed_useful_supported_advice']+=1
            # Independent closed gzip members keep preceding cases recoverable
            # and avoid rewriting hundreds of megabytes after each query.
            with gzip.open(output/'cases.jsonl.gz','at',encoding='utf-8') as stream:
                stream.write(json.dumps(case)+'\n')
            case_count+=1;query_times.append(case['wall_s'])
            if len(query_times)%50==0:
                atomic_json(output/'query_progress.json',dict(map=name,attempts=len(query_times),accepted=len(xs),elapsed_s=time.perf_counter()-started))
        np.savez_compressed(output/(name+'.npz'),x=np.asarray(xs,np.float32).reshape(-1,original['state_size']),q=np.asarray(qs,np.float32).reshape(-1,4,3),y=np.asarray(ys,np.float32).reshape(-1,9),decision=np.asarray([c['decision'] for c in accepted],np.int32))
        records[name]=dict(attempts=len(chosen_rows),accepted=len(xs),query_median_s=float(np.median(query_times)) if query_times else None,query_p95_s=float(np.percentile(query_times,95)) if query_times else None,excluded=dict(excluded),tensor_sha256=sha(output/(name+'.npz')))
        queried_keys[name]=[input_keys[r['decision']] for r in chosen_rows]
        persist_advice_context(mesh,terrain,cache_dir)
        atomic_json(output/'progress.json',dict(maps=records,complete=False));print(json.dumps(dict(advice_map=name,**records[name])),flush=True)
    atomic_json(output/'queried_states.json',queried_keys)
    integrity=all(sha(p)==v for p,v in sources.items()) and protected_hashes()==protected
    result=dict(schema='analytical_navigation_advice_v1',complete=integrity,maps=records,sources=sources,protected=protected,elapsed_wall_s=time.perf_counter()-started,native_outcomes=False,
        cases_path='cases.jsonl.gz',cases_sha256=sha(output/'cases.jsonl.gz'),attempts=case_count,
        queried_states_path='queried_states.json',queried_states_sha256=sha(output/'queried_states.json'))
    atomic_json(output/'manifest.json',result)
    collection_integrity(output,result)
    if not integrity:raise ValueError('Advice sources changed')
    return result


def audit_advice(folder,dataset,dll):
    """Reconstruct every input/label and replay accepted legal control forecasts."""
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.bridge import predict_replies
    from nav.marble_dynamics.search import terminal_support,assess
    from nav.terrain import TerrainGrid,TerrainMap
    from nav.routing import RoutePlanner
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.source_archive import verify_archive
    manifest=json.loads((folder/'manifest.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    original=json.loads((dataset/'manifest.json').read_text())
    if not manifest['complete'] or manifest['native_outcomes']:raise ValueError('Unsupported advice evidence')
    archived=verify_archive(folder/'source_snapshot.zip')
    for path,digest in manifest['sources'].items():
        # Historical implementation is preserved in its verified archive;
        # data, geometry and the exact compiled kernel must still be unchanged.
        if Path(path).suffix not in ('.py','.cpp') and sha(path)!=digest:raise ValueError('Advice source changed: '+path)
    from nav.marble_dynamics.scaling_loop import case_paths,read_cases
    checked=[];torch.set_num_threads(1);journal_rows={};journal_hashes={};attempt_count=0
    for shard in manifest.get('case_shards',[]):
        if sha(folder/shard['path'])!=shard['sha256']:raise ValueError('Correction shard changed')
    rollouts=Path(protocol['rollouts']) if protocol.get('rollouts') else None
    pool=json.loads((rollouts/'summary.json').read_text()) if rollouts else None
    for name in original['maps']:
        journal=None
        if pool:
            report=next(r for r in pool['branches'] if r['map']==name and r['branch']=='fast_navigator')
            journal=rollouts/name/str(report['seed'])/'fast_navigator'/str(report['round'])/'decisions.jsonl.gz'
        cache_dir=protocol.get('cache_dir')
        if cache_dir:
            mesh,comparison_terrain=advice_context(name,dll,Path(cache_dir))
        else:
            mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),dll)
            comparison_terrain=walking_terrain(TerrainGrid(TerrainMap.resolve(name)),mesh) if manifest.get('first_reply_causal_corrections') else None
        geo=geometry(mesh);engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
        comparison_route=RoutePlanner(comparison_terrain) if manifest.get('first_reply_causal_corrections') else None
        with np.load(dataset/(name+'.npz'),allow_pickle=False) as z:
            for key in geo:np.testing.assert_array_equal(geo[key],z[key])
        path=folder/(name+'.npz')
        if sha(path)!=manifest['maps'][name]['tensor_sha256']:raise ValueError('Advice tensor changed')
        map_cases=[c for path in case_paths(folder,manifest,name) for c in read_cases(path) if c['map']==name]
        attempt_count+=len(map_cases)
        accepted=[c for c in map_cases if c['accepted']]
        with np.load(path,allow_pickle=False) as z:
            if len(z['y'])!=len(accepted):raise ValueError('Case/tensor count mismatch')
            for i,case in enumerate(accepted):
                journal_path=case.get('source_journal') or protocol.get('journal_lookup',{}).get(case['source_sha256'])
                current_journal=Path(journal_path) if journal_path else journal
                if current_journal is None:raise ValueError('Missing exact source journal')
                key=str(current_journal)
                if key not in journal_rows:
                    with gzip.open(current_journal,'rt') as stream:journal_rows[key]={r['decision']:r for r in map(json.loads,stream)}
                    journal_hashes[key]=sha(current_journal)
                row=journal_rows[key][case['decision']];goal=row['choice']['actor_goal_context']
                x,q=features(row['before'],row['raw_observation'],goal['destination'],goal['waypoint'],geo['center'],geo['scale'],base_action=row['choice']['base_actor_action'])
                np.testing.assert_array_equal(z['x'][i],x);np.testing.assert_array_equal(z['q'][i],q)
                np.testing.assert_array_equal(z['y'][i],action_target(case['action']))
                if z['decision'][i]!=case['decision'] or case['source_sha256']!=journal_hashes[key]:raise ValueError('Advice provenance mismatch')
                forecast=case['forecast']
                if forecast['proposed'][0]!=case['action'] or forecast['failure'] or forecast['ambiguous_intervals']:raise ValueError('Invalid accepted forecast')
                replay=predict_replies(engine,row['before'],forecast['proposed'],stop_on_ambiguity=True)
                if replay['failure'] or replay['ambiguous_intervals'] or len(replay['states'])!=len(forecast['states']):raise ValueError('Advice forecast replay failed')
                error=max(float(np.max(np.abs(np.asarray(a[k])-b[k]))) for a,b in zip(replay['states'],forecast['states']) for k in ('position','velocity','spin','radius'))
                if error>1e-10 or replay['controls']!=forecast['controls']:raise ValueError('Advice forecast changed')
                if not terminal_support(mesh,replay['states'][-4:])[1]:raise ValueError('Unstable advice endpoint')
                chosen=next(c for c in case['candidates'] if c['label']==case['label'])
                if chosen['rejected'] is not None:raise ValueError('Rejected forecast was selected')
                if 'activation_comparison' in chosen:
                    comparison=chosen['activation_comparison'];baseline=forecast.get('activation_baseline')
                    if not comparison['advantage'] or not causal_activation_advantage(chosen,forecast,comparison['baseline'],baseline):
                        raise ValueError('Activation lacks matched advantage')
                    control=predict_replies(engine,row['before'],baseline['proposed'],stop_on_ambiguity=True)
                    if control['controls']!=baseline['controls'] or control['states']!=baseline['states']:
                        raise ValueError('Activation control partner changed')
                if 'action_comparison' in chosen:
                    comparison=chosen['action_comparison'];baseline=forecast.get('action_baseline')
                    if baseline is None or baseline['proposed'][0]!=row['choice']['base_actor_action']:
                        raise ValueError('First-reply comparison does not use recorded current A2')
                    if not comparison['first_reply_only'] or not comparison['advantage'] or not causal_action_advantage(chosen,forecast,comparison['baseline'],baseline):
                        raise ValueError('Correction lacks matched first-reply advantage')
                    control=predict_replies(engine,row['before'],baseline['proposed'],stop_on_ambiguity=True)
                    if control['controls']!=baseline['controls'] or control['states']!=baseline['states']:
                        raise ValueError('First-reply control partner changed')
                    comparison_route.reset();comparison_route.navigate(row['before']['native']['state']['position'],goal['destination'])
                    actual_info=assess(engine,comparison_route,row['before'],goal['destination'],replay,chosen['label'])
                    control_info=assess(engine,comparison_route,row['before'],goal['destination'],control,'recorded_current_A2_first_reply')
                    for fresh,saved in ((actual_info,chosen),(control_info,comparison['baseline'])):
                        for key in ('cost','rejected','stable_continuation','possible_pickup_interval','resources_consumed'):
                            if fresh[key]!=saved[key]:raise ValueError('First-reply advantage assessment changed: '+key)
                    if not causal_action_advantage(actual_info,replay,control_info,control):
                        raise ValueError('Replayed first-reply correction has no advantage')
                elif manifest.get('first_reply_causal_corrections'):
                    raise ValueError('Filtered correction lacks independent first-reply evidence')
                checked.append(dict(map=name,decision=case['decision'],forecast_steps=len(replay['controls']),maximum_motion_error=error))
        journal_rows.clear();journal_hashes.clear()
        print(json.dumps(dict(audited_advice=name,examples=len(accepted))),flush=True)
    result=dict(passed=True,manifest_sha256=sha(folder/'manifest.json'),cases_sha256=manifest['cases_sha256'],
        accepted=len(checked),attempts=attempt_count,archive_source_files=len(archived['files']),cases=checked,
        native_validation=False,contract='Exact current input/label reconstruction; replay of complete legal analytical control forecasts. Source geometry equals sealed prepared tensors. No native success claim.')
    atomic_json(folder/'independent_audit.json',result);return result


def mask_checkpoint(checkpoint,output):
    """Keep every learned weight; explicitly version only current availability."""
    output.mkdir(parents=True,exist_ok=False);protected=protected_hashes()
    source_sha=sha(checkpoint);saved=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    if saved['schema']!=SCHEMA:raise ValueError('Unsupported navigator checkpoint')
    original_decoder=saved.get('decoder',{}).copy()
    saved['decoder']=dict(original_decoder,legal_actions=True)
    saved['parent_checkpoint_sha256']=source_sha
    snapshot_sources(output,[Path(__file__)])
    torch.save(saved,output/'candidate.pth')
    reread=torch.load(str(output/'candidate.pth'),map_location='cpu',weights_only=False)
    if reread['config']!=saved['config'] or any(not torch.equal(v,reread['model'][k]) for k,v in saved['model'].items()):
        raise ValueError('Mask export changed learned model')
    if sha(checkpoint)!=source_sha or protected_hashes()!=protected:raise ValueError('Protected source changed')
    result=dict(schema=SCHEMA,parent_checkpoint=str(checkpoint),parent_sha256=source_sha,
        checkpoint_sha256=sha(output/'candidate.pth'),model_weights_unchanged=True,
        previous_decoder=original_decoder,decoder=reread['decoder'],policy_training=False,
        gameplay_improvement_established=False,protected_unchanged=True)
    atomic_json(output/'result.json',result);return result



FAILURE_GATE_PROTOCOL = dict(
    schema='recorded_student_forecast_failure_v1', replies=48, decision_ms=64,
    maximum_route_progress_u=.2, stall_displacement_u=.5, stall_speed_u_s=1.,
    below_map_margin_radii=2., preserve_weight=.1, correction_weight=1.,
    min_updates=2000, check_every=500, plateau_patience=5,
    minimum_relative_improvement=.005, minimum_absolute_improvement=.0001,
    forecast_contract='Replay the source student\'s actual recorded control sequence from its BEFORE state. This is an offline open-loop supervision gate, not a counterfactual current-policy closed-loop forecast. Future controls/outcomes never enter x/q or live inference.',
    unknown_contract='Contact ambiguity, unsupported physics, gaps, incomplete non-fall windows and external inventory/gravity/respawn changes are unknown, never failure labels.')


def student_control_window(rows, index, replies=48):
    """Return recorded student replies up to a world/control boundary."""
    first=rows[index]['before'];state=first['native']['state']
    clock=int(first['native']['frame_clock_ms']);object_id=first['native']['object_id']
    destination=rows[index]['choice']['actor_goal_context']['destination']
    actions=[];gem_delta=0.;stop=None
    previous=state
    for row in rows[index:index+replies]:
        current=row['before']['native'];s=current['state']
        if int(current['frame_clock_ms'])!=clock+len(actions)*64 or current['object_id']!=object_id:
            stop='clock_or_object_boundary';break
        if s['oob'] or s['mode']!=0:
            stop='nonplayable_boundary';break
        if s['radius']!=state['radius'] or s['gravity_direction']!=state['gravity_direction']:
            stop='external_geometry_or_gravity_transition';break
        if s['held']>0 and s['held']!=previous['held']:
            stop='external_inventory_pickup';break
        next_goal=row.get('choice',{}).get('actor_goal_context',{}).get('destination')
        if next_goal is None or not np.allclose(next_goal,destination,rtol=0,atol=.1):
            stop='student_goal_changed';break
        action=row.get('action');actor=row.get('choice',{}).get('actor_action')
        if action is None or actor is None or len(action)!=9 or not np.isfinite(action).all() or not np.allclose(action,actor,rtol=0,atol=1e-7):
            stop='non_student_control';break
        actions.append(list(action));previous=s
        outcome=row.get('outcome',{});gem_delta+=max(0.,float(outcome.get('gem_delta',0)))
        if outcome.get('fell') or outcome.get('reconnected') or outcome.get('round_ended'):
            stop='native_fall' if outcome.get('fell') else 'native_world_boundary';break
    if len(actions)<replies and stop is None:stop='journal_end'
    return dict(actions=actions,gem_delta=gem_delta,stop=stop,
        native_fall=stop=='native_fall',complete=len(actions)==replies)


def mission_inbounds(metadata):
    """Exact axis-aligned mission volume; refuse unfamiliar trigger transforms."""
    volumes=[r for r in metadata['mission_annotations'] if r.get('datablock','').lower()=='inboundstrigger']
    if len(volumes)!=1:raise ValueError('Exactly one audited InBoundsTrigger is required')
    v=volumes[0];rotation=np.fromstring(v['rotation'],sep=' ')
    poly=np.fromstring(v['polyhedron'],sep=' ');scale=np.fromstring(v['scale'],sep=' ')
    if len(rotation)!=4 or rotation[3]!=0 or poly.shape!=(12,) or scale.shape!=(3,) or np.any(scale<=0):
        raise ValueError('Unsupported InBoundsTrigger representation')
    edges=poly[3:].reshape(3,3)
    if not np.array_equal(np.abs(edges),np.eye(3)):raise ValueError('Non-axis-aligned trigger polyhedron')
    corners=np.asarray([poly[:3]+np.asarray([a,b,c])@edges for a in (0,1) for b in (0,1) for c in (0,1)])
    world=corners*scale+v['position'];return world.min(axis=0),world.max(axis=0)


def fully_outside_inbounds(state, bounds):
    lo,hi=bounds;p=np.asarray(state['position']);r=float(state['radius'])
    return bool(np.any(p+r<lo) or np.any(p-r>hi))


def classify_student_failure(rollout, mesh, route, destination, window, bounds):
    """Fail only on a complete, unambiguous physical prediction, or certain fall."""
    from nav.marble_dynamics.search import terminal_support
    states=rollout['states'];end=states[-1];start=states[0]
    result=dict(accepted=False,reason=None,predicted_replies=len(states)-1,
        window_stop=window['stop'],native_gem_delta=window['gem_delta'],
        failure=rollout['failure'],ambiguous_intervals=rollout['ambiguous_intervals'],
        end_position=end['position'],end_velocity=end['velocity'])
    if rollout['failure'] or rollout['ambiguous_intervals'] or len(states)!=len(window['actions'])+1:
        result['reason']='unknown_prediction';return result
    if window['gem_delta']>0:
        result['reason']='student_collected_gem';return result
    if fully_outside_inbounds(start,bounds):
        result['reason']='unknown_already_outside_trigger';return result
    if any(fully_outside_inbounds(s,bounds) for s in rollout['native_states'][1:]):
        result.update(accepted=True,reason='fall_leaves_mission_inbounds');return result
    # Below every static collidable triangle while descending under normal
    # gravity is a certain geometry fall; no arbitrary map-specific drop threshold.
    minimum_z=float(np.min(mesh.triangles[:,:,2]))
    if (np.array_equal(start['gravity_direction'],[0,0,-1]) and end['position'][2]<minimum_z-2*end['radius']
            and end['velocity'][2]<-1):
        result.update(accepted=True,reason='fall_below_all_map_surfaces',minimum_map_z=minimum_z);return result
    if not window['complete']:
        result['reason']='unknown_incomplete_window';return result
    supported,stable,normal_speed=terminal_support(mesh,states[-4:])
    result.update(terminal_supported=supported,terminal_stable=stable,normal_speed=normal_speed)
    if not supported:
        result['reason']='unknown_airborne_endpoint';return result
    route.reset();route.navigate(start['position'],destination)
    a=route.distance(start['position']);b=route.distance(end['position'])
    if not math.isfinite(a) or not math.isfinite(b):
        result['reason']='unknown_route_distance';return result
    progress=float(a-b);displacement=float(np.linalg.norm(np.asarray(end['position'])-start['position']))
    tail_speed=max(float(np.linalg.norm(s['velocity'])) for s in states[-4:])
    result.update(route_progress_u=progress,displacement_u=displacement,tail_speed_u_s=tail_speed)
    if stable and displacement<=.5 and tail_speed<=1:
        result.update(accepted=True,reason='stall')
    elif stable and progress<=.2:
        result.update(accepted=True,reason='no_progress')
    else:result['reason']='student_forecast_progress'
    return result


def failure_gate_map(job):
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.bridge_small_step import predict_small_step
    from nav.marble_dynamics.scaling_loop import read_cases
    from nav.routing import RoutePlanner
    name=job['map'];folder=Path(job['advice']);out=Path(job['output']);dataset=Path(job['dataset'])
    manifest=json.loads((folder/'manifest.json').read_text());torch.set_num_threads(1)
    mesh,terrain=advice_context(name,Path(job['dll']),Path(job['cache']))
    metadata_path=ML/'collision_maps'/(name+'.json');metadata=json.loads(metadata_path.read_text());bounds=mission_inbounds(metadata)
    route=RoutePlanner(terrain);engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
    geo=geometry(mesh);cases_path=next(s for s in manifest['case_shards'] if s['map']==name)
    if sha(folder/cases_path['path'])!=cases_path['sha256']:raise ValueError('Case shard changed')
    tensor=folder/(name+'.npz')
    if sha(tensor)!=manifest['maps'][name]['tensor_sha256']:raise ValueError('Advice tensor changed')
    with np.load(tensor,allow_pickle=False) as z:original={k:z[k].copy() for k in ('x','q','y','decision')}
    journal_lookup={h:p for p,h in manifest['sources'].items() if p.endswith('decisions.jsonl.gz')}
    journal_cache={};kept=[];reasons=Counter();labels=Counter();started=time.perf_counter();i=0
    with gzip.open(out/(name+'.gate.jsonl.gz'),'wt',encoding='utf-8') as evidence, gzip.open(out/(name+'.cases.jsonl.gz'),'wt',encoding='utf-8') as retained:
        for case in read_cases(folder/cases_path['path']):
            if not case['accepted']:continue
            path=Path(journal_lookup[case['source_sha256']])
            if str(path) not in journal_cache:
                if sha(path)!=case['source_sha256']:raise ValueError('Student source journal changed')
                rows=list(read_cases(path));lookup={r['decision']:j for j,r in enumerate(rows)}
                journal_cache[str(path)]=(rows,lookup)
            rows,lookup=journal_cache[str(path)];index=lookup[case['decision']];row=rows[index]
            goal=row['choice']['actor_goal_context']
            x,q=features(row['before'],row['raw_observation'],goal['destination'],goal['waypoint'],geo['center'],geo['scale'],base_action=row['choice']['base_actor_action'])
            np.testing.assert_array_equal(x,original['x'][i]);np.testing.assert_array_equal(q,original['q'][i])
            np.testing.assert_array_equal(action_target(case['action']),original['y'][i])
            if int(original['decision'][i])!=case['decision']:raise ValueError('Correction alignment changed')
            window=student_control_window(rows,index,FAILURE_GATE_PROTOCOL['replies'])
            if window['actions']:
                prediction=predict_small_step(engine,row['before'],window['actions'],stop_on_ambiguity=True)
                verdict=classify_student_failure(prediction,mesh,route,goal['destination'],window,bounds)
            else:prediction=None;verdict=dict(accepted=False,reason='unknown_empty_window')
            if verdict['accepted'] and any(fully_outside_inbounds(s,bounds) for s in case['forecast']['native_states']):
                verdict.update(accepted=False,reason='correction_also_leaves_mission_inbounds')
            record=dict(map=name,input_index=i,source_sha256=case['source_sha256'],decision=case['decision'],
                destination=goal['destination'],student_controls=window['actions'],window=window,
                verdict=verdict,student_forecast=prediction)
            evidence.write(json.dumps(record,separators=(',',':'))+'\n');reasons[verdict['reason']]+=1
            if verdict['accepted']:
                kept.append(i)
                retained.write(json.dumps(case,separators=(',',':'))+'\n')
                for button,col in (('jump',6),('use',7),('Blast',8)):
                    labels[button]+=int(original['y'][i,col]>.5)
            i+=1
            if i%100==0:atomic_json(out/(name+'.progress.json'),dict(checked=i,accepted=len(kept),reasons=dict(reasons)))
    if i!=len(original['y']):raise ValueError('Case/tensor count mismatch')
    np.savez_compressed(out/(name+'.npz'),**{k:v[kept] for k,v in original.items()})
    persist_advice_context(mesh,terrain,Path(job['cache']))
    report=dict(map=name,source_labels=i,accepted=len(kept),label_counts=dict(labels),reasons=dict(reasons),
        selected_original_indices=kept,tensor_sha256=sha(out/(name+'.npz')),
        evidence_sha256=sha(out/(name+'.gate.jsonl.gz')),cases_sha256=sha(out/(name+'.cases.jsonl.gz')),wall_s=time.perf_counter()-started)
    atomic_json(out/(name+'.result.json'),report);return report


def gate_failed_student(advice,dataset,output,dll,cache,workers=8):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    output.mkdir(parents=True,exist_ok=False)
    original=json.loads((advice/'manifest.json').read_text());advice_evidence(advice,True)
    if original['accepted']!=7261:raise ValueError('Ablation is sealed to the existing7261 corrections')
    code=[Path(__file__),ML/'nav/marble_dynamics/fast_nav.py',ML/'nav/marble_dynamics/bridge_small_step.py',ML/'nav/marble_dynamics/physics.py']
    sources={str(p):sha(p) for p in code+[advice/'manifest.json',dataset/'manifest.json',dll]}
    # Existing immutable inputs only; historical source hashes belong to their archives.
    for p,h in original['sources'].items():
        if p.endswith('decisions.jsonl.gz'):sources[p]=h
    for s in original['case_shards']:sources[str(advice/s['path'])]=s['sha256']
    for name,info in original['maps'].items():sources[str(advice/(name+'.npz'))]=info['tensor_sha256']
    for name in original['maps']:
        for p in (ML/'collision_maps'/(name+'.json'),ML/'collision_maps'/(name+'.npz')):sources[str(p)]=sha(p)
    import os
    snapshot_sources(output,code,root=Path(os.path.commonpath([str(p.resolve().parent) for p in code])))
    atomic_json(output/'protocol.json',dict(FAILURE_GATE_PROTOCOL,advice=str(advice),dataset=str(dataset),sources=sources,
        exact_original_input_count=7261,new_collection=False,workers=workers,cache_dir=str(cache),
        journal_lookup={h:p for p,h in original['sources'].items() if p.endswith('decisions.jsonl.gz')}))
    jobs=[dict(map=name,advice=str(advice),dataset=str(dataset),output=str(output),dll=str(dll),cache=str(cache)) for name in original['maps']]
    started=time.perf_counter();reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(failure_gate_map,j) for j in jobs]):
            report=f.result();reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='selected_original_indices'}),flush=True)
    if not all(sha(p)==h for p,h in sources.items()):raise ValueError('Failure gate sources changed')
    accepted=sum(r['accepted'] for r in reports)
    if not accepted:raise ValueError('No qualified failure corrections; no fit authorized on an empty gate')
    manifest=dict(schema='analytical_navigation_advice_v1',complete=True,accepted=accepted,source_labels=7261,
        maps={r['map']:r for r in reports},sources=sources,first_reply_causal_corrections=True,
        student_forecast_failure_gate=True,native_outcomes=False,independent_replay_pending=True,
        student_native_outcomes_used_for_supervision=True,
        case_shards=[dict(map=r['map'],path=r['map']+'.cases.jsonl.gz',sha256=r['cases_sha256']) for r in reports],
        cases_sha256=sha(output/'protocol.json'),original_advice=str(advice),gate_wall_s=time.perf_counter()-started)
    atomic_json(output/'manifest.json',manifest);collection_integrity(output,manifest)
    atomic_json(output/'result.json',dict(complete=True,accepted=accepted,source_labels=7261,new_collection=False,
        reports=reports,wall_s=time.perf_counter()-started));return manifest


def explicit_preservation_loss(pred,x,y,base_count,preserve_weight=.1,correction_weight=1.):
    if not 0<base_count<=len(pred) or preserve_weight<=0 or correction_weight<=0:
        raise ValueError('Positive explicit weights and nonempty preserve-base anchors required')
    anchor=base_preservation_objective(pred[:base_count],x[:base_count,-9:],axis_mode='linear')
    correction=objective(pred[base_count:],y[base_count:],torch.ones(len(pred)-base_count),axis_mode='linear') if len(pred)>base_count else pred.sum()*0
    return preserve_weight*anchor+correction_weight*correction


def plateau_step(best,stale,loss,relative=.005,absolute=.0001):
    if not math.isfinite(loss):raise ValueError('Nonfinite convergence monitor')
    if not math.isfinite(best) or best-loss>=max(absolute,abs(best)*relative):return loss,0
    return best,stale+1


def advice_monitor_weights(y,count=12):
    """Expected sampling distribution of the unchanged action-balanced minibatch."""
    if not len(y):return torch.empty(0)
    weight=torch.zeros(len(y));remaining=count
    for button in (7,8):
        positive=y[:,button]>.5;n=min(2,remaining) if positive.any() else 0
        if n:weight[positive]+=float(n)/float(positive.sum())/count
        remaining-=n
    weight+=remaining/count/len(y);return weight


def ablation_batch_indices(data,extra,names,update,rng):
    """The unchanged ablation sampler, also used to restore its RNG cursor."""
    name=names[update%len(names)];d=data[name];a=extra[name]
    count=12 if len(a['y']) else 0
    ix=rng.choice(d['train'],48-count,replace=len(d['train'])<48-count)
    ax=advice_indices(a,rng,count,True) if count else np.empty(0,dtype=np.int64)
    return name,ix,ax


def fit_failure_ablation(dataset,advice,initial,output,wall_s=1200,max_updates=30000,resume=False):
    """One explicit loss ratio, convergence stop; safety exhaustion is incomplete."""
    output.mkdir(parents=True,exist_ok=resume);manifest=json.loads((dataset/'manifest.json').read_text())
    gated=json.loads((advice/'manifest.json').read_text())
    if not gated.get('student_forecast_failure_gate') or gated['source_labels']!=7261:raise ValueError('Requires sealed failure-gated corpus')
    torch.set_num_threads(4);torch.manual_seed(101313);rng=np.random.RandomState(101313)
    data=load_data(dataset,manifest);extra=load_advice(advice,manifest,True)
    saved=torch.load(str(initial),map_location='cpu',weights_only=False)
    model=FastMapPolicy(**saved['config']);model.load_state_dict(saved['model'])
    if not model.residual or model.axis_mode!='linear' or not saved.get('decoder',{}).get('legal_actions'):
        raise ValueError('The exact preserve-base linear masked architecture is required')
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);sources={str(p):sha(p) for p in [Path(__file__),dataset/'manifest.json',advice/'manifest.json',advice/'collection_integrity.json',initial]}
    for n in manifest['maps']:sources[str(advice/(n+'.npz'))]=sha(advice/(n+'.npz'))
    protected=protected_hashes();previous=None;start_update=0;elapsed_before=0.;history=[]
    if resume:
        previous=json.loads((output/'result.json').read_text())
        prior=json.loads((output/'protocol.json').read_text())
        if previous['converged'] or not previous['sources_unchanged']:
            raise ValueError('Only an intact safety-limited fit can continue')
        if any(prior[k]!=str(v) for k,v in (('dataset',dataset),('advice',advice),('initial',initial))):
            raise ValueError('Resume must use the same corpus and initial checkpoint')
        if any(sha(p)!=h for p,h in prior['sources'].items() if Path(p).resolve()!=Path(__file__).resolve()) or prior['protected']!=protected:
            raise ValueError('Ablation inputs changed before continuation')
        if sha(output/'candidate.pth')!=previous['checkpoint_sha256']:
            raise ValueError('Selected training checkpoint changed before continuation')
        last=torch.load(str(output/'last.pth'),map_location='cpu',weights_only=False)
        if last['update']!=previous['updates'] or last['converged']:
            raise ValueError('Last optimizer checkpoint does not match incomplete fit')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        start_update=last['update'];elapsed_before=previous['wall_s']
        if max_updates<=start_update:raise ValueError('Continuation safety ceiling must exceed completed updates')
        history=json.loads((output/'history.json').read_text())
        stage=len(list(output.glob('continuation_*.json')))+1
        atomic_json(output/('continuation_'+str(stage)+'.json'),dict(previous_result=previous,
            start_update=start_update,max_wall_s=wall_s,safety_max_updates=max_updates,
            sources=sources,protected=protected,recipe_changed=False,
            optimizer_restored=True,rng_restored='Exact sampler replay for initial stage; saved RNG state thereafter'))
        snapshot_sources(output/('continuation_'+str(stage)),[Path(__file__)])
    else:
        snapshot_sources(output,[Path(__file__)])
        atomic_json(output/'protocol.json',dict(FAILURE_GATE_PROTOCOL,dataset=str(dataset),advice=str(advice),initial=str(initial),
        max_wall_s=wall_s,safety_max_updates=max_updates,seed=101313,batch=48,advice_batch=12,
        separately_normalized_losses=True,advice_action_sampling=True,architecture_changed=False,
        convergence_monitor='Fixed training anchors and every gated training correction weighted by the existing action-balanced sampler, equal maps. Development demonstrations reported only, never selected on or trained.',
            sources=sources,protected=protected))
    monitor={}
    for n,d in data.items():
        ids=rng.choice(d['train'],min(512,len(d['train'])),replace=False)
        monitor[n]=dict(anchors=ids,advice_weight=advice_monitor_weights(extra[n]['y']))
    if resume:
        if 'rng_state' in last:rng.set_state(last['rng_state'])
        else:
            for cursor in range(start_update):ablation_batch_indices(data,extra,manifest['maps'],cursor,rng)
    started=time.perf_counter();best=float('inf');stale=0;best_exact=float('inf');best_update=0;converged=False
    for item in history:
        value=item['equal_map_training_objective'];best,stale=plateau_step(best,stale,value)
        if value<best_exact:best_exact=value;best_update=item['update']
    def measure():
        model.eval();metrics={}
        with torch.no_grad():
            for n,d in data.items():
                ix=monitor[n]['anchors'];a=extra[n];aw=monitor[n]['advice_weight'];base=0.;corr=0.
                for j in range(0,len(ix),128):
                    z=ix[j:j+128];p=model(d['x'][z],d['q'][z],d['features'],d['bbox_lo'],d['bbox_hi'])
                    base+=float(base_preservation_objective(p,d['x'][z,-9:],axis_mode='linear'))*len(z)
                for j in range(0,len(a['y']),128):
                    z=slice(j,j+128);p=model(a['x'][z],a['q'][z],d['features'],d['bbox_lo'],d['bbox_hi'])
                    corr+=float(objective(p,a['y'][z],aw[z],axis_mode='linear'))*float(aw[z].sum())
                base/=len(ix);metrics[n]=dict(preserve_loss=base,correction_loss=corr,combined=.1*base+corr,corrections=len(a['y']))
        model.train();return dict(equal_map_training_objective=float(np.mean([v['combined'] for v in metrics.values()])),per_map=metrics)
    for update in range(start_update,max_updates+1):
        already_measured=bool(history and history[-1]['update']==update)
        if (update%500==0 or update==max_updates) and not already_measured:
            metrics=measure();loss=metrics['equal_map_training_objective'];best,stale=plateau_step(best,stale,loss)
            metrics.update(update=update,stale_checks=stale,wall_s=elapsed_before+time.perf_counter()-started);history.append(metrics)
            if loss<best_exact:
                best_exact=loss;best_update=update
                torch.save(dict(schema=SCHEMA,config=model.config,model=model.state_dict(),update=update,metrics=metrics,
                    decoder=saved['decoder'],dataset_sha256=sha(dataset/'manifest.json'),selection='Lowest fixed training-objective checkpoint of one convergence fit; native performance unknown'),output/'candidate.pth')
            atomic_json(output/'history.json',history);print(json.dumps(dict(update=update,training_objective=loss,stale_checks=stale,wall_s=metrics['wall_s'])),flush=True)
            if update>=2000 and stale>=5:converged=True;break
        if time.perf_counter()-started>=wall_s or update==max_updates:break
        name,ix,ax=ablation_batch_indices(data,extra,manifest['maps'],update,rng)
        d=data[name];a=extra[name];count=len(ax);x=d['x'][ix];q=d['q'][ix];y=d['y'][ix]
        if count:
            x=torch.cat([x,a['x'][ax]]);q=torch.cat([q,a['q'][ax]]);y=torch.cat([y,a['y'][ax]])
        pred=model(x,q,d['features'],d['bbox_lo'],d['bbox_hi']);loss=explicit_preservation_loss(pred,x,y,48-count)
        if not torch.isfinite(loss):raise ValueError('Nonfinite ablation loss')
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    torch.save(dict(schema=SCHEMA,config=model.config,model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,
        decoder=saved['decoder'],dataset_sha256=sha(dataset/'manifest.json'),converged=converged,rng_state=rng.get_state()),output/'last.pth')
    # Report sealed development performance once; selection uses training convergence, not Dev.
    selected=torch.load(str(output/'candidate.pth'),map_location='cpu',weights_only=False);model.load_state_dict(selected['model'])
    dev=validate(model,data,full_development=True)
    unchanged=all(sha(p)==h for p,h in sources.items()) and protected_hashes()==protected
    result=dict(complete=converged,converged=converged,updates=update,best_update=best_update,best_training_objective=best_exact,
        stop='training_plateau' if converged else 'safety_limit_incomplete',wall_s=elapsed_before+time.perf_counter()-started,
        sources_unchanged=unchanged,checkpoint_sha256=sha(output/'candidate.pth'),development=dev,
        protected_unchanged=unchanged,gameplay_improvement_established=False)
    atomic_json(output/'result.json',result)
    if not unchanged:raise ValueError('Ablation sources/protected artifacts changed')
    return result



def audit_student_failure_gate(folder,dataset,dll,cache):
    """Selected-candidate audit: replay each retained own-control failure gate."""
    from nav.marble_dynamics.scaling_loop import read_cases
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.marble_dynamics.bridge_small_step import predict_small_step
    from nav.routing import RoutePlanner
    from nav.source_archive import verify_archive
    manifest=json.loads((folder/'manifest.json').read_text())
    if not manifest.get('student_forecast_failure_gate') or manifest['source_labels']!=7261:
        raise ValueError('Not the declared single-ablation failure gate')
    verify_archive(folder/'source_snapshot.zip')
    for path,digest in manifest['sources'].items():
        if Path(path).suffix not in ('.py','.cpp') and sha(path)!=digest:
            raise ValueError('Failure gate immutable source changed: '+path)
    lookup={h:p for p,h in manifest['sources'].items() if p.endswith('decisions.jsonl.gz')}
    original=Path(manifest['original_advice']);checked=[];started=time.perf_counter()
    for name,report in manifest['maps'].items():
        if sha(folder/(name+'.gate.jsonl.gz'))!=report['evidence_sha256']:
            raise ValueError('Failure/refusal evidence changed')
        if sha(folder/(name+'.npz'))!=report['tensor_sha256']:
            raise ValueError('Gated tensors changed')
        mesh,terrain=advice_context(name,dll,cache);engine=GeneralPhysics(mesh);engine.bridge_physics_ms=16
        route=RoutePlanner(terrain);geo=geometry(mesh)
        bounds=mission_inbounds(json.loads((ML/'collision_maps'/(name+'.json')).read_text()))
        with np.load(folder/(name+'.npz'),allow_pickle=False) as z:new={k:z[k].copy() for k in ('x','q','y','decision')}
        with np.load(original/(name+'.npz'),allow_pickle=False) as z:
            for key,value in new.items():np.testing.assert_array_equal(value,z[key][report['selected_original_indices']])
        journals={};count=0
        for evidence in read_cases(folder/(name+'.gate.jsonl.gz')):
            if not evidence['verdict']['accepted']:continue
            source=evidence['source_sha256']
            if source not in journals:
                rows=list(read_cases(lookup[source]));journals[source]=(rows,{r['decision']:j for j,r in enumerate(rows)})
            rows,index=journals[source];i=index[evidence['decision']];row=rows[i]
            window=student_control_window(rows,i,FAILURE_GATE_PROTOCOL['replies'])
            if window!=evidence['window'] or window['actions']!=evidence['student_controls']:
                raise ValueError('Accepted gate did not use the actual source student controls')
            goal=row['choice']['actor_goal_context']
            x,q=features(row['before'],row['raw_observation'],goal['destination'],goal['waypoint'],geo['center'],geo['scale'],base_action=row['choice']['base_actor_action'])
            np.testing.assert_array_equal(x,new['x'][count]);np.testing.assert_array_equal(q,new['q'][count])
            prediction=predict_small_step(engine,row['before'],window['actions'],stop_on_ambiguity=True)
            verdict=classify_student_failure(prediction,mesh,route,goal['destination'],window,bounds)
            if verdict!=evidence['verdict'] or not verdict['accepted']:
                raise ValueError('Independent student failure gate replay changed')
            saved=evidence['student_forecast']
            if prediction['controls']!=saved['controls'] or len(prediction['native_states'])!=len(saved['native_states']):
                raise ValueError('Student forecast timing/control replay changed')
            error=max(float(np.max(np.abs(np.asarray(a[k])-b[k]))) for a,b in zip(prediction['native_states'],saved['native_states']) for k in ('position','velocity','spin'))
            if error>1e-7:raise ValueError('Student forecast motion replay changed')
            checked.append(dict(map=name,decision=evidence['decision'],reason=verdict['reason'],maximum_motion_error=error));count+=1
        if count!=report['accepted']:raise ValueError('Accepted gate count mismatch')
        print(json.dumps(dict(audited_student_gate=name,accepted=count)),flush=True)
    if len(checked)!=manifest['accepted']:raise ValueError('Incomplete student gate audit')
    result=dict(passed=True,manifest_sha256=sha(folder/'manifest.json'),accepted=len(checked),cases=checked,
        elapsed_wall_s=time.perf_counter()-started,source_controls_and_inputs_reconstructed=True,
        contract='Independent replay of each accepted recorded-student forecast failure. Original teacher advantage/targets require the separate audit-advice check; no native success or counterfactual current-policy claim.')
    atomic_json(folder/'student_failure_gate_audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--teacher',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--residual',action='store_true',help='Learn map-aware corrections around recorded current A2 replies')
    train=sub.add_parser('fit');train.add_argument('--dataset',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True);train.add_argument('--updates',type=int,default=1500)
    train.add_argument('--wall-s',type=float,default=900.)
    train.add_argument('--advice',type=Path);train.add_argument('--initial',type=Path)
    train.add_argument('--balanced-buttons',action='store_true')
    train.add_argument('--linear-axes',action='store_true')
    train.add_argument('--advice-actions',action='store_true',help='Reserve two validated use and two Blast-positive examples within each advice minibatch when available; no live exploration floor')
    train.add_argument('--activation-only',action='store_true',help='Freeze geometry, state and movement/jump outputs; train only existing use/Blast output rows')
    train.add_argument('--defer-advice-replay',action='store_true',help='Scaling only: checked collection integrity; full replay deferred to selected candidate')
    train.add_argument('--preserve-base',action='store_true',help='Anchor original train examples to current A2 actions; learn only independently matched first-reply corrections')
    bench=sub.add_parser('benchmark');bench.add_argument('--dataset',type=Path,required=True)
    bench.add_argument('--checkpoint',type=Path,required=True);bench.add_argument('--output',type=Path,required=True)
    advice=sub.add_parser('advice');advice.add_argument('--rollouts',type=Path,required=True);advice.add_argument('--dataset',type=Path,required=True);advice.add_argument('--output',type=Path,required=True);advice.add_argument('--sweep-dll',type=Path,required=True);advice.add_argument('--max-per-map',type=int,default=96);advice.add_argument('--query-s',type=float,default=1.)
    audit=sub.add_parser('audit-advice');audit.add_argument('--folder',type=Path,required=True)
    audit.add_argument('--dataset',type=Path,required=True);audit.add_argument('--sweep-dll',type=Path,required=True)
    filtered=sub.add_parser('filter-corrections');filtered.add_argument('--advice',type=Path,required=True)
    filtered.add_argument('--dataset',type=Path,required=True);filtered.add_argument('--output',type=Path,required=True)
    filtered.add_argument('--sweep-dll',type=Path,required=True);filtered.add_argument('--query-s',type=float,default=1.)
    mask=sub.add_parser('mask-checkpoint');mask.add_argument('--checkpoint',type=Path,required=True)
    mask.add_argument('--output',type=Path,required=True)
    gated=sub.add_parser('gate-student-failures');gated.add_argument('--advice',type=Path,required=True)
    gated.add_argument('--dataset',type=Path,required=True);gated.add_argument('--output',type=Path,required=True)
    gated.add_argument('--sweep-dll',type=Path,required=True);gated.add_argument('--cache-dir',type=Path,required=True)
    gated.add_argument('--workers',type=int,default=8)
    ablation=sub.add_parser('fit-failure-ablation');ablation.add_argument('--dataset',type=Path,required=True)
    ablation.add_argument('--advice',type=Path,required=True);ablation.add_argument('--initial',type=Path,required=True)
    ablation.add_argument('--output',type=Path,required=True);ablation.add_argument('--wall-s',type=float,default=1200)
    ablation.add_argument('--safety-max-updates',type=int,default=30000)
    ablation.add_argument('--resume',action='store_true',help='Continue the same safety-limited fit with restored optimizer/sampler; no new recipe')
    gate_audit=sub.add_parser('audit-student-failures');gate_audit.add_argument('--folder',type=Path,required=True)
    gate_audit.add_argument('--dataset',type=Path,required=True);gate_audit.add_argument('--sweep-dll',type=Path,required=True)
    gate_audit.add_argument('--cache-dir',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.teacher.resolve(),a.output.resolve(),a.residual)
    elif a.command=='fit':result=fit(a.dataset.resolve(),a.output.resolve(),a.updates,a.wall_s,
        a.advice.resolve() if a.advice else None,a.initial.resolve() if a.initial else None,a.balanced_buttons,a.linear_axes,a.advice_actions,a.activation_only,a.preserve_base,a.defer_advice_replay)
    elif a.command=='benchmark':result=benchmark(a.dataset.resolve(),a.checkpoint.resolve(),a.output.resolve())
    elif a.command=='advice':result=collect_advice(a.rollouts.resolve(),a.dataset.resolve(),a.output.resolve(),a.sweep_dll.resolve(),a.max_per_map,a.query_s)
    elif a.command=='audit-advice':result=audit_advice(a.folder.resolve(),a.dataset.resolve(),a.sweep_dll.resolve())
    elif a.command=='filter-corrections':result=filter_corrections(a.advice.resolve(),a.dataset.resolve(),a.output.resolve(),a.sweep_dll.resolve(),a.query_s)
    elif a.command=='gate-student-failures':result=gate_failed_student(a.advice.resolve(),a.dataset.resolve(),a.output.resolve(),a.sweep_dll.resolve(),a.cache_dir.resolve(),a.workers)
    elif a.command=='fit-failure-ablation':result=fit_failure_ablation(a.dataset.resolve(),a.advice.resolve(),a.initial.resolve(),a.output.resolve(),a.wall_s,a.safety_max_updates,a.resume)
    elif a.command=='audit-student-failures':result=audit_student_failure_gate(a.folder.resolve(),a.dataset.resolve(),a.sweep_dll.resolve(),a.cache_dir.resolve())
    else:result=mask_checkpoint(a.checkpoint.resolve(),a.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','protected','source_hashes','cases')}),flush=True)


if __name__=='__main__':main()
