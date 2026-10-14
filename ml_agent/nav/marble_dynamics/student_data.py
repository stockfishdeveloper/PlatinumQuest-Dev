"""Native motion-learning inputs with every static collision triangle retained.

Preparation only. Native poses/labels remain authoritative in the linked raw
archives. Neural features use Float32, with motion targets formed in Float64
before conversion. No future state, gem score, map ID or reward is an input.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json
from nav.marble_dynamics.evaluate import load_rows
from nav.marble_dynamics.record import ML, DEV, TEST, sha
from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.state_copy import clone as clone_state

PARAMETERS=('max_roll_velocity','angular_acceleration','braking_acceleration',
    'static_friction','kinetic_friction','bounce_kinetic_friction','gravity',
    'max_dot_slide','bounce_restitution','air_acceleration','jump_impulse',
    'min_bounce_velocity','mass')
PARAMETER_SCALES=np.asarray([15,75,30,1,1,1,20,1,1,10,7.5,1,1],float)
MOTION_SCALES=np.asarray([1.,1.,1.,20.,20.,20.,100.,100.,100.])
MOVES=('x','y','z','yaw','pitch','jump','use','trigger1')
STATE_FEATURES=(['position_'+a for a in 'xyz']+['velocity_'+a for a in 'xyz']+
    ['spin_'+a for a in 'xyz']+['radius']+['gravity_'+a for a in 'xyz']+
    ['best_contact_'+a for a in 'xyz']+['camera_offset_'+a for a in 'xyz']+
    [key+'_'+trig for key in ('yaw','next_yaw','pitch','next_pitch') for trig in ('sin','cos')]+
    [key+str(i) for i in range(6) for key in ('effect_active_','effect_left_','effect_signed_clock_left_')]+
    ['held_'+str(i) for i in range(6)]+list(PARAMETERS)+
    ['effect_duration_'+str(i) for i in range(6)]+['ground_ms','smooth_camera','dt_s']+
    ['move_'+key for key in MOVES]+['clock_mod32','map_scale','radius_map'])
SURFACE_FEATURES=(['operative_v'+str(i)+'_'+a for i in range(3) for a in 'xyz']+
    ['source_v'+str(i)+'_'+a for i in range(3) for a in 'xyz']+
    ['normal_'+a for a in 'xyz']+['native_plane_'+a for a in 'xyz']+['native_plane_d']+
    ['friction','restitution','force']+['center_'+a for a in 'xyz']+
    ['extent_'+a for a in 'xyz']+['area']+['boundary_edge_'+str(i) for i in range(3)])


def motion(state):
    return np.concatenate([np.asarray(state[k],dtype=np.float64) for k in ('position','velocity','spin')])


def geometry(mesh):
    center=(mesh.bounds[0]+mesh.bounds[1])*.5
    scale=max(float(np.max(mesh.bounds[1]-mesh.bounds[0])),1.)
    operational=(mesh.triangles-center)/scale
    original=(mesh.source_triangles-center)/scale
    normals=mesh.normals.astype(np.float64)
    native_normals=mesh.planes[:,:3]
    plane_d=(mesh.planes[:,3]+native_normals@center)/scale
    materials=np.asarray([[mesh.info['materials'][int(i)][key] for key in
        ('friction','restitution','force')] for i in mesh.material_ids],dtype=np.float64)
    materials[:,2]/=100.
    areas=np.linalg.norm(np.cross(operational[:,1]-operational[:,0],
        operational[:,2]-operational[:,0]),axis=1)*.5
    features=np.column_stack([operational.reshape(-1,9),original.reshape(-1,9),
        normals,native_normals,plane_d,materials,operational.mean(axis=1),
        operational.max(axis=1)-operational.min(axis=1),areas,
        mesh.boundary_edges]).astype(np.float32)
    assert len(features)==len(mesh.source_triangles)==len(mesh.triangles)
    assert features.shape[1]==len(SURFACE_FEATURES)
    assert np.all(np.isfinite(features))
    return dict(features=features,bbox_lo=((mesh.lo-center)/scale).astype(np.float32),
        bbox_hi=((mesh.hi-center)/scale).astype(np.float32),center=center,scale=scale,
        triangle_ids=np.arange(len(features),dtype=np.int32),
        object_ids=mesh.object_ids,surface_keys=mesh.surface_keys,material_ids=mesh.material_ids)


def state_features(before,control,dt_ms,center,scale):
    state=clone_state(before);move=apply_control(state,control)
    held=int(state['held'])
    if not 0<=held<=5:raise ValueError('Unsupported native held type')
    params=state['parameters']
    values=list((np.asarray(state['position'],float)-center)/scale)
    values+=list(np.asarray(state['velocity'],float)/25.)
    values+=list(np.asarray(state['spin'],float)/100.)+[state['radius']/.18975]
    for key in ('gravity_direction','best_contact_normal','camera_offset'):
        values+=list(state[key])
    for key in ('yaw','next_yaw','pitch','next_pitch'):
        values.extend([np.sin(state[key]),np.cos(state[key])])
    for i,(active,left) in enumerate(state['effects']):
        # Native TTL is clamped at zero. Keep the signed deadline difference
        # too: an active effect exactly at expiry and one past expiry differ.
        signed=int(state['effect_end_clock_ms'][i])-int(state['clock_ms']) if active else 0
        values.extend([float(active),float(left)/10000.,signed/10000.])
    values.extend(float(held==i) for i in range(6))
    values+=list(np.asarray([params[k] for k in PARAMETERS],float)/PARAMETER_SCALES)
    values+=list(np.asarray(params['effect_duration_ms'],float)/10000.)
    values.extend([float(state['ground_ms'])/1000.,float(state['smooth_camera']),dt_ms/1000.])
    values.extend(float(move[k])/(np.pi if k in ('yaw','pitch') else 1.) for k in MOVES)
    values.extend([int(state['clock_ms'])%32/32.,scale/100.,state['radius']/scale])
    result=np.asarray(values,np.float32)
    assert len(result)==len(STATE_FEATURES),(len(result),len(STATE_FEATURES))
    if not np.all(np.isfinite(result)):raise ValueError('Nonfinite physical input')
    return result,state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--maps',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'source_snapshots';snapshot.mkdir()
    source_paths=[Path(__file__).with_name(name+'.py') for name in
        ('student_data','physics_student','rollout','state_copy','evaluate','record')]
    for path in source_paths:shutil.copy2(path,snapshot/path.name)
    shutil.copy2(ML/'nav/collision/mesh.py',snapshot/'collision_mesh.py')
    declaration=dict(schema='native_motion_student_v2',datasets=a.datasets,maps=a.maps,
        state_features=STATE_FEATURES,surface_features=SURFACE_FEATURES,
        motion_scales=MOTION_SCALES.tolist(),
        source_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in
            ('student_data','physics_student','rollout','state_copy','evaluate','record')},
        geometry_reader_sha256=sha(ML/'nav/collision/mesh.py'),
        split_warning='Map-level train/dev/test roles preserve existing split. Every map was already examined during analytical-engineering development; this is not a previously unseen map test.',
        fitting_started=False,policy_training_started=False,
        contract='Native16ms motion-only benchmark. All collision triangles, operational and source vertices, native planes, polygon-boundary flags and physical materials retained. Ordered control callbacks applied without truncation. Camera/effect metadata evolution, scripted activation, autoregressive validation and gameplay integration are separate future work. Float32 neural features are an approximation; original reviewed collision archives and native poses remain immutable. Terminal/OOB and within-step script-body transitions are excluded from this motion task; this dataset is not a fall-safety classifier.')
    atomic_json(a.output/'prepared.json',declaration)
    maps=[];sources=[]
    for name in a.maps:
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));geo=geometry(mesh)
        np.savez_compressed(a.output/(name+'.surfaces.npz'),**geo)
        xs=[];ys=[];source_ids=[];native_frames=[];strata=[];excluded=Counter()
        for dataset in a.datasets:
            root=ML/'datasets/marble_dynamics'/dataset/name
            if not (root/'manifest.json').exists():continue
            manifest=json.loads((root/'manifest.json').read_text())
            assert manifest['geometry_sha256']==mesh.info['mesh_sha256']
            for trial in manifest['trials']:
                if not trial['valid']:excluded['invalid_fixture']+=1;continue
                path=root/trial['file'];source_id=len(sources)
                raw_sha256=hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
                if raw_sha256!=trial['raw_sha256']:raise ValueError('Native archive provenance mismatch: '+str(path))
                sources.append(dict(path=str(path),sha256=sha(path),manifest=str(root/'manifest.json'),
                    manifest_sha256=sha(root/'manifest.json'),map=name,dataset=dataset,trial=trial['id'],
                    pair_group=trial.get('pair_group'),raw_sha256=raw_sha256,
                    family=trial['family'],intervention=trial.get('intervention')))
                for index,row in enumerate(load_rows(path)):
                    before=row['before'];after=row['after']
                    reason=None
                    if row['dt_ms']!=16:reason='different_physics_cadence'
                    elif before['mode']!=0 or after['mode']!=0 or before['oob'] or after['oob']:reason='nonplayable_or_oob'
                    elif before['radius']!=after['radius'] or before['parameters']!=after['parameters']:reason='script_body_transition'
                    elif not 0<=int(before['held'])<=5:reason='unsupported_native_inventory'
                    if reason:excluded[reason]+=1;continue
                    control=dict(move=row['move'],events=row['control_events'])
                    x,canonical=state_features(before,control,16,geo['center'],geo['scale'])
                    y=((motion(after)-motion(canonical))/MOTION_SCALES).astype(np.float32)
                    if not np.all(np.isfinite(y)):raise ValueError('Nonfinite native label')
                    xs.append(x);ys.append(y);source_ids.append(source_id);native_frames.append(index)
                    # Sampling classes only: no reward, policy-use bonus or outcome head.
                    strata.append(2 if row['move']['use'] and before['held'] else 1 if row['move']['jump'] else 0)
        if not xs:raise ValueError('No supported native inputs for '+name)
        np.savez_compressed(a.output/(name+'.steps.npz'),x=np.asarray(xs),y=np.asarray(ys),
            source_id=np.asarray(source_ids,np.int32),native_frame=np.asarray(native_frames,np.int32),
            stratum=np.asarray(strata,np.uint8))
        record=dict(map=name,role='dev' if name in DEV else 'test' if name in TEST else 'train',
            steps=len(xs),surfaces=len(geo['features']),geometry_sha256=mesh.info['mesh_sha256'],
            geometry_file_sha256=sha(mesh.path) if hasattr(mesh,'path') else sha(ML/'collision_maps'/(name+'.npz')),
            exclusions=dict(excluded),strata=dict(Counter(strata)))
        maps.append(record);print(json.dumps(record),flush=True)
        outputs={path.name:sha(path) for path in sorted(a.output.glob('*.npz'))}
        atomic_json(a.output/'manifest.json',dict(**declaration,map_records=maps,sources=sources,
            output_hashes=outputs,preparation_complete=len(maps)==len(a.maps)))


if __name__=='__main__':main()
