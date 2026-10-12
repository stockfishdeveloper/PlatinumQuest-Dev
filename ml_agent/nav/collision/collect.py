"""Owned, sequential game probes. Run with: python -m nav.collision.collect.

Raw files and runtime logs live outside OneDrive. Converted artifacts are atomic.
No optimizer or trained policy is loaded, and only our own game PID is stopped.
"""
import argparse
import collections
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json, convert_snapshot
from nav.collision.mission import annotations_and_coverage
from nav.env import HuntEnv
from nav.protocol import NOOP_ACTION, RAW_POS, RAW_VEL, CONTACT_FIELDS, format_teleport

ROOT = Path(__file__).resolve().parents[3]
ML = ROOT / 'ml_agent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_file(env, command, path):
    """Bridge requests are asynchronous relative to the pending observations."""
    if path.exists():
        path.unlink()
    env.control(command)
    for _ in range(128):
        env.step(NOOP_ACTION, repeat=1)
        if path.exists() and path.stat().st_size:
            return
    raise RuntimeError('No probe output: '+command.split()[0])


def validation_rays(mesh, seed):
    rng = np.random.RandomState(seed)
    centers = mesh.triangles.mean(axis=1)
    # Every retained triangle, every orientation. Native float32 endpoints.
    a = centers + mesh.normals * .15
    b = centers - mesh.normals * .15
    lo, hi = mesh.bounds
    random_a = rng.uniform(lo-2, hi+2, (400, 3))
    random_b = rng.uniform(lo-2, hi+2, (400, 3))
    # Vertical columns and horizontal long segments add empty-space/occlusion checks.
    xy = rng.uniform(lo[:2], hi[:2], (200, 2))
    vertical_a = np.column_stack([xy, np.full(len(xy), hi[2]+2)])
    vertical_b = np.column_stack([xy, np.full(len(xy), lo[2]-2)])
    rays = np.concatenate([np.stack([a, b], axis=1),
                           np.stack([random_a, random_b], axis=1),
                           np.stack([vertical_a, vertical_b], axis=1)]).astype(np.float32)
    return rays


def compare_rays(mesh, rays, output):
    native = [json.loads(x) for x in output.read_text().splitlines()]
    if len(native) != len(rays):
        raise RuntimeError('Missing engine ray results')
    disagreements, count = [], collections.Counter()
    max_error = 0.
    t0 = time.perf_counter()
    for i, (ray, actual) in enumerate(zip(rays, native)):
        predicted = mesh.raycast(ray[0], ray[1])
        reason = None
        if bool(predicted) != actual['hit']:
            reason = 'hit_miss'
        elif predicted:
            error = float(np.linalg.norm(np.asarray(predicted['point'])-actual['point']))
            max_error = max(max_error, error)
            if error > .01:
                reason = 'point'
            elif abs(float(np.dot(predicted['normal'], actual['normal']))) < .999:
                reason = 'normal'
            elif predicted['object_id'] != actual['object_id']:
                # Exact coincident surfaces may legally have different tie winners.
                reason = 'coincident_object'
        count['hit' if actual['hit'] else 'miss'] += 1
        if reason:
            count[reason] += 1
            disagreements.append(dict(id=i, reason=reason, ray=ray.tolist(), engine=actual, mesh=predicted))
    return dict(kind='native_container_castRay_vs_collision_triangles', rays=len(rays),
                every_triangle_tested=len(mesh.triangles), counts=dict(count),
                max_point_error_u=max_error, elapsed_s=time.perf_counter()-t0,
                disagreements=disagreements,
                caveat='DTS castRay can use LOS detail instead of contact collision detail; investigate mismatches.')


def validation_sweeps(mesh, seed):
    rng = np.random.RandomState(seed)
    centers = mesh.triangles.mean(axis=1)
    # Test every triangle with a small sphere, plus marble-size trials.
    start = centers+mesh.normals*.16
    end = centers-mesh.normals*.15
    small = np.column_stack([start, end, np.full(len(start), .01)])
    ids = rng.choice(len(start), min(500, len(start)), replace=False)
    marble = np.column_stack([centers[ids]+mesh.normals[ids]*.34,
                              centers[ids]-mesh.normals[ids]*.15, np.full(len(ids), .19)])
    # Edge/vertex approaches exercise more than flat face intersections.
    ids = rng.choice(len(start), min(200, len(start)), replace=False)
    targets = mesh.triangles[ids, 0]*.5+mesh.triangles[ids, 1]*.5
    edges = np.column_stack([targets+mesh.normals[ids]*.34,
                             targets-mesh.normals[ids]*.15, np.full(len(ids), .19)])
    lo,hi=mesh.bounds
    # Known clear air above ALL exported geometry: explicit negative controls.
    a,b=rng.uniform(lo[:2],hi[:2],(200,2)),rng.uniform(lo[:2],hi[:2],(200,2))
    empty=np.column_stack([a,np.full(200,hi[2]+10),b,np.full(200,hi[2]+12),np.full(200,.19)])
    # Long sweeps matter to airborne planning. Start clear of nearby surfaces;
    # this is a geometric clearance check, not a solid-occupancy certification.
    starts=[]
    for i in rng.permutation(len(centers)):
        s=centers[i]+mesh.normals[i]*.5
        if not len(mesh.sphere_contacts(s,.19)['triangle_ids']):
            starts.append(s)
        if len(starts)==200:
            break
    starts=np.asarray(starts)
    ends=rng.uniform(lo-5,hi+5,(len(starts),3))
    long=np.column_stack([starts,ends,np.full(len(starts),.19)])
    return np.concatenate([small, marble, edges,empty,long]).astype(np.float32)


def compare_sweeps(mesh, sweeps, output):
    native = [json.loads(x) for x in output.read_text().splitlines()]
    if len(native) != len(sweeps):
        raise RuntimeError('Missing engine swept-sphere results')
    mismatches, counts = [], collections.Counter()
    max_error = 0.
    max_surface_error = 0.
    for i, (sweep, actual) in enumerate(zip(sweeps, native)):
        predicted = mesh.sweep_sphere(sweep[:3], sweep[3:6], float(sweep[6]))
        reason = None
        if bool(predicted) != actual['hit']:
            reason = 'hit_miss'
        elif predicted:
            error = float(np.linalg.norm(np.asarray(predicted['center'])-actual['center']))
            max_error = max(max_error, error)
            if error > .002:
                reason = 'center'
            elif np.dot(predicted['normal'], actual['normal']) < .999:
                reason = 'normal'
            elif any(abs(predicted['material'][k]-actual[k]) > .002 for k in ('friction','restitution','force')):
                contacts = mesh.sphere_contacts(predicted['center'],float(sweep[6])+.002)
                nearby = set(int(x) for x in mesh.material_ids[contacts['triangle_ids']])
                matches = any(all(abs(mesh.info['materials'][mi][k]-actual[k])<.002
                                  for k in ('friction','restitution','force')) for mi in nearby)
                reason = 'material_boundary_tie' if matches else 'material'
        counts['hit' if actual['hit'] else 'miss'] += 1
        if reason:
            counts[reason] += 1
            mismatches.append(dict(id=i, reason=reason, sweep=sweep.tolist(), engine=actual, mesh=predicted))
        if actual['hit']:
            point=np.asarray(actual['point'])
            ids=mesh.query_aabb(point-.01,point+.01)
            surface_error=float(np.min(np.linalg.norm(mesh.closest_points(point,ids)-point,axis=1))) if len(ids) else float('inf')
            max_surface_error=max(max_surface_error,surface_error)
            if surface_error>.002:
                counts['contact_point_unrepresented']+=1
                mismatches.append(dict(id=i,reason='contact_point_unrepresented',distance_u=surface_error,
                                       sweep=sweep.tolist(),engine=actual,mesh=predicted))
    return dict(kind='native_Marble_testMove_segmented_vs_collision_sweep', sweeps=len(sweeps),
                max_native_segment_u=1.,known_clear_negative_controls=200,long_sweep_controls=200,
                every_triangle_tested=len(mesh.triangles), counts=dict(counts),
                max_center_error_u=max_error,max_native_contact_surface_distance_u=max_surface_error,
                disagreements=mismatches)


def physics_checks(mesh, env, seed):
    """Sample real marble collisions; retain observed outcomes, never assume contact."""
    rng = np.random.RandomState(seed)
    centers, normals = mesh.triangles.mean(axis=1), mesh.normals
    groups = dict(floor=np.flatnonzero(normals[:,2]>.99),
                  wall=np.flatnonzero(np.abs(normals[:,2])<.01),
                  underside=np.flatnonzero(normals[:,2]<-.99),
                  slope=np.flatnonzero((normals[:,2]>.05)&(normals[:,2]<.99)))
    for oid in np.unique(mesh.object_ids):
        if mesh.objects[int(oid)]['class'] != 'InteriorInstance':
            groups['shape_'+str(oid)] = np.flatnonzero(mesh.object_ids==oid)
    selected = []
    for label, ids in groups.items():
        good = []
        for i in rng.permutation(ids):
            c, n = centers[i], normals[i]
            if len(mesh.sphere_contacts(c+n*.35, .19)['triangle_ids']):
                continue
            pred = mesh.sweep_sphere(c+n*.35, c-n*.15, .19)
            if pred and np.linalg.norm(np.asarray(pred['point'])-c)<.02:
                good.append(int(i))
            if len(good) >= (1 if label.startswith('shape_') else 4):
                break
        selected.extend((label, i) for i in good)
    env.control('CONTACT 1')
    results = []
    for label, i in selected:
        # Explicit test reset cancels pending server respawn schedules. A reset
        # here is a fixture operation, never an action/reward in training.
        env.control('RESPAWN')
        for _ in range(24):
            env.step(NOOP_ACTION, repeat=1)
        n, c = normals[i], centers[i]
        start, velocity = c+n*.35, -n*4
        env.step(NOOP_ACTION, repeat=1)
        env.control(format_teleport(*start, *velocity))
        for _ in range(2):
            env.step(NOOP_ACTION, repeat=1)
        observed = env.msg.obs[RAW_POS].astype(float)
        accepted = np.linalg.norm(observed-start)<.05
        samples = []
        for _ in range(12):
            msg, flags = env.step(NOOP_ACTION, repeat=1)
            samples.append(dict(position=msg.obs[RAW_POS].tolist(), velocity=msg.obs[RAW_VEL].tolist(),
                                fell=flags['fell'], telemetry=dict(zip(CONTACT_FIELDS,
                                  msg.extra.tolist())) if msg.extra is not None else None))
        contact = any(s['telemetry'] and (s['telemetry']['collisions']>0 or
                      s['telemetry']['contact_steps']>0) for s in samples)
        interrupted = any(np.linalg.norm(np.asarray(s['position'])-start)>3 for s in samples)
        results.append(dict(category=label, triangle_id=i, object_id=int(mesh.object_ids[i]),
                            target=c.tolist(), normal=n.tolist(), teleport_accepted=bool(accepted),
                            contact_seen=bool(contact), respawn_interrupted=bool(interrupted), samples=samples))
        if any(s['fell'] for s in samples):
            env.control('RESPAWN')
            for _ in range(50):
                env.step(NOOP_ACTION, repeat=1)
    return dict(trials=len(results), accepted=sum(x['teleport_accepted'] and not x['respawn_interrupted'] for x in results),
                interrupted=sum(x['respawn_interrupted'] for x in results),
                contacts=sum(x['teleport_accepted'] and not x['respawn_interrupted'] and x['contact_seen'] for x in results), results=results)


def collect(name, index, args, audit):
    work = Path(os.environ['TEMP']) / 'pq_checks/collision_maps_20261011'
    work.mkdir(parents=True, exist_ok=True)
    raw = work / (name+'.engine.jsonl')
    output = args.output / (name+'.npz')
    exe = ROOT / 'Marble Blast Platinum/marbleblast_geometry_probe.exe'
    port = args.port
    env = HuntEnv(port, speed=3, log=lambda x: print(x, flush=True))
    env.srv.settimeout(70)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    log = (work / (name+'.game.log')).open('wb')
    proc = subprocess.Popen([str(exe), '-autotrain', name, '-aiport', str(port),
                             '-offline', '-aicompetitive', '0'],
                            cwd=str(exe.parent), startupinfo=startup, stdout=log, stderr=log)
    begin = time.perf_counter()
    try:
        env.connect()
        env.conn.settimeout(40)
        if env.info.get('mission') != name:
            raise RuntimeError('Wrong mission: '+str(env.info))
        env.control('SPINWAIT 600000')
        env.control('FIXEDSTEP 16')
        for _ in range(400):
            env.step(NOOP_ACTION, repeat=1)
        command_file(env, 'GEOMETRYEXPORT '+raw.as_posix(), raw)
        mission = ROOT / audit['mission']
        if sha(mission) != audit['mission_sha256']:
            raise RuntimeError('Mission changed since inventory audit')
        metadata = dict(map=name, mission=audit['mission'], mission_sha256=sha(mission),
                        source_inventory=audit['source_files'], mission_classes=audit['mission_classes'],
                        probe_executable_sha256=sha(exe), game_info=env.info,
                        engine_instrumentation_sha256=sha(Path(__file__).with_name('engine_geometry.inc')))
        info = convert_snapshot(raw, output, metadata)
        info.update(annotations_and_coverage(mission, info['objects']))
        saved_raw = args.output/'snapshots'/(name+'.engine.jsonl.gz')
        saved_raw.parent.mkdir(parents=True,exist_ok=True)
        tmp = saved_raw.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
        os.replace(str(tmp),str(saved_raw))
        info['raw_snapshot_file'] = str(saved_raw.relative_to(args.output)).replace('\\','/')
        atomic_json(output.with_suffix('.json'),info)
        missing = [x for x in info['source_object_coverage'] if x['class_name'] in
                   ('InteriorInstance','TSStatic') and len(x['runtime_matches'])!=1]
        if missing:
            raise RuntimeError('Missing source colliders: '+str(missing))
        # Independently account for every hull-referenced DIF surface. A whole
        # object bounding box is not itself proof of complete hull selection.
        by_file={Path(x['path']).name.lower():x for x in info['source_inventory']}
        hull_coverage=[]
        for obj in info['objects']:
            if obj['class']!='InteriorInstance':
                continue
            source=by_file[Path(obj['interior_file']).name.lower()]
            expected=source['render_surfaces_referenced_by_hulls']+source['null_surfaces_referenced_by_hulls']
            hull_coverage.append(dict(object_id=obj['id'],source=obj['interior_file'],
                                      hull_referenced_surfaces=expected,exported_polygons=obj['polygons']))
            if obj['polygons']<expected:
                raise RuntimeError('Incomplete interior hull surface coverage: '+str(hull_coverage[-1]))
        info['interior_hull_surface_coverage']=hull_coverage
        atomic_json(output.with_suffix('.json'),info)
        mesh = CollisionMap(output)
        rays = validation_rays(mesh, 20261011+index)
        ray_input, ray_output = work/(name+'.rays.in'), work/(name+'.rays.jsonl')
        np.savetxt(ray_input, rays.reshape(-1, 6), fmt='%.9g')
        command_file(env, 'GEOMETRYRAYS '+ray_input.as_posix()+' '+ray_output.as_posix(), ray_output)
        report = compare_rays(mesh, rays, ray_output)
        sweeps = validation_sweeps(mesh, 20261011+index)
        sweep_input, sweep_output = work/(name+'.sweeps.in'), work/(name+'.sweeps.jsonl')
        np.savetxt(sweep_input, sweeps, fmt='%.9g')
        command_file(env, 'GEOMETRYSWEEPS '+sweep_input.as_posix()+' '+sweep_output.as_posix()+' 1', sweep_output)
        sweep_report = compare_sweeps(mesh, sweeps, sweep_output)
        for source in (ray_input,ray_output,sweep_input,sweep_output):
            target = args.output/'snapshots'/(source.name+'.gz')
            tmp = target.with_suffix('.tmp')
            tmp.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
            os.replace(str(tmp),str(target))
        physical = physics_checks(mesh, env, 20261011+index)
        atomic_json(args.output/(name+'.validation.json'), dict(mesh_sha256=info['mesh_sha256'],
                    rays=report, sweeps=sweep_report, physics=physical))
        print(json.dumps(dict(map=name, triangles=info['triangles'], objects=len(info['objects']),
                              classes=dict(collections.Counter(o['class'] for o in info['objects'])),
                              ray_counts=report['counts'], max_error=report['max_point_error_u'],
                              sweep_counts=sweep_report['counts'], sweep_error=sweep_report['max_center_error_u'],
                              physical_trials=physical['trials'], physical_contacts=physical['contacts'],
                              elapsed_s=round(time.perf_counter()-begin, 1))), flush=True)
        return dict(map=name, mesh_sha256=info['mesh_sha256'], triangles=info['triangles'],
                    objects=len(info['objects']), ray_counts=report['counts'],
                    sweep_counts=sweep_report['counts'], physics_contacts=physical['contacts'],
                    physics_trials=physical['trials'])
    finally:
        env.close()
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', nargs='+')
    p.add_argument('--port', type=int, default=9051)
    p.add_argument('--output', type=Path, default=ML/'collision_maps')
    args = p.parse_args()
    config = json.loads((ML/'configs/real_maps_20261009.json').read_text())
    inventory = json.loads((ML/'docs/MAP_REPRESENTATION_AUDIT_2026-10-11.json').read_text())
    by_name = {x['map']: x for x in inventory['maps']}
    results = []
    existing = args.output/'collection.json'
    if existing.exists():
        results = json.loads(existing.read_text())['maps']
    for name in args.maps or config['maps']:
        row = collect(name, config['maps'].index(name), args, by_name[name])
        results = [r for r in results if r['map']!=name]+[row]
        atomic_json(args.output/'collection.json', dict(maps=results, training_started=False))


if __name__ == '__main__':
    main()
