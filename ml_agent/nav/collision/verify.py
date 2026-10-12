"""Coverage, source identity, query timing and viewer/model consistency audit."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap, atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('ml_agent/collision_maps'))
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    names=json.loads((root/'ml_agent/configs/real_maps_20261009.json').read_text())['maps']
    rows=[]
    for name in names:
        mesh=CollisionMap(args.directory/(name+'.npz'))
        info=mesh.info
        raw=gzip.decompress((args.directory/info['raw_snapshot_file']).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==info['raw_snapshot_sha256']
        polygons=[r for r in map(json.loads,raw.decode().splitlines()) if r['kind']=='polygon']
        source_materials={(p['material'],p['friction'],p['restitution'],p['force'],p['properties_resolved']) for p in polygons}
        stored_materials={(m['name'],m['friction'],m['restitution'],m['force'],m['resolved']) for m in info['materials']}
        assert source_materials==stored_materials,(name,'material properties changed or missing')
        native_planes=collections.defaultdict(set)
        for poly in polygons:
            native_planes[(poly['object_id'],poly['surface_key'])].add(tuple(np.asarray(poly['normal']+[poly['plane_d']],dtype=np.float32)))
        assert all(tuple(plane) in native_planes[(int(oid),int(key))]
                   for plane,oid,key in zip(mesh.planes,mesh.object_ids,mesh.surface_keys)),(name,'native collision plane changed')
        # Independent vertex and area accounting catches lost features/transforms
        # without implementing the engine or repeating fan triangulation.
        expected=collections.defaultdict(set)
        expected_area=collections.defaultdict(float)
        actual_area=collections.defaultdict(float)
        for poly in polygons:
            v=np.asarray(poly['vertices'],dtype=np.float32).astype(float)
            expected[poly['object_id']].update(tuple(x) for x in v)
            key=(poly['object_id'],poly['surface_key'])
            expected_area[key]+=.5*np.linalg.norm(np.cross(v,np.roll(v,-1,axis=0)).sum(axis=0))
        for oid, vertices in expected.items():
            actual=set(tuple(x) for x in mesh.source_triangles[mesh.object_ids==oid].reshape(-1,3))
            assert vertices==actual,(name,oid,'exported vertex missing or shifted')
        area=.5*np.linalg.norm(np.cross(mesh.source_triangles[:,1]-mesh.source_triangles[:,0],
                                      mesh.source_triangles[:,2]-mesh.source_triangles[:,0]),axis=1)
        for oid,key,value in zip(mesh.object_ids,mesh.surface_keys,area):
            actual_area[(int(oid),int(key))]+=float(value)
        area_error=max(abs(v-actual_area[k]) for k,v in expected_area.items())
        # Native float32 vertices can deviate slightly from their stored plane;
        # vector polygon area and summed triangle areas then differ in rounding.
        assert all(abs(v-actual_area[k])<max(1e-5,abs(v)*1e-6) for k,v in expected_area.items()),(name,'polygon area coverage')
        for source in info['source_inventory']:
            assert sha(root/source['path'])==source['sha256']
        assets=[]
        for obj in info['objects']:
            if not obj['polygons']:
                continue
            filename=obj['interior_file'] or obj['shape_file'] or obj['shape_name']
            assert filename,(name,obj['id'],'unidentified collider asset')
            rel='platinum/'+filename[2:] if filename.startswith('~/') else filename
            asset=root/'Marble Blast Platinum'/rel
            assert asset.exists(),str(asset)
            assets.append(dict(object_id=obj['id'],file=str(asset.relative_to(root)).replace('\\','/'),sha256=sha(asset)))
        info['runtime_collision_assets']=assets
        atomic_json(mesh.path.with_suffix('.json'),info)
        coverage=info['source_object_coverage']
        assert all(len(x['runtime_matches'])==1 for x in coverage),(name,'ambiguous or missing mission object')
        assert all(x['collision_polygons']>0 for x in coverage if x['class_name'] in
                   ('InteriorInstance','TSStatic') or 'glass' in x['datablock'].lower())
        view=json.loads((args.directory/'viewer'/(name+'.viewer.json')).read_text())
        assert view['mesh_sha256']==info['mesh_sha256']
        assert view['displayed_triangles']==len(mesh.triangles)
        assert view['geometry_sha256']==hashlib.sha256(mesh.triangles.astype('<f8').tobytes()).hexdigest()
        browser=json.loads((args.directory/'viewer/browser_checks.json').read_text())
        br=next(x for x in browser['maps'] if x['map']==name)
        assert br['hash']==info['mesh_sha256'] and br['triangles']==len(mesh.triangles) and not br['javascript_errors']
        assert br['rendered_triangles']==len(mesh.triangles) and br['rendered_geometry_sha256']==view['geometry_sha256']
        centers=mesh.triangles.mean(axis=1)
        ids=np.linspace(0,len(mesh.triangles)-1,100,dtype=int)
        latencies=[]
        for i in ids:
            start=time.perf_counter()
            mesh.query_aabb(centers[i]-10,centers[i]+10)
            latencies.append((time.perf_counter()-start)*1000)
        validation=json.loads(mesh.path.with_name(name+'.validation.json').read_text())
        assert validation['mesh_sha256']==info['mesh_sha256']
        sweep=validation['sweeps']
        assert not sweep['counts'].get('hit_miss',0) and not sweep['counts'].get('center',0)
        assert not sweep['counts'].get('contact_point_unrepresented',0) and not sweep['counts'].get('material',0)
        assert all(x['exported_polygons']>=x['hull_referenced_surfaces'] for x in info['interior_hull_surface_coverage'])
        rows.append(dict(map=name,triangles=len(mesh.triangles),all_raw_vertices_preserved=True,
                         polygon_area_max_error_u2=area_error,source_objects=len(coverage),
                         native_sweeps=sweep['sweeps'],max_sweep_center_error_u=sweep['max_center_error_u'],
                         physical=validation['physics']['contacts'],valid_physical=validation['physics']['accepted'],
                         sweep_exceptions={k:v for k,v in sweep['counts'].items() if k not in ('hit','miss')},
                         memory_query_arrays_bytes=sum(x.nbytes for x in (mesh.triangles,mesh.source_triangles,mesh.normals,mesh.planes,
                              mesh.lo,mesh.hi,mesh.object_ids,mesh.material_ids,mesh.surface_keys,mesh.boundary_edges)),
                         compressed_npz_bytes=mesh.path.stat().st_size,aabb_p95_ms=float(np.percentile(latencies,95)),
                         max_native_plane_projection_u=mesh.max_plane_projection_u,
                         max_native_contact_surface_distance_u=sweep['max_native_contact_surface_distance_u'],
                         native_negative_hits=sweep['counts'].get('miss',0)))
        print(json.dumps(rows[-1]),flush=True)
    atomic_json(args.directory/'verification.json',dict(maps=rows,triangles=sum(r['triangles'] for r in rows),
                  native_sweeps=sum(r['native_sweeps'] for r in rows),physical_contacts=sum(r['physical'] for r in rows),
                  valid_physical=sum(r['valid_physical'] for r in rows),training_started=False))


if __name__=='__main__':
    main()
