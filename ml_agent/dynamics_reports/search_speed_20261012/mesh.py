"""Queries on the same serialized triangles displayed by the review viewer.

No walkability filtering, height raster, or per-map rules. Coordinates use game
world units and z-up; normals and material properties come from engine collision.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    # OneDrive/Windows scanners can temporarily hold the destination open.
    # Preserve atomic replacement and fail explicitly if the lock persists.
    for attempt in range(12):
        try:
            os.replace(str(tmp), str(path));break
        except PermissionError:
            if attempt==11:raise
            time.sleep(.05)


def convert_snapshot(raw, output, metadata=None):
    """Convert engine polygons to triangles without discarding orientation/classes."""
    raw, output = Path(raw), Path(output)
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if records[0].get('schema') != 'engine_collision_v1' or records[-1].get('kind') != 'complete':
        raise ValueError('Incomplete or unsupported engine snapshot')
    objects = [r for r in records if r['kind'] == 'object']
    polygons = [r for r in records if r['kind'] == 'polygon']
    if len(objects) != records[-1]['objects'] or len(polygons) != records[-1]['polygons']:
        raise ValueError('Engine completion counts disagree')
    mats, triangles, normals, material_ids, object_ids, surface_keys = [], [], [], [], [], []
    planes, boundary_edges = [], []
    seen, duplicate, degenerate = set(), 0, 0
    for p in polygons:
        v = np.asarray(p['vertices'], dtype=np.float64)
        n = np.asarray(p['normal'], dtype=np.float64)
        if not np.isfinite(v).all() or not np.isfinite(n).all():
            raise ValueError('Nonfinite collision geometry')
        if len(v) < 3:
            raise ValueError('Unsupported collision polygon with fewer than three vertices')
        key = (p['object_id'], p['surface_key'], p['material_index'],
               tuple(sorted(tuple(x) for x in v)), tuple(n))
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        if np.linalg.norm(n) < 1e-8:
            raise ValueError('Zero collision normal')
        n /= np.linalg.norm(n)
        if np.max(np.abs(v @ n + float(p['plane_d']) / np.linalg.norm(p['normal']))) > 0.01:
            raise ValueError('Nonplanar exported polygon')
        mat = dict(name=p['material'], friction=p['friction'], restitution=p['restitution'],
                   force=p['force'], resolved=p['properties_resolved'])
        if mat not in mats:
            mats.append(mat)
        mi = mats.index(mat)
        for j in range(1, len(v)-1):
            t = np.stack([v[0], v[j], v[j+1]])
            cross = np.cross(t[1]-t[0], t[2]-t[0])
            if np.linalg.norm(cross) < 1e-9:
                degenerate += 1
                # Native sphere collision also tests polygon edges/corners.
                # Retain line/point fans; zero area does not prove no contact.
            if cross @ n < 0:
                t = t[[0, 2, 1]]
                edges = [j == len(v)-2, True, j == 1]
            else:
                edges = [j == 1, True, j == len(v)-2]
            triangles.append(t)
            normals.append(n.copy())
            material_ids.append(mi)
            object_ids.append(p['object_id'])
            surface_keys.append(p['surface_key'])
            planes.append(list(p['normal'])+[p['plane_d']])
            boundary_edges.append(edges)
    if not triangles:
        raise ValueError('No collision triangles')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, triangles=np.asarray(triangles, dtype=np.float32),
                        normals=np.asarray(normals, dtype=np.float32),
                        material_ids=np.asarray(material_ids, dtype=np.int32),
                        object_ids=np.asarray(object_ids, dtype=np.int32),
                        surface_keys=np.asarray(surface_keys, dtype=np.uint32),
                        planes=np.asarray(planes, dtype=np.float32),
                        boundary_edges=np.asarray(boundary_edges, dtype=bool))
    os.replace(str(tmp), str(output))
    info = dict(schema='collision_map_v1', coordinate_frame='world_z_up',
                collision_source='engine_PLC_Collision_client', engine_header=records[0],
                raw_snapshot_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                mesh_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                raw_polygons=len(polygons), duplicate_polygons_removed=duplicate,
                degenerate_fan_triangles_retained=degenerate, triangles=len(triangles),
                objects=objects, materials=mats, **(metadata or {}))
    atomic_json(output.with_suffix('.json'), info)
    return info


class CollisionMap:
    def __init__(self, path):
        self.path = Path(path)
        self.info = json.loads(self.path.with_suffix('.json').read_text())
        if self.info['schema'] != 'collision_map_v1':
            raise ValueError('Unsupported collision map')
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.info['mesh_sha256']:
            raise ValueError('Collision mesh hash mismatch')
        with np.load(self.path, allow_pickle=False) as d:
            self.source_triangles = d['triangles'].astype(np.float64)
            self.normals = d['normals'].astype(np.float64)
            self.material_ids = d['material_ids']
            self.object_ids = d['object_ids']
            self.surface_keys = d['surface_keys']
            self.planes = d['planes'].astype(np.float64)
            self.boundary_edges = d['boundary_edges']
        # Native collision uses the stored plane, not a new plane through
        # rounded world vertices. Project faces onto that authoritative plane;
        # retain the exact source coordinates for edge/corner collision.
        n=self.planes[:,:3]
        nn=np.einsum('ij,ij->i',n,n)
        signed=np.einsum('ikj,ij->ik',self.source_triangles,n)+self.planes[:,3,None]
        self.triangles=self.source_triangles-signed[:,:,None]*n[:,None,:]/nn[:,None,None]
        self.max_plane_projection_u=float(np.max(np.abs(signed)/np.sqrt(nn)[:,None]))
        self.lo = np.minimum(self.triangles.min(axis=1),self.source_triangles.min(axis=1))
        self.hi = np.maximum(self.triangles.max(axis=1),self.source_triangles.max(axis=1))
        self.bounds = np.stack([self.lo.min(axis=0), self.hi.max(axis=0)])
        self.objects = {r['id']: r for r in self.info['objects']}

    def query_aabb(self, lo, hi):
        """All intersecting triangle bounds; conservative, never a nearest-K crop."""
        lo, hi = np.asarray(lo), np.asarray(hi)
        return np.flatnonzero(np.all(self.hi >= lo, axis=1) & np.all(self.lo <= hi, axis=1))

    def features(self, indices, origin=(0, 0, 0)):
        """Detailed model input: relative vertices, normals, material values, IDs."""
        indices = np.asarray(indices, dtype=np.int64)
        props = np.asarray([[m['friction'], m['restitution'], m['force']] for m in self.info['materials']])
        planes = self.planes[indices].copy()
        planes[:,3] += planes[:,:3] @ np.asarray(origin)
        return dict(vertices=self.triangles[indices]-np.asarray(origin), normals=self.normals[indices],
                    materials=props[self.material_ids[indices]], object_ids=self.object_ids[indices],
                    triangle_ids=indices, material_ids=self.material_ids[indices],
                    planes=planes, boundary_edges=self.boundary_edges[indices],
                    surface_keys=self.surface_keys[indices])

    def raycast(self, start, end, cull_backfaces=True):
        """Nearest exact triangle intersection of a finite segment, or None."""
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        direction = end-start
        ids = self.query_aabb(np.minimum(start, end)-1e-7, np.maximum(start, end)+1e-7)
        if not len(ids):
            return None
        tri=self.triangles[ids]
        planes=self.planes[ids]
        det=planes[:,:3] @ direction
        valid=det < -1e-10 if cull_backfaces else np.abs(det)>1e-10
        t=np.divide(-planes[:,3]-planes[:,:3] @ start,det,out=np.zeros(len(ids)),where=valid)
        points=start+direction*t[:,None]
        ab,ac=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        cross=np.cross(ab,ac)
        nn=np.einsum('ij,ij->i',cross,cross)
        rel=points-tri[:,0]
        u=np.divide(np.einsum('ij,ij->i',np.cross(rel,ac),cross),nn,out=np.zeros(len(ids)),where=nn>1e-24)
        v=np.divide(np.einsum('ij,ij->i',np.cross(ab,rel),cross),nn,out=np.zeros(len(ids)),where=nn>1e-24)
        valid &= nn>1e-24
        valid &= (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7) & (t >= -1e-7) & (t <= 1+1e-7)
        if not valid.any():
            return None
        i = int(np.argmin(np.where(valid, t, np.inf)))
        idx = int(ids[i])
        return dict(triangle_id=idx, object_id=int(self.object_ids[idx]), t=float(t[i]),
                    point=(start+t[i]*direction).tolist(), normal=self.normals[idx].tolist(),
                    material=self.info['materials'][int(self.material_ids[idx])])

    def closest_points(self, point, indices):
        """Closest points on faces/edges/vertices, without raster precision loss."""
        p = np.asarray(point, dtype=np.float64)
        tri = self.triangles[np.asarray(indices)]
        if not len(tri):
            return np.empty((0, 3))
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac = b-a, c-a
        n = np.cross(ab, ac)
        nn = np.einsum('ij,ij->i', n, n)
        height = np.divide(np.einsum('ij,ij->i', p-a,n),nn,out=np.zeros(len(tri)),where=nn>0)
        proj = p-n*height[:,None]
        rel = proj-a
        u = np.divide(np.einsum('ij,ij->i',np.cross(rel,ac),n),nn,out=np.zeros(len(tri)),where=nn>0)
        v = np.divide(np.einsum('ij,ij->i',np.cross(ab,rel),n),nn,out=np.zeros(len(tri)),where=nn>0)
        inside = (nn>0) & (u >= 0) & (v >= 0) & (u+v <= 1)
        options = []
        for x, y in [(a,b),(b,c),(c,a)]:
            e = y-x
            ee = np.einsum('ij,ij->i', e,e)
            s = np.divide(np.einsum('ij,ij->i',p-x,e),ee,out=np.zeros(len(tri)),where=ee>0).clip(0,1)
            options.append(x+e*s[:,None])
        options = np.stack(options, axis=1)
        k = np.argmin(((options-p)**2).sum(axis=2), axis=1)
        closest = options[np.arange(len(tri)), k]
        closest[inside] = proj[inside]
        return closest

    def sweep_sphere(self, start, end, radius, native_dt_s=None):
        """First continuous sphere/triangle contact (faces, edges and vertices).

        Starts must be clear of geometry. One-sided approach follows marble
        collision planes. This geometric predictor does not include game forces,
        moving objects. Optional native_dt_s reproduces the engine's float edge
        discriminants and 0.1 ms exit-root tolerance for dynamics comparison;
        the default geometric query and reviewed geometry remain unchanged.
        """
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        velocity = end-start
        if radius <= 0:
            raise ValueError('Radius must be positive')
        speed2 = float(velocity @ velocity)
        if speed2 < 1e-18:
            return None
        ids = self.query_aabb(np.minimum(start, end)-radius, np.maximum(start, end)+radius)
        if not len(ids):
            return None
        tri, normals = self.triangles[ids], self.planes[ids,:3]
        approach = normals @ velocity
        valid = approach < -0.001*np.sqrt(speed2)
        distances = normals @ start + self.planes[ids,3]
        times = np.divide(radius-distances, approach, out=np.full(len(ids), np.inf), where=valid)
        safe_times = np.where(valid, times, 0.)
        centers = start+velocity*safe_times[:,None]
        points = centers-normals*radius
        ab, ac = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
        # Replace invalid infinities before barycentric arithmetic.
        rel = np.where(valid[:,None], points-tri[:,0], 0.)
        cross = np.cross(ab,ac)
        nn = np.einsum('ij,ij->i',cross,cross)
        u = np.divide(np.einsum('ij,ij->i',np.cross(rel,ac),cross),nn,out=np.zeros(len(ids)),where=nn>0)
        v = np.divide(np.einsum('ij,ij->i',np.cross(ab,rel),cross),nn,out=np.zeros(len(ids)),where=nn>0)
        face = valid & (nn>0) & (times > 0) & (times < 1) & (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
        best = np.where(face, times, np.inf)
        closest = np.where(face[:,None], points, 0.)
        edge_tri=getattr(self,'source_triangles',self.triangles)[ids]
        for j in range(3):
            a, b = edge_tri[:,j], edge_tri[:,(j+1)%3]
            edge = b-a
            length2 = np.einsum('ij,ij->i', edge, edge)
            pv = start-a
            boundary = self.boundary_edges[ids,j] & (length2>0)
            ve = np.divide(edge @ velocity,length2,out=np.zeros(len(ids)),where=length2>0)
            pe = np.divide(np.einsum('ij,ij->i',pv,edge),length2,out=np.zeros(len(ids)),where=length2>0)
            vr = velocity-edge*ve[:,None]
            pr = pv-edge*pe[:,None]
            qa = np.einsum('ij,ij->i', vr, vr)
            qb = 2*np.einsum('ij,ij->i', pr, vr)
            qc = np.einsum('ij,ij->i', pr, pr)-radius*radius
            disc = qb*qb-4*qa*qc
            root=np.sqrt(np.maximum(0,disc))
            if native_dt_s is not None:
                # Native equations use cross products, including edge-length
                # scaling, and round the discriminant and square root to F32.
                scaled=np.maximum(0,disc)*length2*length2/(native_dt_s**2)
                root=np.divide(np.sqrt(scaled.astype(np.float32)).astype(np.float32).astype(float)*native_dt_s,
                               length2,out=np.zeros(len(ids)),where=length2>0)
            t = np.divide(-qb-root, 2*qa,
                          out=np.full(len(ids), np.inf), where=qa>1e-18)
            exit_t=np.divide(-qb+root,2*qa,out=np.full(len(ids),-np.inf),where=qa>1e-18)
            along = pe+np.where(np.isfinite(t),t,0.)*ve
            keep = valid & boundary & (disc >= 0) & (t > 0) & (t < 1) & (t < best) & (along >= 0) & (along <= 1)
            if native_dt_s is not None:keep &= exit_t*native_dt_s>.0001
            best[keep] = t[keep]
            closest[keep] = (a+edge*along[:,None])[keep]
            qb = 2*(pv @ velocity)
            qc = np.einsum('ij,ij->i', pv,pv)-radius*radius
            disc = qb*qb-4*speed2*qc
            root=np.sqrt(np.maximum(0,disc))
            if native_dt_s is not None:
                root=np.sqrt((np.maximum(0,disc)/(native_dt_s**2)).astype(np.float32)).astype(np.float32).astype(float)*native_dt_s
            t = (-qb-root)/(2*speed2)
            exit_t=(-qb+root)/(2*speed2)
            keep = valid & (disc >= 0) & (t > 0) & (t < 1) & (t < best)
            if native_dt_s is not None:keep &= exit_t*native_dt_s>.0001
            best[keep] = t[keep]
            closest[keep] = a[keep]
        if not np.isfinite(best).any():
            return None
        i = int(np.argmin(best))
        idx = int(ids[i])
        center = start+best[i]*velocity
        n = center-closest[i]
        n /= np.linalg.norm(n)
        return dict(triangle_id=idx, object_id=int(self.object_ids[idx]), t=float(best[i]),
                    center=center.tolist(), point=closest[i].tolist(), normal=n.tolist(),
                    material=self.info['materials'][int(self.material_ids[idx])])

    def sphere_contacts(self, center, radius):
        """Geometric sphere overlaps; returns potential contacts, not dynamics."""
        center = np.asarray(center, dtype=np.float64)
        if radius <= 0:
            raise ValueError('Radius must be positive')
        ids = self.query_aabb(center-radius, center+radius)
        pts = self.closest_points(center, ids)
        distances = np.linalg.norm(pts-center, axis=1)
        keep = distances <= radius+1e-7
        return dict(triangle_ids=ids[keep], closest_points=pts[keep], distances=distances[keep],
                    features=self.features(ids[keep], center))
