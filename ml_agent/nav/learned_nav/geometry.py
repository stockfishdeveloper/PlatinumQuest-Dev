"""Stage 3 geometry contract (KOTMJUMP_NAVIGATION_DESIGN.md section 3, stage 1): one artifact per map.

    python -m nav.learned_nav.geometry export Sprawl_Hunt_phys [more maps]     # -> datasets/learned_nav/geometry/<map>.npz

Contents, all in world coordinates:
* the collision surfaces of every InteriorInstance (a DIF's single detail level; verified per map in the game),
  fan-triangulated, with their normals and material (texture name -> friction / restitution / force, data/init.cs);
* a layered height raster (RES = 0.2 u, up to K levels per cell, top first) of the floor-like faces (normal z > 0.3),
  with the exact face normal of every level (slopes come from normals, never from height differences);
* exact edge segments from the mesh: every edge of a floor face that no other floor face shares, classified by what
  lies just beyond it: VOID (nothing below), DROP (a lower floor, with its height), WALL (a wall or higher floor).
  Their precision is the mesh's, not the grid's, which is where lips are decided.
Queries: heights_at / floor_below / normal_below on the raster, and edge_rays (distance along 2-D rays to the first
edge segment of the marble's floor level, with the segment's type and drop).
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
GEOM_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'geometry')
PLAT = os.path.join(HERE, '..', 'Marble Blast Platinum', 'platinum')
RES = 0.2
K = 4
FLOOR_NZ = 0.3              # faces steeper than ~72 deg are walls for the raster and the edge set
LEVEL_MERGE = 0.05          # samples closer than this in one cell are the same surface (shared triangle edges)
VOID, DROP, WALL, SEAM = 0, 1, 2, 3     # SEAM edges (floor continues beyond) are dropped at export
SCHEMA = 'geometry_v1'


def _sha(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]


def material_table():
    """texture name (lower case) -> (material, friction, restitution, force) from data/init.cs."""
    t = open(os.path.join(PLAT, 'data', 'init.cs'), errors='replace').read()
    props = {}
    for m in re.finditer(r'new MaterialProperty\((\w+)\)\s*\{(.*?)\};', t, re.S):
        body = m.group(2)
        g = lambda k, d: float(re.search(r'%s\s*=\s*([-\d.]+)' % k, body).group(1)) if re.search(r'%s\s*=\s*([-\d.]+)' % k, body) else d
        props[m.group(1)] = (g('friction', 1.0), g('restitution', 1.0), g('force', 0.0))
    out = {}
    for m in re.finditer(r'addMaterialMapping\("([^"]*)",\s*(\w+)\)', t):
        f, r, fo = props.get(m.group(2), (1.0, 1.0, 0.0))
        out[m.group(1).lower()] = (m.group(2), f, r, fo)
    return out


def _collect(map_name):
    import hxDif
    from generate_terrain_map import resolve_mission, parse_interiors, transform_surfaces
    from generate_map_topdown import extract_surfaces
    mis = resolve_mission(map_name)
    tris, norms, mats, hashes = [], [], [], {os.path.basename(mis): _sha(mis)}
    mat_names = []
    for dif, pos, rot, scl in parse_interiors(mis):
        d = hxDif.Dif.Load(dif)
        hashes[os.path.basename(dif)] = _sha(dif)
        assert len(d.interiors) == 1, f'{dif}: {len(d.interiors)} detail levels (not handled)'
        it = d.interiors[0]
        names = [str(n).lower() for n in it.materialList]
        surf_mat = [names[s.textureIndex] if 0 <= s.textureIndex < len(names) else '' for s in it.surfaces]
        polys = transform_surfaces(extract_surfaces(d), pos, rot, scl)
        assert len(polys) == len(surf_mat)
        for (verts, n), mname in zip(polys, surf_mat):
            v = np.asarray(verts, dtype=np.float64)
            if len(v) < 3:
                continue
            base = mname.split('/')[-1]
            if base not in mat_names:
                mat_names.append(base)
            mi = mat_names.index(base)
            for i in range(1, len(v) - 1):
                tri = np.stack([v[0], v[i], v[i + 1]])
                if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-9:
                    continue
                tris.append(tri); norms.append(n); mats.append(mi)
    return mis, np.asarray(tris), np.asarray(norms, dtype=np.float64), np.asarray(mats, dtype=np.int32), mat_names, hashes


def _rasterize(tris, norms, floor):
    ft = tris[floor]; fn = norms[floor]
    x0 = math.floor(ft[:, :, 0].min() / RES) * RES - 1.0; x1 = ft[:, :, 0].max() + 1.0
    y0 = math.floor(ft[:, :, 1].min() / RES) * RES - 1.0; y1 = ft[:, :, 1].max() + 1.0
    xs = np.arange(x0, x1 + 1e-9, RES); ys = np.arange(y0, y1 + 1e-9, RES)
    W, H = len(xs), len(ys)
    cells, zs, tids = [], [], []
    fidx = np.nonzero(floor)[0]
    for k, (t, n) in enumerate(zip(ft, fn)):
        i0 = max(0, int(math.ceil((t[:, 0].min() - x0) / RES))); i1 = min(W - 1, int(math.floor((t[:, 0].max() - x0) / RES)))
        j0 = max(0, int(math.ceil((t[:, 1].min() - y0) / RES))); j1 = min(H - 1, int(math.floor((t[:, 1].max() - y0) / RES)))
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        px = x0 + ii.ravel() * RES; py = y0 + jj.ravel() * RES
        a, b, c = t[0], t[1], t[2]
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / det
        l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det
        l3 = 1 - l1 - l2
        e = -1e-9
        inside = (l1 >= e) & (l2 >= e) & (l3 >= e)
        if not inside.any():
            continue
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        cells.append(jj.ravel()[inside] * W + ii.ravel()[inside]); zs.append(z[inside]); tids.append(np.full(inside.sum(), fidx[k]))
    cells = np.concatenate(cells); zs = np.concatenate(zs); tids = np.concatenate(tids)
    order = np.lexsort((-zs, cells))
    cells, zs, tids = cells[order], zs[order], tids[order]
    heights = np.full((K, H * W), np.nan, dtype=np.float32)
    ntri = np.full((K, H * W), -1, dtype=np.int32)
    start = np.r_[0, np.nonzero(np.diff(cells))[0] + 1]
    end = np.r_[start[1:], len(cells)]
    for s, e_ in zip(start, end):
        c = cells[s]; lev = 0; last = None
        for q in range(s, e_):
            if last is not None and last - zs[q] < LEVEL_MERGE:
                continue
            if lev >= K:
                break
            heights[lev, c] = zs[q]; ntri[lev, c] = tids[q]; last = zs[q]; lev += 1
    return xs.astype(np.float64), ys.astype(np.float64), heights.reshape(K, H, W), ntri.reshape(K, H, W)


def _edges(tris, norms, floor, xs, ys, heights):
    """Floor-face edges no other floor face shares, classified by what lies 0.3 u beyond them."""
    key = lambda p: (round(p[0], 3), round(p[1], 3), round(p[2], 3))
    count = {}
    owner = {}
    fidx = np.nonzero(floor)[0]
    for t in fidx:
        tri = tris[t]
        for i in range(3):
            a, b = key(tri[i]), key(tri[(i + 1) % 3])
            e = (a, b) if a < b else (b, a)
            count[e] = count.get(e, 0) + 1
            owner[e] = t
    segs = []
    for e, c in count.items():
        if c != 1:
            continue
        t = owner[e]
        a = np.asarray(e[0]); b = np.asarray(e[1])
        d = b[:2] - a[:2]
        L = float(np.hypot(*d))
        if L < 1e-4:
            continue
        cen = tris[t].mean(axis=0)
        n2 = np.array([d[1], -d[0]]) / L
        mid = (a + b) / 2.0
        if np.dot(cen[:2] - mid[:2], n2) > 0:
            n2 = -n2                                              # outward: away from the face
        segs.append(np.r_[a, b, n2])
    segs = np.asarray(segs)
    if not len(segs):
        return np.zeros((0, 11))
    mid = (segs[:, 0:3] + segs[:, 3:6]) / 2.0
    probe = mid[:, :2] + segs[:, 6:8] * 0.3
    zlip = mid[:, 2]
    H = _heights_at(xs, ys, heights, probe[:, 0], probe[:, 1])        # (K, n)
    typ = np.full(len(segs), VOID, dtype=np.float64); drop = np.full(len(segs), np.inf)
    for i in range(len(segs)):
        lv = H[:, i]; lv = lv[np.isfinite(lv)]
        if not len(lv):
            continue
        near = lv[np.abs(lv - zlip[i]) <= 0.3]
        if len(near):
            typ[i] = SEAM                                         # the floor continues: a seam between faces whose
            continue                                              # vertices do not match (T-junctions), not an edge
        above = lv[(lv > zlip[i] + 0.3) & (lv < zlip[i] + 3.0)]
        below = lv[lv < zlip[i] - 0.3]
        if len(above):
            typ[i] = WALL; drop[i] = -(above.min() - zlip[i])
        elif len(below):
            typ[i] = DROP; drop[i] = zlip[i] - below.max()
    keep = typ != SEAM
    return np.c_[segs, typ, drop, zlip][keep]


def _heights_at(xs, ys, heights, qx, qy):
    qx = np.asarray(qx, dtype=np.float64); qy = np.asarray(qy, dtype=np.float64)
    i = np.rint((qx - xs[0]) / RES).astype(np.int64); j = np.rint((qy - ys[0]) / RES).astype(np.int64)
    ok = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
    out = np.full((heights.shape[0], len(qx)), np.nan, dtype=np.float32)
    if ok.any():
        out[:, ok] = heights[:, j[ok], i[ok]]
    return out


def export(map_name):
    t0 = time.time()
    mis, tris, norms, mats, mat_names, hashes = _collect(map_name)
    floor = norms[:, 2] > FLOOR_NZ
    xs, ys, heights, ntri = _rasterize(tris, norms, floor)
    edges = _edges(tris, norms, floor, xs, ys, heights)
    mt = material_table()
    mprops = np.array([mt.get(m, ('DefaultMaterial', 1.0, 1.0, 0.0))[1:] for m in mat_names], dtype=np.float32)
    os.makedirs(GEOM_DIR, exist_ok=True)
    out = os.path.join(GEOM_DIR, f'{map_name}.npz')
    np.savez_compressed(out, tris=tris.astype(np.float32), norms=norms.astype(np.float32), mats=mats,
                        mat_names=np.array(mat_names), mat_props=mprops, xs=xs, ys=ys, heights=heights, ntri=ntri,
                        edges=edges.astype(np.float32), res=np.float32(RES))
    manifest = {'schema': SCHEMA, 'map': map_name, 'mission': os.path.relpath(mis, HERE), 'hashes': hashes,
                'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'triangles': int(len(tris)), 'floor_triangles': int(floor.sum()),
                'raster': [len(xs), len(ys)], 'res': RES, 'levels': K,
                'floor_cells': int(np.isfinite(heights[0]).sum()), 'multi_level_cells': int(np.isfinite(heights[1]).sum()),
                'edges': {'void': int((edges[:, 8] == VOID).sum()), 'drop': int((edges[:, 8] == DROP).sum()),
                          'wall': int((edges[:, 8] == WALL).sum())},
                'edge_length_u': round(float(np.hypot(edges[:, 3] - edges[:, 0], edges[:, 4] - edges[:, 1]).sum()), 1),
                'materials': {m: mt.get(m, ('DefaultMaterial',))[0] for m in mat_names},
                'special_materials': sorted({mt[m][0] for m in mat_names if m in mt and mt[m][0] != 'DefaultMaterial'}),
                'seconds': round(time.time() - t0, 1)}
    json.dump(manifest, open(out.replace('.npz', '.json'), 'w'), indent=1)
    return manifest


class Geometry:
    """A loaded geometry artifact with the queries the recorder, the verification and the model use."""

    def __init__(self, map_name):
        d = np.load(os.path.join(GEOM_DIR, f'{map_name}.npz'), allow_pickle=False)
        self.map = map_name
        self.xs = d['xs']; self.ys = d['ys']; self.heights = d['heights']; self.ntri = d['ntri']
        self.norms = d['norms']; self.tris = d['tris']; self.edges = d['edges'].astype(np.float64)
        self.mats = d['mats']; self.mat_props = d['mat_props']; self.mat_names = list(d['mat_names'])
        self.res = float(d['res'])
        e = self.edges
        self._emin = np.minimum(e[:, 0:2], e[:, 3:5]); self._emax = np.maximum(e[:, 0:2], e[:, 3:5])
        # the levels of each raster cell side by side (cell-major): one row gather per query instead of K strided ones
        # (the planner makes ~10^5 queries a decision; same values as _heights_at)
        self._hT = np.ascontiguousarray(self.heights.reshape(self.heights.shape[0], -1).T)

    def heights_at(self, qx, qy):
        qx = np.asarray(qx, dtype=np.float64); qy = np.asarray(qy, dtype=np.float64)
        i = np.rint((qx - self.xs[0]) / RES).astype(np.int64); j = np.rint((qy - self.ys[0]) / RES).astype(np.int64)
        nx = len(self.xs)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < len(self.ys))
        if ok.all():
            return np.ascontiguousarray(self._hT[j * nx + i].T)          # (K, N) contiguous, as callers reduce over K
        out = np.full((self.heights.shape[0], len(qx)), np.nan, dtype=np.float32)
        if ok.any():
            out[:, ok] = self._hT[j[ok] * nx + i[ok]].T
        return out

    def _level_below(self, x, y, z, tol=0.3):
        i = int(round((x - self.xs[0]) / self.res)); j = int(round((y - self.ys[0]) / self.res))
        if not (0 <= i < len(self.xs) and 0 <= j < len(self.ys)):
            return None, None
        col = self.heights[:, j, i]
        best = None
        for k in range(self.heights.shape[0]):
            h = col[k]
            if np.isfinite(h) and h <= z + tol and (best is None or h > col[best]):
                best = k
        return (best, (j, i)) if best is not None else (None, None)

    def floor_below(self, x, y, z, tol=0.3):
        k, ji = self._level_below(x, y, z, tol)
        return None if k is None else float(self.heights[k, ji[0], ji[1]])

    def normal_below(self, x, y, z, tol=0.3):
        k, ji = self._level_below(x, y, z, tol)
        if k is None:
            return None
        return self.norms[self.ntri[k, ji[0], ji[1]]].astype(np.float64)

    def material_below(self, x, y, z, tol=0.3):
        k, ji = self._level_below(x, y, z, tol)
        if k is None:
            return None
        return self.mat_names[self.mats[self.ntri[k, ji[0], ji[1]]]]

    RAY_STEP = 0.1
    FOLLOW_TOL = 0.25          # u of height change per 0.1 u step that still counts as the same floor (~68 deg)
    EDGE_Z_TOL = 0.35          # an edge belongs to the followed floor if its height is this close to it

    def follow(self, x, y, zfloor, angles, max_u=16.0):
        """The floor the marble is on, followed along each ray on the raster: (n_angles, n_steps) heights, NaN
        after the floor ends, and the distance where it ends (inf if it reaches max_u)."""
        ds = np.arange(1, int(round(max_u / self.RAY_STEP)) + 1) * self.RAY_STEP
        ca = np.cos(angles); sa = np.sin(angles)
        H = self.heights_at((x + np.outer(ca, ds)).ravel(), (y + np.outer(sa, ds)).ravel())
        H = H.reshape(self.heights.shape[0], len(angles), len(ds))
        prof = np.full((len(angles), len(ds)), np.nan); end = np.full(len(angles), np.inf)
        for k in range(len(angles)):
            zc = zfloor
            for s in range(len(ds)):
                col = H[:, k, s]; fin = np.isfinite(col)
                if fin.any():
                    j = int(np.argmin(np.where(fin, np.abs(col - zc), np.inf)))
                    if abs(col[j] - zc) <= self.FOLLOW_TOL:
                        zc = float(col[j]); prof[k, s] = zc
                        continue
                end[k] = ds[s]
                break
        return ds, prof, end

    def edge_rays(self, x, y, zfloor, angles, max_u=16.0):
        """For each world angle: (distance, type, drop) of the first exact edge segment of the marble's own floor
        (followed along the ray), or, where the followed floor ends without an exported edge, the raster end
        (type VOID/DROP from what lies below, distance to 0.1 u). (inf, -1, 0) if the floor reaches max_u."""
        angles = np.asarray(angles, dtype=np.float64)
        ds, prof, end = self.follow(x, y, zfloor, angles, max_u)
        e = self.edges
        near = ((self._emax[:, 0] >= x - max_u) & (self._emin[:, 0] <= x + max_u) & (self._emax[:, 1] >= y - max_u)
                & (self._emin[:, 1] <= y + max_u))
        s = e[near]
        out = np.zeros((len(angles), 3)); out[:, 0] = np.inf; out[:, 1] = -1
        ax, ay = s[:, 0] - x, s[:, 1] - y
        bx, by = s[:, 3] - s[:, 0], s[:, 4] - s[:, 1]
        for k, ang in enumerate(angles):
            dx, dy = math.cos(ang), math.sin(ang)
            found = False
            if len(s):
                den = dx * by - dy * bx
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = (ax * by - ay * bx) / den            # distance along the ray
                    u = (ax * dy - ay * dx) / den            # position along the segment
                ok = (np.abs(den) > 1e-12) & (t > 0) & (t <= min(max_u, end[k] + 0.3)) & (u >= -1e-6) & (u <= 1 + 1e-6)
                if ok.any():
                    idx = np.nonzero(ok)[0]
                    idx = idx[np.argsort(t[idx])]
                    for i in idx:
                        si = min(len(ds) - 1, max(0, int(t[i] / self.RAY_STEP) - 1))
                        zf = prof[k, si] if np.isfinite(prof[k, si]) else (prof[k, :si][np.isfinite(prof[k, :si])][-1:] if si else [])
                        zf = float(zf) if np.ndim(zf) == 0 else (float(zf[0]) if len(zf) else zfloor)
                        zseg = s[i, 2] + u[i] * (s[i, 5] - s[i, 2])
                        if abs(zseg - zf) <= self.EDGE_Z_TOL:
                            out[k] = (t[i], s[i, 8], s[i, 9] if np.isfinite(s[i, 9]) else 99.0)
                            found = True
                            break
            if not found and np.isfinite(end[k]):
                zl = prof[k][np.isfinite(prof[k])]
                zl = float(zl[-1]) if len(zl) else zfloor
                px, py = x + dx * (end[k] + 0.2), y + dy * (end[k] + 0.2)
                col = self.heights_at([px], [py])[:, 0]
                below = col[np.isfinite(col) & (col < zl - 0.3)]
                above = col[np.isfinite(col) & (col > zl + 0.3)]
                if len(above):
                    out[k] = (end[k], WALL, -(float(above.min()) - zl))
                elif len(below):
                    out[k] = (end[k], DROP, zl - float(below.max()))
                else:
                    out[k] = (end[k], VOID, 99.0)
        return out


def main():
    if len(sys.argv) < 3 or sys.argv[1] != 'export':
        print(__doc__)
        return 1
    for m in sys.argv[2:]:
        man = export(m)
        print(json.dumps({k: man[k] for k in ('map', 'triangles', 'floor_triangles', 'raster', 'floor_cells', 'multi_level_cells',
                                              'edges', 'edge_length_u', 'special_materials', 'seconds')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
