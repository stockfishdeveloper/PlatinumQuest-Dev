"""Fine (0.1 u) height raster for the landing predictor (2026-09-25, HANDOFF 28.40).

The observation crop and walk grid stay at 0.5 / 1.0 u; the landing predictor needs the lip and the far
floor to a few centimetres, so it reads a 0.1 u raster of the same .dif floor polygons, built once per
map with generate_terrain_map.build_height_stack and cached in terrain_maps/terrain_fine_<map>.npz.

    python -m nav.fine_terrain KingOfTheMarble_Hunt      # build (or rebuild) the fine raster
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINE_RES = 0.1
_CACHE = {}


def fine_path(map_name):
    return os.path.join(HERE, 'terrain_maps', f'terrain_fine_{map_name}.npz')


def build(map_name, k_max=4):
    sys.path.insert(0, HERE)
    import hxDif
    from generate_terrain_map import resolve_mission, parse_interiors, parse_in_bounds_z, transform_surfaces, build_height_stack
    from generate_map_topdown import extract_surfaces
    mis = resolve_mission(map_name)
    z_min, z_max = parse_in_bounds_z(mis)
    surfaces = []
    for dif_path, pos, rot, scl in parse_interiors(mis):
        surfaces += transform_surfaces(extract_surfaces(hxDif.Dif.Load(dif_path)), pos, rot, scl)
    xs, ys, heights, stats = build_height_stack(surfaces, FINE_RES, z_min, z_max, k_max)
    out = fine_path(map_name)
    np.savez_compressed(out, xs=xs.astype(np.float32), ys=ys.astype(np.float32), heights=heights.astype(np.float32),
                        grid_res=np.float32(FINE_RES), map_name=np.array(map_name))
    print(f'wrote {out}: {len(xs)}x{len(ys)} cells at {FINE_RES} u, {stats["walkable"]:,} with floor')
    return out


class FineTerrain:
    """heights_at(xs, ys) -> (K, N) on the 0.1 u raster; NaN = no floor."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.xs = d['xs']; self.ys = d['ys']; self.heights = d['heights']
        self.res = float(d['grid_res']); self.x0 = float(self.xs[0]); self.y0 = float(self.ys[0])
        self.K, self.H, self.W = self.heights.shape

    def heights_at(self, xs, ys):
        xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
        i = np.rint((xs - self.x0) / self.res).astype(np.int64); j = np.rint((ys - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.W) & (j >= 0) & (j < self.H)
        out = np.full((self.K, len(xs)), np.nan, dtype=np.float32)
        if ok.any():
            out[:, ok] = self.heights[:, j[ok], i[ok]]
        return out


def get(map_name):
    """The fine raster for a map (built on first use), or None if the map cannot be resolved."""
    if map_name in _CACHE:
        return _CACHE[map_name]
    p = fine_path(map_name)
    try:
        if not os.path.exists(p):
            build(map_name)
        _CACHE[map_name] = FineTerrain(p)
    except Exception as e:      # keep the caller usable on the coarse map
        print(f'fine terrain unavailable for {map_name}: {e}', flush=True)
        _CACHE[map_name] = None
    return _CACHE[map_name]


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'KingOfTheMarble_Hunt')
