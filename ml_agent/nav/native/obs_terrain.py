"""Compiled terrain queries for the learned model's observation (obs_native.cpp; 2026-10-19, search speed-up toward
real-time play). native_queries(terrain) returns a wrapper around a TerrainGrid that answers crop / floor_z /
floor_height / edge_rays / gap_along in C++ and passes every other attribute to the terrain itself. The answers are
bit-identical to the NumPy methods (tools: scratchpad test_obs_native.py, 0 of 3,000 random states differ per query);
inputs whose NumPy types the port does not reproduce, and crops with an order-sensitive tie, go to NumPy.
Measured 10-19 (Cragmire, warm): crop 260 -> 40 us, edge_rays 360 -> 13-25 us, gap_along 110 -> 8 us.
The planner's FrozenNavigator uses it; training and live play keep the NumPy terrain."""
import ctypes
import os
from pathlib import Path

import numpy as np

import terrain_obs as TO
import nav.terrain as TR

ENABLED = True
DEFAULT_DLL = Path(os.environ.get('TEMP', '.')) / 'pq_checks' / 'marble_sim' / 'obs_native.dll'
_P = ctypes.c_void_p; _D = ctypes.c_double; _I = ctypes.c_int
_LIB = {}
_WEAK = (float, np.float32)        # a z of these types meets the float32 heights in float32 (NumPy 2 promotion)


def _ptr(a):
    return a.ctypes.data_as(_P)


def _lib(dll=None):
    path = str(dll or DEFAULT_DLL)
    lib = _LIB.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        lib.ob_abi.restype = _I
        if lib.ob_abi() != 1:
            raise RuntimeError('obs_native ABI mismatch: rebuild with nav/native/build.ps1')
        lib.ob_create.restype = _P
        lib.ob_create.argtypes = [_I, _I, _I, _D, _D, _D, _P, _D, _D, _D, _D, _I, _P, _I, _P, _I]
        lib.ob_destroy.restype = None; lib.ob_destroy.argtypes = [_P]
        lib.ob_floor_height.restype = _D; lib.ob_floor_height.argtypes = [_P, _D, _D, _D]
        lib.ob_crop.restype = _I; lib.ob_crop.argtypes = [_P, _D, _D, _D, _P, _P, _I, _P]
        lib.ob_edge_rays.restype = None; lib.ob_edge_rays.argtypes = [_P, _D, _D, _D, _P, _I, _P]
        lib.ob_gap_along.restype = None; lib.ob_gap_along.argtypes = [_P, _D, _D, _D, _D, _D, _I, _I, _P]
        _LIB[path] = lib
    return lib


class NativeTerrainQueries:
    """A TerrainGrid whose observation queries run in C++ (see the module docstring)."""

    def __init__(self, terrain, dll=None):
        t = terrain
        if not (isinstance(t.heights, np.ndarray) and t.heights.dtype == np.float32 and t.heights.ndim == 3 and t.K <= 64
                and type(t.x0) is float and type(t.y0) is float and type(t.res) is float
                and t._fine.shape == (TR.CROP_CELLS ** 2, 2) and TR.CROP_CHANNELS == 3):
            raise ValueError('terrain not supported by obs_native')
        lib = _lib(dll)
        d = self.__dict__
        d['_t'] = t; d['_lib'] = lib
        d['_heights'] = np.ascontiguousarray(t.heights)          # the C++ side reads this array in place
        d['_ray_s'] = np.ascontiguousarray(TO._RAY_S, dtype=np.float32)
        d['_ray_dirs'] = np.ascontiguousarray(TO._RAY_DIRS, dtype=np.float32)
        d['_fine'] = np.ascontiguousarray(t._fine, dtype=np.float32)
        d['_coarse'] = np.ascontiguousarray(t._coarse, dtype=np.float32)
        d['_rels'] = np.zeros(4 * TO.GEM_RAYS); d['_gap'] = np.zeros(3)
        d['_h'] = lib.ob_create(int(t.K), int(t.H), int(t.W), float(t.x0), float(t.y0), float(t.res), _ptr(d['_heights']),
                                float(TO.FOLLOW_TOL), float(TO.DZ_SCALE), float(TO.RAY_RANGE), float(TO.RAY_STEP), int(TO.RAY_STEPS),
                                _ptr(d['_ray_s']), int(TO.RAY_HEADINGS), _ptr(d['_ray_dirs']), int(TO.GEM_RAYS))

    def __del__(self):
        try:
            self.__dict__['_lib'].ob_destroy(self.__dict__['_h'])
        except Exception:
            pass

    def __getattr__(self, name):                 # everything else: the terrain itself
        return getattr(self.__dict__['_t'], name)

    def __reduce__(self):                        # travels as its terrain (parallel search workers rebuild their own)
        return native_queries, (self.__dict__['_t'],)

    @property
    def terrain(self):
        return self.__dict__['_t']

    def crop(self, x, y, z):
        d = self.__dict__
        if type(z) in _WEAK:
            out = np.empty(TR.CROP_SHAPE, dtype=np.float32)
            if not d['_lib'].ob_crop(d['_h'], float(x), float(y), float(z), _ptr(d['_fine']), _ptr(d['_coarse']), TR.CROP_CELLS ** 2, _ptr(out)):
                return out
        return d['_t'].crop(x, y, z)

    def floor_height(self, x, y, z):
        d = self.__dict__
        if type(z) in _WEAK:
            return d['_lib'].ob_floor_height(d['_h'], float(x), float(y), float(z))
        return d['_t'].floor_height(x, y, z)

    def floor_z(self, x, y, z):
        return self.floor_height(x, y, z)

    def edge_rays(self, x, y, z, gem_rels):
        d = self.__dict__
        if type(z) in _WEAK and len(gem_rels) <= TO.GEM_RAYS:
            rels = d['_rels']; rels[:] = 0.
            for i, (dx, dy, dz, present) in enumerate(gem_rels):
                if type(dx) not in (float, np.float64) or type(dy) not in (float, np.float64):
                    break
                rels[4 * i] = dx; rels[4 * i + 1] = dy; rels[4 * i + 3] = 1. if present else 0.
            else:
                out = np.empty(TO.EDGE_DIM, dtype=np.float32)
                d['_lib'].ob_edge_rays(d['_h'], float(x), float(y), float(z), _ptr(rels), len(gem_rels), _ptr(out))
                return out
        return d['_t'].edge_rays(x, y, z, gem_rels)

    def gap_along(self, x, y, z, ux, uy, max_u=TO.RAY_RANGE):
        d = self.__dict__
        if (max_u == TO.RAY_RANGE and type(x) is float and type(y) is float and type(z) is float
                and type(ux) in (float, np.float64) and type(uy) in (float, np.float64)):
            g = d['_gap']
            d['_lib'].ob_gap_along(d['_h'], x, y, z, float(ux), float(uy), 1 if type(ux) is np.float64 else 0,
                                   1 if type(uy) is np.float64 else 0, _ptr(g))
            return float(g[0]), float(g[1]), float(g[2])
        return d['_t'].gap_along(x, y, z, ux, uy, max_u)


_CACHE = {}


def native_queries(terrain, dll=None):
    """The compiled-query wrapper for this terrain (one per terrain object), or the terrain itself when the DLL or the
    terrain is not supported."""
    if not ENABLED or terrain is None or isinstance(terrain, NativeTerrainQueries):
        return terrain
    entry = _CACHE.get(id(terrain))
    if entry is not None and entry.terrain is terrain:
        return entry
    try:
        entry = NativeTerrainQueries(terrain, dll)
    except (OSError, ValueError, AttributeError, RuntimeError):
        return terrain
    _CACHE[id(terrain)] = entry
    return entry
