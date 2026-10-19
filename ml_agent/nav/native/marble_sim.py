"""Python side of the compiled marble physics (nav/native/marble_sim.cpp), 2026-10-16.

NativeSim(engine) loads one map's geometry from a GeneralPhysics engine (its collision mesh, native polygons and
contact-order hulls) into the DLL. step(state, move) is the compiled counterpart of GeneralPhysics.step: the same
numbers for the same inputs (tools/native_check.py compares them on recorded steps).
"""
import ctypes
import os
from pathlib import Path

import numpy as np

DEFAULT_DLL = Path(os.environ.get('TEMP', '.')) / 'pq_checks' / 'marble_sim' / 'marble_sim.dll'
PARAM_KEYS = ('max_roll_velocity', 'angular_acceleration', 'braking_acceleration', 'static_friction', 'kinetic_friction',
              'bounce_kinetic_friction', 'gravity', 'max_dot_slide', 'bounce_restitution', 'air_acceleration',
              'jump_impulse', 'min_bounce_velocity', 'mass')
S_SIZE = 68
EVENT_KINDS = {1: 'ambiguous_contact', 2: 'support', 3: 'jump', 4: 'collision', 5: 'native_solver_limit'}
ERRORS = {1: 'Only ordinary playable marble mode is modeled', 2: 'Gravity rotation matrix is not yet modeled',
          3: 'Super Speed requires recorded previous best-contact normal',
          4: 'Script-driven activation requires an external-event model', 5: 'Random-force material is not modeled',
          6: 'Prediction diverged'}

_P = ctypes.c_void_p


def _ptr(a):
    return a.ctypes.data_as(_P)


class NativeSim:
    def __init__(self, engine, dll=None):
        self.lib = ctypes.CDLL(str(dll or DEFAULT_DLL))
        L = self.lib
        L.ms_abi.restype = ctypes.c_int
        if L.ms_abi() != 8:
            raise ValueError('Incompatible marble_sim ABI')
        L.ms_state_size.restype = ctypes.c_int
        if L.ms_state_size() != S_SIZE:
            raise ValueError('marble_sim state layout mismatch')
        L.ms_create.restype = _P
        L.ms_create.argtypes = [ctypes.c_int] + [_P] * 13 + [ctypes.c_int, _P]
        L.ms_set_polygons.argtypes = [_P, ctypes.c_int] + [_P] * 7
        L.ms_add_object.argtypes = [_P, ctypes.c_longlong, _P, _P, ctypes.c_int, _P, _P, _P, _P, _P, _P, ctypes.c_int, _P, _P, _P]
        L.ms_destroy.argtypes = [_P]
        L.ms_step.restype = ctypes.c_int
        L.ms_step.argtypes = [_P, _P, _P, ctypes.c_int, _P, _P, ctypes.c_int, _P]
        L.ms_contacts.restype = ctypes.c_int
        L.ms_reply.restype = ctypes.c_int
        L.ms_reply.argtypes = [_P, _P, _P, _P, ctypes.c_longlong, ctypes.c_int, ctypes.c_double, ctypes.c_int, _P, ctypes.c_int, _P, _P]
        L.ms_contacts.argtypes = [_P, _P, ctypes.c_double, _P, ctypes.c_int]
        D = ctypes.c_double; I = ctypes.c_int; LL = ctypes.c_longlong; F = ctypes.c_float
        L.ms_terrain_create.restype = _P
        L.ms_terrain_create.argtypes = [_P, F, F, D, I, I, I, _P, _P, _P, I, LL, _P, LL, _P, _P, _P, _P, _P, D, D, D, D]
        L.ms_terrain_destroy.argtypes = [_P]
        L.ms_terrain_seed_point.argtypes = [_P, _P, I]
        L.ms_terrain_set_edge.restype = None; L.ms_terrain_set_edge.argtypes = [_P, _P]
        L.ms_terrain_seed_segment.argtypes = [_P, _P, I]
        L.ms_cell_of.argtypes = [_P, D, D, I, _P]
        L.ms_level_at.restype = I; L.ms_level_at.argtypes = [_P, I, I, I, D]
        L.ms_line_clear.restype = I; L.ms_line_clear.argtypes = [_P, D, D, D, D, I, D, I, D, I, I]
        L.ms_path_waypoint.restype = I; L.ms_path_waypoint.argtypes = [_P, _P, D, D, _P, I, D, _P]
        L.ms_dist_at.restype = D; L.ms_dist_at.argtypes = [_P, _P, D, D, D, D, I, D, I]
        L.ms_segment_clear.restype = I; L.ms_segment_clear.argtypes = [_P, _P, _P]
        L.ms_translated.argtypes = [_P, _P, _P, _P, _P]
        L.ms_py_round.restype = D; L.ms_py_round.argtypes = [D, I]
        L.ms_g6.restype = D; L.ms_g6.argtypes = [D]
        L.ms_hypot.restype = D; L.ms_hypot.argtypes = [D, D]
        L.ms_seq_action.argtypes = [_P, _P, _P, _P, _P, _P, I, _P]
        L.ms_forecast.restype = I
        L.ms_forecast.argtypes = [_P, _P, _P, _P, _P, _P, I, I, I, _P, I, _P, _P, _P, _P, D, LL, _P, _P, _P, _P, _P]
        self.terrains = {}
        mesh = engine.mesh
        self.mesh = mesh
        mesh._face_geometry(np.asarray([], dtype=np.int64))
        keep = []

        def arr(a, dtype):
            a = np.ascontiguousarray(a, dtype=dtype); keep.append(a); return a
        n = len(mesh.triangles)
        materials = mesh.info['materials']
        props = arr([[float(m['friction']), float(m['restitution']), float(m['force'])] for m in materials], np.float64)
        args = [arr(mesh.triangles, np.float64), arr(mesh.source_triangles, np.float64), arr(mesh.planes, np.float64),
                arr(mesh.boundary_edges, np.uint8), arr(mesh._face_ab, np.float64), arr(mesh._face_ac, np.float64),
                arr(mesh._face_cross, np.float64), arr(mesh._face_nn, np.float64), arr(mesh.lo, np.float64),
                arr(mesh.hi, np.float64), arr(mesh.material_ids, np.int32), arr(mesh.object_ids, np.int64),
                arr(engine.polygon_ids, np.int32)]
        self.handle = L.ms_create(n, *[_ptr(a) for a in args], len(materials), _ptr(props))
        polys = engine.polygons
        vcount = arr([len(p['vertices']) for p in polys], np.int32)
        verts = arr(np.concatenate([p['vertices'] for p in polys]), np.float64)
        normals = arr([p['normal'] for p in polys], np.float64)
        ds = arr([p['d'] for p in polys], np.float64)
        edges = arr(np.concatenate([p['edges'] for p in polys]), np.float64)
        valid = arr(np.concatenate([p['edge_valid'] for p in polys]), np.uint8)
        first = arr([p['first_triangle'] for p in polys], np.int32)
        L.ms_set_polygons(self.handle, len(polys), _ptr(vcount), _ptr(verts), _ptr(normals), _ptr(ds), _ptr(edges), _ptr(valid), _ptr(first))
        for oid, o in engine.contact_order.objects.items():
            inv = arr(np.asarray(o['inverse'], np.float32).reshape(16), np.float32)
            scale = arr(o['scale'], np.float32)
            hlo = arr(o['hull_lo'], np.float32); hhi = arr(o['hull_hi'], np.float32)
            blo = arr(o['bbox_lo'], np.float32); bhi = arr(o['bbox_hi'], np.float32)
            bstart = arr([int(b.binStart) for b in o['bins']], np.int32); bcount = arr([int(b.binCount) for b in o['bins']], np.int32)
            idx = arr([int(i) for i in o['indices']], np.int32)
            hp = o['hull_polygons']
            off = arr(np.concatenate([[0], np.cumsum([len(h) for h in hp])]), np.int32)
            flat = arr([pid for h in hp for pid in h] or [0], np.int32)
            L.ms_add_object(self.handle, int(oid), _ptr(inv), _ptr(scale), len(hp), _ptr(hlo), _ptr(hhi), _ptr(blo), _ptr(bhi),
                            _ptr(bstart), _ptr(bcount), len(idx), _ptr(idx), _ptr(off), _ptr(flat))
        self._keep = keep
        self._in = np.zeros(S_SIZE); self._out = np.zeros(S_SIZE); self._move = np.zeros(6)
        self._events = np.zeros((256, 11)); self._nev = ctypes.c_int(0)

    def __del__(self):
        try:
            for handle, _ in self.terrains.values():
                self.lib.ms_terrain_destroy(handle)
            self.lib.ms_destroy(self.handle)
        except Exception:
            pass

    def terrain(self, terrain):
        """The C++ walk grid / route graph / ground clearance for a walking terrain (registered once, keyed by object)."""
        entry = self.terrains.get(id(terrain))
        if entry is not None and entry[1]['object'] is terrain:
            return entry[0]
        from nav.terrain import DROP_EDGE, STEP_UP, LEVEL_MERGE
        keep = dict(object=terrain)
        def arr(name, a, dtype):
            a = np.ascontiguousarray(a, dtype=dtype); keep[name] = a; return a
        graph = getattr(terrain, '_graph', None)
        walk_z = arr('walk_z', terrain.walk_z, np.float32)
        wxs = arr('wxs', terrain.wxs, np.float32); wys = arr('wys', terrain.wys, np.float32)
        node_idx = arr('node_idx', terrain._node_idx, np.int64)
        if graph is not None:
            indptr = arr('indptr', graph.indptr, np.int64); indices = arr('indices', graph.indices, np.int64)
            gdata = arr('gdata', graph.data, np.float64); nodes = arr('nodes', terrain._nodes, np.int64)
            nnodes = len(indptr) - 1; nnz = len(indices)
        else:
            indptr = indices = nodes = arr('dummy_i', [0], np.int64); gdata = arr('dummy_d', [0.], np.float64); nnodes = nnz = 0
        obstacle = arr('obstacle', self.mesh.normals[:, 2] <= .4, np.uint8)
        clearance = getattr(terrain, 'ground_clearance', None)
        radius = float(getattr(clearance, 'radius', .189750001))
        handle = self.lib.ms_terrain_create(self.handle, float(terrain.wxs[0]), float(terrain.wys[0]), float(terrain.walk_res),
            int(terrain.KW), int(terrain.wH), int(terrain.wW), _ptr(walk_z), _ptr(wxs), _ptr(wys), 1 if graph is not None else 0,
            nnodes, _ptr(indptr), nnz, _ptr(indices), _ptr(gdata), _ptr(node_idx), _ptr(nodes), _ptr(obstacle),
            radius, float(DROP_EDGE), float(STEP_UP), float(LEVEL_MERGE))
        edge = getattr(terrain, 'edge_dist', None)
        if edge is not None:   # search.open_top reads it (ABI 7)
            self.lib.ms_terrain_set_edge(handle, _ptr(arr('edge', edge, np.float32)))
        # the clearance results Python already holds (persistent caches): the same predicate, seeded as they are
        original = getattr(clearance, 'original', clearance)
        buf = np.zeros(7)
        for point, clear in list(getattr(original, 'point_cache', {}).items()):
            buf[:3] = point; self.lib.ms_terrain_seed_point(handle, _ptr(buf), 1 if clear else 0)
        for key, clear in list(getattr(clearance, 'regions', {}).items()):
            if len(key) == 7:
                buf[:] = key; self.lib.ms_terrain_seed_segment(handle, _ptr(buf), 1 if clear else 0)
        self.terrains[id(terrain)] = (handle, keep)
        return handle

    def bind_clearance(self, terrain):
        """Route the walking terrain's clearance misses (MemoGroundClearance.segment) to the compiled predicate, and
        register the terrain now (setup time) instead of inside the first decision that needs it."""
        clearance = getattr(terrain, 'ground_clearance', None)
        th = self.terrain(terrain)
        if clearance is None or not hasattr(clearance, 'native_segment'):
            return th
        fn = self.lib.ms_segment_clear; a = np.empty(3); b = np.empty(3); pa = _ptr(a); pb = _ptr(b)

        def segment(start, end):
            a[:] = start; b[:] = end
            return bool(fn(th, pa, pb))
        clearance.native_segment = segment
        return th

    # ---------------------------------------------------------------- state <-> packed array
    @staticmethod
    def pack(s, out):
        out[0:3] = s['position']; out[3:6] = s['velocity']; out[6:9] = s['spin']
        out[9] = s['radius']; out[10] = s['yaw']; out[11] = s.get('next_yaw', 0.); out[12] = s.get('pitch', .45)
        out[13] = 1. if s.get('smooth_camera', False) else 0.; out[14:17] = s['gravity_direction']
        out[17] = int(s['held']); out[18] = s['mode']
        eff = s['effects']
        for i in range(6):
            out[19 + i] = 1. if eff[i][0] else 0.; out[25 + i] = eff[i][1]
        if 'effect_end_clock_ms' in s:
            out[31:37] = s['effect_end_clock_ms']; out[37] = 1.
        else:
            out[37] = 0.
        out[38] = int(s.get('clock_ms', 0))
        if 'best_contact_normal' in s:
            out[39:42] = s['best_contact_normal']; out[42] = 1.
        else:
            out[42] = 0.
        prm = s['parameters']
        for k, key in enumerate(PARAM_KEYS):
            out[43 + k] = prm.get(key, 1. if key == 'mass' else 0.)
        out[56] = 1. if 'mass' in prm else 0.
        if 'effect_duration_ms' in prm:
            out[57:63] = prm['effect_duration_ms']; out[63] = 1.
        else:
            out[63] = 0.
        out[64] = 1. if 'next_yaw' in s else 0.
        out[65:68] = s.get('camera_offset', (0., 0., 0.))

    @staticmethod
    def unpack(d, template):
        """A new state dict: the template's keys, with every field the step changes taken from the packed array."""
        x = d.tolist() if hasattr(d, 'tolist') else list(d)   # one conversion to Python floats
        s = dict(template)
        s['position'] = x[0:3]; s['velocity'] = x[3:6]; s['spin'] = x[6:9]
        s['radius'] = x[9]; s['yaw'] = x[10]
        if x[64]:
            s['next_yaw'] = x[11]
        s['held'] = int(x[17])
        s['effects'] = [[bool(x[19]), int(x[25])], [bool(x[20]), int(x[26])], [bool(x[21]), int(x[27])],
                        [bool(x[22]), int(x[28])], [bool(x[23]), int(x[29])], [bool(x[24]), int(x[30])]]
        s['effect_end_clock_ms'] = [int(x[31]), int(x[32]), int(x[33]), int(x[34]), int(x[35]), int(x[36])]
        s['clock_ms'] = int(x[38])
        s['camera_offset'] = x[65:68]
        if x[42]:
            s['best_contact_normal'] = x[39:42]
        prm = dict(template['parameters'])
        for k, key in enumerate(PARAM_KEYS):
            if key in prm:
                prm[key] = x[43 + k]
        s['parameters'] = prm
        return s

    def step(self, state, move, dt_ms=16):
        self.pack(state, self._in)
        m = self._move
        m[0] = move['x']; m[1] = move['y']; m[2] = move.get('yaw', 0.)
        m[3] = 1. if move.get('jump') else 0.; m[4] = 1. if move.get('use') else 0.; m[5] = 1. if move.get('trigger1') else 0.
        code = self.lib.ms_step(self.handle, _ptr(self._in), _ptr(m), int(dt_ms), _ptr(self._out), _ptr(self._events), 256, ctypes.byref(self._nev))
        if code:
            raise ValueError(ERRORS.get(code, 'native sim error %d' % code))
        s = self.unpack(self._out, state)
        n = min(self._nev.value, 256)
        events = []
        for row in self._events[:n]:
            kind = EVENT_KINDS[int(row[0])]
            events.append(dict(kind=kind, sub=int(row[1]), triangle_id=int(row[2]), a=row[3:6].tolist(), b=row[6:9].tolist(), value=float(row[9])))
        return s, events

    def contacts(self, p, radius):
        out = np.zeros((64, 9))
        n = self.lib.ms_contacts(self.handle, _ptr(np.ascontiguousarray(p, dtype=np.float64)), float(radius), _ptr(out), 64)
        return out[:min(n, 64)]
