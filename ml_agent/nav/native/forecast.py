"""Compiled forecast of the 16 ms-physics / 64 ms-reply bridge (2026-10-16).

predict_native(sim, engine, dynamics, actions, stop_on_ambiguity) returns what
nav.marble_dynamics.bridge_small_step.predict_small_step returns for the same inputs: the proposals, the bridge's
reply translation, Blast and inventory script steps stay in Python (once per 64 ms reply); the four 16 ms physics
steps of each reply run in nav/native/marble_sim.cpp (ms_reply). The 16 ms tick states are kept packed in a NativeRows
array: positions and radii are read from it directly (pickup tests), a dict is built only when one is indexed.
"""
import copy
import ctypes

import numpy as np

from nav.marble_dynamics.state_copy import clone as clone_state
from nav.marble_dynamics.bridge import translated,observe_proposal_ticks
from nav.marble_dynamics.inventory import check as check_inventory
from nav.marble_dynamics.blast import event as blast_event, advance as advance_blast
from nav.marble_dynamics.planning_budget import check as check_budget
from nav.native.marble_sim import S_SIZE, ERRORS, PARAM_KEYS, _ptr


class NativeRows:
    """Packed 16 ms tick states of a compiled forecast; behaves as a read-only list of state dicts."""
    __slots__ = ('rows', 'template', '_cache')

    def __init__(self, rows, template):
        self.rows = rows; self.template = template; self._cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return NativeRows(self.rows[i], self.template)
        n = len(self.rows)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError('native state index out of range')
        s = self._cache.get(i)
        if s is None:
            from nav.native.marble_sim import NativeSim
            s = self._cache[i] = NativeSim.unpack(self.rows[i], self.template)
        return s

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __add__(self, other):
        if isinstance(other, NativeRows):
            return NativeRows(np.concatenate([self.rows, other.rows]), self.template)
        return list(self) + list(other)

    def __radd__(self, other):
        return list(other) + list(self)

    @property
    def positions(self):
        return self.rows[:, 0:3]

    @property
    def radii(self):
        return self.rows[:, 9]


def _move_array(move, out):
    out[0] = move['x']; out[1] = move['y']; out[2] = move.get('yaw', 0.)
    out[3] = 1. if move.get('jump') else 0.; out[4] = 1. if move.get('use') else 0.; out[5] = 1. if move.get('trigger1') else 0.
    return out


PER_REPLY_STATS = {}   # 2026-11-01 (C++ port of the per-reply families): controller type -> [forecasts, replies, seconds]
                       # of the forecasts still run reply by reply in Python; printed at exit when non-empty (diagnostic only)


def _tally(actions, replies, seconds):
    k = type(actions).__name__ if callable(actions) else 'list'
    c = PER_REPLY_STATS.get(k)
    if c is None:
        c = PER_REPLY_STATS[k] = [0, 0, 0.]
        if len(PER_REPLY_STATS) == 1:
            import atexit, sys
            atexit.register(lambda: print('PER_REPLY_STATS ' + ' '.join('%s:%d/%d/%.1fs' % (n, v[0], v[1], v[2]) for n, v in
                                          sorted(PER_REPLY_STATS.items(), key=lambda kv: -kv[1][2])), file=sys.stderr))
    c[0] += 1; c[1] += replies; c[2] += seconds


import os as _os
VERIFY_PORT = _os.environ.get('PQ_VERIFY_PORT', '') == '1'   # diagnostic: run the Python controller beside each ported family (ABI 15)
VERIFY_LOG = _os.path.join(_os.environ.get('TEMP', '.'), 'pq_checks', 'port_verify.log')
VERIFY_STATS = {}   # tail kind -> [forecasts, mismatches]
_SNAP_ATTRS = ('passed', 'previous', 'jumped', 'turned', 'hop_jumped')


def _snapshot(actions):
    snap = {a: copy.copy(getattr(actions, a)) for a in _SNAP_ATTRS if hasattr(actions, a)}
    routes = [getattr(actions, 'route', None), getattr(actions, 'next_route', None)]
    if getattr(actions, 'after', None) is not None:
        routes.append(actions.after[0])
    snap['_routes'] = [(r, r.waypoint) for r in routes if r is not None]
    return snap


def _restore(actions, snap):
    for a, v in snap.items():
        if a != '_routes':
            setattr(actions, a, v)
    for r, wp in snap['_routes']:
        r.waypoint = wp


def _verify_twin(sim, engine, dynamics, actions, tail, stop_on_ambiguity):
    """The Python per-reply forecast of the same proposal (its state restored afterwards), for _verify_compare."""
    snap = _snapshot(actions)
    hook = getattr(engine, 'native_step_hook', None)
    counted = [0]

    def counting(n):   # the Python path advances the (virtual) clock per reply; count it and roll it back afterwards
        counted[0] += n
        if hook is not None:
            hook(n)
    engine.native_step_hook = counting
    try:
        ref = _predict_per_reply_inner(sim, engine, dynamics, actions, stop_on_ambiguity)
    except Exception as exc:   # the Python controller may refuse what the compiled one also refuses, or hit the deadline
        ref = exc
    finally:
        engine.native_step_hook = hook
        if hook is not None and counted[0]:
            hook(-counted[0])
    after = _snapshot(actions); _restore(actions, snap)
    return ref, after, counted[0]


def _verify_compare(verify, actions, tail, rows, status, replies):
    ref, py_after, py_steps = verify
    c = VERIFY_STATS.setdefault(tail, [0, 0]); c[0] += 1
    if len(VERIFY_STATS) == 1 and c[0] == 1:
        import atexit, sys
        atexit.register(lambda: print('VERIFY_PORT ' + ' '.join('kind%d:%d/%d' % (k, v[1], v[0]) for k, v in sorted(VERIFY_STATS.items())), file=sys.stderr))
    done_replies, nrows, fail_reply = status[0], status[1], status[2]
    attempted = done_replies + (1 if fail_reply >= 0 else 0)
    why = None
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    if py_steps != status[6]:
        why = 'steps counted native %d python %d (native budget %d; python %s)' % (status[6], py_steps, status[5],
                                                                                  type(ref).__name__ if isinstance(ref, Exception) else 'ok')
    elif status[5]:
        if not isinstance(ref, PlanningBudgetExceeded):
            why = 'native budget cut, python %s' % (type(ref).__name__ if isinstance(ref, Exception) else 'completed')
    elif isinstance(ref, Exception):
        why = 'python raised %s' % str(ref)[:80]
    else:
        nat_prop = replies[:attempted, 0:9]
        py_prop = np.asarray(ref['proposed'], dtype=float).reshape(-1, 9) if ref['proposed'] else np.zeros((0, 9))
        n = min(len(nat_prop), len(py_prop))
        bad = [i for i in range(n) if not np.array_equal(nat_prop[i], py_prop[i])]
        if bad:
            i = bad[0]; why = 'reply %d action native %s python %s' % (i, nat_prop[i].tolist(), py_prop[i].tolist())
        elif len(nat_prop) != len(py_prop):
            why = 'proposed length native %d python %d (native fail %s stopped %s; python fail %s stopped %s)' % (
                len(nat_prop), len(py_prop), fail_reply, status[4], ref['failure'], ref.get('stopped_on_ambiguity'))
        else:
            py_rows = ref['native_states'].rows
            if py_rows.shape[0] != nrows or not np.array_equal(py_rows, rows[:nrows]):
                why = 'rows differ (native %d python %d)' % (nrows, py_rows.shape[0])
            elif (ref['failure'] is None) != (fail_reply < 0):
                why = 'failure native %s python %s' % (fail_reply, ref['failure'])
            else:
                nat_after = _snapshot(actions)
                for a in _SNAP_ATTRS:
                    if a in nat_after and not _same_state(nat_after[a], py_after[a]):
                        why = 'state %s native %r python %r' % (a, nat_after[a], py_after[a]); break
                if why is None:
                    for (r, wn), (_, wp) in zip(nat_after['_routes'], py_after['_routes']):
                        if not _same_state(wn, wp):
                            why = 'waypoint native %r python %r' % (wn, wp); break
    if why is not None:
        c[1] += 1
        try:
            _os.makedirs(_os.path.dirname(VERIFY_LOG), exist_ok=True)
            with open(VERIFY_LOG, 'a', encoding='utf-8') as f:
                f.write('%s kind %d: %s\n' % (type(actions).__name__, tail, why))
        except OSError:
            pass


def _same_state(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return a is not None and b is not None and np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))
    return a == b


def _predict_per_reply(sim, engine, dynamics, actions, stop_on_ambiguity=False):
    import time as _time
    t0 = _time.process_time()   # CPU time: the sim patches time.perf_counter to its virtual clock (the 11-01 tally read virtual seconds)
    try:
        return _predict_per_reply_inner(sim, engine, dynamics, actions, stop_on_ambiguity)
    finally:
        _tally(actions, len(actions) if not callable(actions) else int(actions.steps), _time.process_time() - t0)


def _predict_per_reply_inner(sim, engine, dynamics, actions, stop_on_ambiguity=False):
    native = dynamics['native']; state = clone_state(native['state'])
    if state['oob'] or state['mode'] != 0:
        raise ValueError('Playable non-OOB state required')
    if dynamics.get('blast') is not None:
        state['blast_context'] = copy.deepcopy(dynamics['blast'])
    if dynamics.get('inventory') is not None:
        state['inventory_context'] = copy.deepcopy(dynamics['inventory'])
    clock = int(native['frame_clock_ms']); move = copy.deepcopy(native['pending_move'])
    pending = list(dynamics['bridge_action']); hold = int(dynamics['pow_yaw_hold']); aim = float(dynamics['pow_yaw'])
    count = len(actions) if not callable(actions) else int(actions.steps)
    rows = np.empty((4 * count + 1, S_SIZE)); sim.pack(state, rows[0]); nrows = 1
    states = [clone_state(state)]; controls = []; events = []; proposed = []; native_controls = []
    failure = None; stopped = False
    m0 = np.empty(6); m1 = np.empty(6); dv = np.zeros(3); amb = (ctypes.c_int * 4)()
    lib = sim.lib; handle = sim.handle
    p_m0 = _ptr(m0); p_m1 = _ptr(m1); p_dv = _ptr(dv); base = rows.ctypes.data; stride = S_SIZE * 8; P = ctypes.c_void_p
    for i in range(count):
        check_budget(engine)
        action = actions(state, i) if callable(actions) else actions[i]
        proposed.append(list(action)); next_move, camera, hold, aim = translated(pending, hold, aim)
        blast = state.get('blast_context')
        blast = dict(blast) if blast is not None else None   # the saved previous state keeps its own copy
        try:
            check_inventory(state.get('inventory_context'), state, pending)
            impulse = blast_event(blast, state, bool(pending[8]))
        except ValueError as exc:
            failure = dict(step=i, reason=str(exc)); break
        has_camera = 0; yaw = 0.
        for e in camera:
            if e['kind'] != 'camera_yaw':
                raise ValueError('Unsupported external command: ' + e['kind'])
            has_camera = 1; yaw = float(e['value'])
        if impulse is not None:
            dv[:] = impulse['delta_velocity']
        _move_array(move, m0); _move_array(next_move, m1)
        result = lib.ms_reply(handle, P(base + (nrows - 1) * stride), p_m0, p_m1,
                              clock + i * 64, has_camera, yaw, 1 if impulse is not None else 0, p_dv,
                              1 if stop_on_ambiguity else 0, P(base + nrows * stride), amb)
        done = result & 0xff; code = result >> 8
        frames = []
        for k in range(done):
            control = dict(move=dict(move if k == 0 else next_move), events=camera if k == 0 else [])
            if k == 0 and impulse is not None:
                control['script_events'] = [impulse]
            frames.append(control); native_controls.append(control)
            advance_blast(blast, k == 0 and impulse is not None, 16)
        nrows += done
        hook = getattr(engine, 'native_step_hook', None)
        if hook is not None:
            hook(done)
        if code:
            failure = dict(step=i, reason=ERRORS.get(code, 'native sim error %d' % code)); break
        macro = dict(move=dict(frames[0]['move']), events=[dict(e) for e in frames[0]['events']], subframes=frames)
        script_events = [ev for frame in frames for ev in frame.get('script_events', [])]
        if script_events:
            macro['script_events'] = script_events
        observe_proposal_ticks(actions,NativeRows(rows[nrows-done-1:nrows],state))
        state = sim.unpack(rows[nrows - 1], state)
        if blast is not None:
            state['blast_context'] = blast
        states.append(state); controls.append(macro)
        ambiguous = [k for k in range(done) if amb[k]]
        events.append([dict(kind='ambiguous_contact', subframe=k) for k in ambiguous])
        if stop_on_ambiguity and ambiguous:
            stopped = True
        move = next_move; pending = list(action)
        if stopped:
            break
    return dict(states=states, controls=controls, events=events, proposed=proposed, failure=failure,
                native_states=NativeRows(rows[:nrows], states[0]), native_controls=native_controls,
                time_s=[i * .064 for i in range(len(states))],
                ambiguous_intervals=[i for i, ev in enumerate(events) if ev],
                requested_replies=count, stopped_on_ambiguity=stopped, physics_frame_ms=16, decision_ms=64,
                assumptions='Measured local16ms physics/64ms decisions (compiled nav/native/marble_sim.cpp), static geometry, '
                            'ordinary movement/jump and native powers1-5, observed regular/item-charged Blast.')


# ------------------------------------------------------------------------------- reply results built on demand (10-19)
LAZY_ROLLOUTS = True   # 2026-10-19 (operator: speed up the search toward real time): predict_native returns its states,
                       # controls and events as sequences built on first use from the packed C++ output (same values as the
                       # eager lists before). A search reads only a few states of most candidates (the last one, the tail,
                       # one mid-plan state, the positions); building all 48 state dicts per candidate was ~12 % of a
                       # search, and the packed form pickles ~30x faster for the parallel search's workers.


class _LazySeq:
    """Read-only list built on demand; slices and concatenations give plain lists."""
    __slots__ = ()

    def _all(self):
        raise NotImplementedError

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError('rollout sequence index out of range')
        return self._item(i)

    def _item(self, i):
        return self._all()[i]

    def __iter__(self):
        for i in range(self._n):
            yield self._item(i)

    def __add__(self, other):
        return list(self) + list(other)

    def __radd__(self, other):
        return list(other) + list(self)

    def __eq__(self, other):
        try:
            return list(self) == list(other)
        except TypeError:
            return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = None

    def __repr__(self):
        return repr(list(self))


class ReplyStates(_LazySeq):
    """The reply-boundary states of a compiled forecast. State i (i >= 1) is what predict_native built eagerly before:
    state 0's keys, the packed fields of reply i's last tick row (marble_sim.NativeSim.unpack), next_yaw and
    best_contact_normal from the last reply row that set them (else state 0's), and the Blast meter of reply i - 1."""
    __slots__ = ('native', 'idx', 'state0', 'blast', 'ny_last', 'bcn_last', '_n', '_cache')

    def __init__(self, native, idx, state0, blast):
        self.native = native; self.idx = idx; self.state0 = state0; self.blast = blast; self._n = len(idx); self._cache = {}
        self.ny_last = self.bcn_last = None   # built by the first state dict (2026-11-01: most candidates never build one)

    def _lasts(self):
        rows = self.native.rows; idx = self.idx; ar = np.arange(self._n)
        ny = np.where(rows[idx, 64] != 0, ar, -1); ny[0] = -1
        bc = np.where(rows[idx, 42] != 0, ar, -1); bc[0] = -1
        self.ny_last = np.maximum.accumulate(ny); self.bcn_last = np.maximum.accumulate(bc)

    def __getstate__(self):
        return (self.native, self.idx, self.state0, self.blast, self.ny_last, self.bcn_last, self._n)

    def __setstate__(self, st):
        self.native, self.idx, self.state0, self.blast, self.ny_last, self.bcn_last, self._n = st; self._cache = {}

    @property
    def positions(self):
        """(n, 3) float64 positions of the states (the same values as state['position'])."""
        return self.native.rows[self.idx, 0:3]

    @property
    def held(self):
        """Held powerup of each state (the same values as state['held'])."""
        return self.native.rows[self.idx, 17].astype(np.int64)

    def _item(self, i):
        s = self._cache.get(i)
        if s is not None:
            return s
        if i == 0:
            return self.state0
        rows = self.native.rows; x = rows[self.idx[i]].tolist(); s0 = self.state0
        if self.ny_last is None:
            self._lasts()
        s = dict(s0)
        s['position'] = x[0:3]; s['velocity'] = x[3:6]; s['spin'] = x[6:9]
        s['radius'] = x[9]; s['yaw'] = x[10]
        j = int(self.ny_last[i])
        if j >= 1:
            s['next_yaw'] = float(rows[self.idx[j], 11])
        s['held'] = int(x[17])
        s['effects'] = [[bool(x[19]), int(x[25])], [bool(x[20]), int(x[26])], [bool(x[21]), int(x[27])],
                        [bool(x[22]), int(x[28])], [bool(x[23]), int(x[29])], [bool(x[24]), int(x[30])]]
        s['effect_end_clock_ms'] = [int(x[31]), int(x[32]), int(x[33]), int(x[34]), int(x[35]), int(x[36])]
        s['clock_ms'] = int(x[38])
        s['camera_offset'] = x[65:68]
        j = int(self.bcn_last[i])
        if j >= 1:
            s['best_contact_normal'] = rows[self.idx[j], 39:42].tolist()
        prm = dict(s0['parameters'])
        for k, key in enumerate(PARAM_KEYS):
            if key in prm:
                prm[key] = x[43 + k]
        s['parameters'] = prm
        if self.blast is not None:
            b = dict(s0['blast_context']); b['value'] = float(self.blast[i - 1, 0]); b['special'] = bool(self.blast[i - 1, 1])
            s['blast_context'] = b
        self._cache[i] = s
        return s


class _ControlsBuilder:
    """Builds a compiled forecast's per-reply macro controls, per-tick controls and events together (as predict_native
    did eagerly: the per-tick controls are the macros' subframe objects)."""
    __slots__ = ('table', 'pend_move', 'n_done', 'fail_reply', '_built')

    def __init__(self, table, pend_move, n_done, fail_reply):
        self.table = table; self.pend_move = pend_move; self.n_done = n_done; self.fail_reply = fail_reply; self._built = None

    def __getstate__(self):
        return (self.table, self.pend_move, self.n_done, self.fail_reply)

    def __setstate__(self, st):
        self.table, self.pend_move, self.n_done, self.fail_reply = st; self._built = None

    def build(self):
        if self._built is not None:
            return self._built
        controls = []; events = []; native_controls = []; table = self.table.tolist()
        for i in range(len(table)):
            R = table[i]; done = int(R[31])
            move0 = copy.deepcopy(self.pend_move) if i == 0 else _move_dict(table[i - 1][15:21])
            move1 = _move_dict(R[15:21])
            camera = [dict(kind='camera_yaw', value=float(R[21]))]
            impulse = None
            if R[22] == 1:
                impulse = dict(kind='blast_impulse', impulse=R[23:26], delta_velocity=R[26:29], meter_before=R[29], special=bool(R[30]))
            frames = []
            for k in range(done):
                control = dict(move=dict(move0 if k == 0 else move1), events=camera if k == 0 else [])
                if k == 0 and impulse is not None:
                    control['script_events'] = [impulse]
                frames.append(control); native_controls.append(control)
            if i == self.fail_reply:
                break
            macro = dict(move=dict(frames[0]['move']), events=[dict(e) for e in frames[0]['events']], subframes=frames)
            if impulse is not None:
                macro['script_events'] = [impulse]
            controls.append(macro)
            events.append([dict(kind='ambiguous_contact', subframe=k) for k in range(done) if R[32 + k]])
        self._built = (controls, events, native_controls)
        return self._built


class _BuiltSeq(_LazySeq):
    __slots__ = ('builder', 'part', '_n', 'script_event_count')

    def __init__(self, builder, part, n, script_event_count=None):
        self.builder = builder; self.part = part; self._n = n
        self.script_event_count = script_event_count   # macro controls: replies with a Blast impulse (their script events)

    def __getstate__(self):
        return (self.builder, self.part, self._n, self.script_event_count)

    def __setstate__(self, st):
        self.builder, self.part, self._n, self.script_event_count = st

    def _all(self):
        return self.builder.build()[self.part]


# ---------------------------------------------------------------------------------------------- whole forecast in C++
Q_SIZE = 64; R_SIZE = 40   # ABI 15: + the bridge state (hold, pow_yaw) after each reply
ERRORS_EXTRA = {7: 'Blast context unavailable in this snapshot', 8: 'Triple Blast, Fireball or cannon actuation is not verified',
                10: 'Positive native mass required for Blast', 11: 'Positive Blast charge time required',
                12: 'Timed Mega activation/expiry requires a verified script-event model',
                13: 'Script/native radius transition is not verified', 14: 'Script-held Mega activation is not verified',
                20: 'Unsupported compiled controller'}


def seq_array(spec):
    """search.SequenceProposal -> the 64-value layout of marble_sim.cpp (route / through state included; the brake
    prefix and aim offset of the Settle / Recharge / Aim wrappers are set by _controller)."""
    q = np.zeros(Q_SIZE)
    q[0:3] = spec.route.key
    if spec.route.waypoint is not None:
        q[3] = 1; q[4:7] = spec.route.waypoint[:3]
    q[7:10] = np.asarray(spec.gem, dtype=float)[:3]
    q[10] = spec.offset; q[11] = spec.scale; q[12] = spec.jump; q[13] = spec.use; q[14] = spec.turn_after
    q[15] = -1 if spec.brake_after is None else spec.brake_after; q[16] = spec.blast; q[17] = 1 if spec.direct else 0
    if spec.through is not None:
        q[18] = 1; q[19:22] = spec.next_route.key
        if spec.next_route.waypoint is not None:
            q[22] = 1; q[23:26] = spec.next_route.waypoint[:3]
        q[26:29] = np.asarray(spec.through[1], dtype=float)[:3]; q[29] = spec.through[2]
    if spec.arrive is not None:
        q[30] = 1; q[31] = spec.arrive
    q[32] = 1 if spec.passed else 0
    if spec.previous is not None:
        q[33] = 1; q[34:37] = spec.previous[:3]
    if getattr(spec, 'min_speed', None) is not None:   # ABI 8 (v62)
        q[40] = 1; q[41] = spec.min_speed
    if getattr(spec, 'box', None) is not None:         # ABI 9: the through gem's pickup box (search.THROUGH_BOX)
        q[42] = 1; q[43:49] = np.asarray(spec.box, dtype=float)[:6]
    # ABI 10: fast families were previously silently serialized as ordinary plans.
    from nav.marble_dynamics.search import FAST_MIN_SPEED
    q[49] = float(spec.fast or 0.)
    q[50] = float(FAST_MIN_SPEED)
    df = getattr(spec, 'direct_from', None)   # ABI 12: stored + 1 (0 = never)
    q[51] = 0. if df is None else float(int(df) + 1)
    import nav.marble_dynamics.search as S   # ABI 13
    q[52] = float(getattr(spec, 'lead_k', 0.) or 0.); q[53] = float(S.LEAD_AIM_MAX)
    q[54] = 1. if getattr(spec, 'land', False) else 0.
    q[55:61] = (S.LAND_G, S.LAND_AIR, S.LAND_DZ, S.LAND_MIN_VZ, S.LAND_MIN_T, S.LAND_AFTER_USE)
    q[61:64] = (float(getattr(spec, 'land_short', False)), S.LAND_SHORT_ARRIVE, S.LAND_SHORT_DECEL)
    return q


def _seq_writeback(spec, q):
    spec.route.waypoint = tuple(float(v) for v in q[4:7]) if q[3] else None
    if spec.through is not None:
        spec.next_route.waypoint = tuple(float(v) for v in q[23:26]) if q[22] else None
    spec.passed = bool(q[32]); spec.previous = np.asarray(q[34:37], dtype=float) if q[33] else None


def _goal_key(v):
    return tuple(round(float(x), 3) for x in np.asarray(v)[:3])


SEQ_TAILS = (1, 7, 8, 9, 10)   # tail kinds run by seq_call: the sequence plan and its one-rule subclasses (ABI 15; 10: edge pops, 2026-11-04)


def _family_data(actions):
    """(tail kind, xdata, after-route field, its terrain) for the per-reply families ported to C++ on 2026-11-01 (ABI 15):
    search.GapRunProposal (4), climbs.ClimbProposal (5), racing_line.TrackLine (6); None when the Python controller must run."""
    import nav.marble_dynamics.search as S
    from nav.marble_dynamics.climbs import ClimbProposal, RADIUS as CLIMB_RADIUS, LEAD_S as CLIMB_LEAD_S, PLACE_BELOW_U
    from nav.marble_dynamics.racing_line import TrackLine, A_STICK
    from nav.routing import goal_key
    kind = type(actions)
    if kind is S.GapRunProposal:
        x = np.zeros(16)
        x[0:3] = actions.launch[:3]; x[3:5] = actions.u[:2]; x[5:8] = actions.land[:3]; x[8] = actions.land_along; x[9] = actions.edge_along
        x[10] = actions.trigger; x[11] = actions.blast_after; x[12] = actions.speed; x[13] = S.GAP_RUN_LEAD_S; x[14] = S.GAP_RUN_LOOK_U
        x[15] = -1 if actions.jumped is None else int(actions.jumped)
        return 4, x, None, None
    if kind is ClimbProposal:
        x = np.zeros(28); x[0] = 1. if actions.kind == 'wall' else 0.; x[1:4] = actions.launch[:3]; x[4:6] = actions.u[:2]; x[6:9] = actions.land[:3]
        x[9] = actions.z_place; x[10] = actions.jump_u; x[11] = actions.blast_after; x[12] = CLIMB_RADIUS; x[13] = CLIMB_LEAD_S; x[14] = PLACE_BELOW_U
        field = terrain = None
        if actions.after is not None:
            nroute, ngem = actions.after
            if (not isinstance(ngem, np.ndarray) or ngem.dtype != np.float64 or ngem.ndim != 1 or len(ngem) < 3 or nroute.field is None
                    or nroute.key != goal_key(ngem) or nroute.field.dtype != np.float32 or not nroute.field.flags.c_contiguous
                    or getattr(nroute.terrain, 'ground_clearance', None) is None):
                return None
            wp = nroute.waypoint
            if wp is not None and not all(type(v) is float for v in wp[:3]):
                return None
            x[15] = 1; x[16:19] = nroute.key[:3]; x[19:22] = ngem[:3]
            if wp is not None:
                x[22] = 1; x[23:26] = wp[:3]
            field = nroute.field; terrain = nroute.terrain
        x[26] = -1 if actions.jumped is None else int(actions.jumped); x[27] = 1. if actions.turned else 0.
        return 5, x, field, terrain
    if kind is TrackLine:
        n = len(actions.t)
        if n < 2 or actions.p.shape != (n, 2) or actions.v.shape != (n, 2) or actions.a.shape != (n, 2):
            return None
        x = np.concatenate([[float(n), actions.z, actions.KP, actions.KD, actions.LEAD, actions.STOP_DECEL, A_STICK, 0.],
                            np.asarray(actions.t, dtype=float), np.asarray(actions.p, dtype=float).ravel(),
                            np.asarray(actions.v, dtype=float).ravel(), np.asarray(actions.a, dtype=float).ravel()])
        return 6, x, None, None
    return None


def _sub_rule_data(actions):
    """(tail kind, xdata) for the SequenceProposal subclasses with one extra rule (ABI 15), else None."""
    from nav.marble_dynamics.edge_hits import WallHitProposal
    from nav.marble_dynamics.pad_hops import PadHopProposal, ThroughHopProposal
    import nav.marble_dynamics.pad_hops as PH
    kind = type(actions)
    if kind is WallHitProposal:
        return 7, np.asarray([actions.trigger_u, 1. if actions.jumped else 0.], dtype=float)
    if kind is PadHopProposal:
        return 8, np.asarray([actions.hop_speed, actions.pad_height, actions.pad_radius, PH.HOP_MIN_BELOW_U, PH.JUMP_IMPULSE, PH.GRAVITY,
                              PH.PAD_ABOVE_U, 1. if actions.jumped else 0.], dtype=float)
    if kind is ThroughHopProposal:
        return 9, np.asarray([actions.pad_radius, actions.hop_rise, PH.PAD_ABOVE_U, PH.JUMP_IMPULSE, PH.GRAVITY, PH.ARRIVE_MIN_U_S,
                              PH.ARRIVE_MAX_U_S, 1. if actions.hop_jumped else 0.], dtype=float)
    from nav.marble_dynamics.edge_pop import EdgePopProposal
    if kind is EdgePopProposal:   # 2026-11-04: the speed cap held at the edge (the jump is the sequence plan's own)
        return 10, np.asarray([actions.cap, 0.], dtype=float)
    return None


def _family_writeback(tail, spec, x):
    """The per-forecast state the Python controller would hold after the same forecast (as _seq_writeback for a sequence plan)."""
    if tail == 4:
        spec.jumped = None if x[15] < 0 else int(x[15])
    elif tail == 5:
        spec.jumped = None if x[26] < 0 else int(x[26]); spec.turned = bool(x[27])
        if spec.after is not None:
            spec.after[0].waypoint = (float(x[23]), float(x[24]), float(x[25])) if x[22] else None
    elif tail == 7:
        spec.jumped = bool(x[1])
    elif tail == 8:
        spec.jumped = bool(x[7])
    elif tail == 9:
        spec.hop_jumped = bool(x[7])


def _controller(sim, actions, count):
    """(prefix actions, tail kind, SequenceProposal or brake goal or family proposal, extra) for controllers the compiled loop runs,
    else None. extra = (xdata, field, terrain) for the ported families (tail kinds 4-9), else None."""
    import nav.marble_dynamics.search as S
    from nav.marble_dynamics.ballistic import NoInput
    if not callable(actions):
        return [list(a) for a in actions], 0, None, None
    kind = type(actions)
    if kind is NoInput:
        return [actions(None, i) for i in range(count)], 0, None, None
    if kind is S.BrakeContinuation:
        return [list(a) for a in actions.prefix], 2, np.asarray(actions.gem, dtype=float), None
    from nav.marble_dynamics.landing import LandingContinuation as SupportLanding
    if kind is SupportLanding:
        # the fixed prefix, then Proposal(goal) (no braking) toward the sampled support point
        return [list(a) for a in actions.prefix], 3, np.asarray(actions.goal, dtype=float), None
    fam = _family_data(actions)
    if fam is not None:
        tail, x, field, terrain = fam
        return [], tail, actions, (x, field, terrain)
    import nav.marble_dynamics.gem_demo as GD
    if kind is GD.Proposal:
        # 2026-11-01: a bare braking Proposal (the stopping probes of search.check_backups / braking_viability) = the brake tail
        if (actions.brake and not actions.coast and not actions.offset and actions.jump_at < 0 and actions.use_at < 0
                and actions.speed_distance is None and not actions.min_speed and actions.top is None):
            return [], 2, np.asarray(actions.goal, dtype=float)[:3], None
        return None
    from nav.marble_dynamics.settle import SettleProposal
    from nav.marble_dynamics.recharge import RechargeProposal
    from nav.marble_dynamics.phased_search import AimProposal
    brake, aim = 0, None; tail = 1; extra = None
    if kind in (S.LandingContinuation, S.LaunchContinuation):
        prefix, spec = [list(a) for a in actions.prefix], actions.follow
        if type(spec) is not S.SequenceProposal:
            return None
    elif kind is S.SequenceProposal:
        prefix, spec = [], actions
    elif kind is AimProposal:
        prefix, spec, aim = [], actions, float(actions.aim_offset)
    elif kind is SettleProposal or kind is RechargeProposal:
        # brake toward the gem (Proposal(gem, brake=True), index 0), then the sequence at index - prefix
        prefix, spec = [], actions.after
        if type(spec) is not S.SequenceProposal:
            return None
        brake = int(actions.prefix if kind is SettleProposal else actions.wait_replies)
        if not np.array_equal(np.asarray(actions.gem, dtype=float)[:3], np.asarray(spec.gem, dtype=float)[:3]):
            return None
    else:
        sub = _sub_rule_data(actions)
        if sub is None:
            return None
        tail, x = sub; prefix, spec = [], actions; extra = (x, None, None)
    if spec.route.field is None or spec.route.key != _goal_key(spec.gem):
        return None
    if spec.through is not None:
        nr = spec.next_route
        if nr.field is None or nr.key != _goal_key(spec.through[1]) or nr.terrain is not spec.route.terrain:
            return None
    spec._native_brake, spec._native_aim = brake, aim
    return prefix, tail, spec, extra


def _blast_array(b):
    if b is None:
        return np.zeros(13)
    return np.asarray([1, b['enabled'], b['frozen'], b['triple'], b['fireball'], b['cannon'], b['special'], b['charging'],
                       b['value'], b['required'], b.get('special_power', 0.), b.get('power', 0.), b.get('charge_ms', 0.)], dtype=float)


def _inventory_array(c):
    if c is None:
        return np.zeros(5)
    return np.asarray([1, c['mega_active'], c['permanent_mega'], c.get('normal_radius', 0.), c['held_type']], dtype=float)


def _move_dict(row):
    return dict(x=float(row[0]), y=float(row[1]), z=0., yaw=0., pitch=0., jump=bool(row[3]), use=bool(row[4]), trigger1=False)


def _consts():
    import nav.marble_dynamics.search as S
    import nav.marble_dynamics.gem_demo as GD
    return np.asarray([GD.TARGET_SPEED_CAP, GD.BRAKE_DECEL, GD.THROTTLE_GAIN, GD.THROTTLE_CAP, S.PICKUP_MIN_SPEED,
                       S.THROUGH_SWITCH_U, 1. if S.ROUTE_SPEED_LAW else 0., float(S.OPEN_TOP or 0.), float(S.OPEN_NEAR_U),
                       float(S.OPEN_FAR_U), float(S.TRAVEL_TOP or 0.), float(S.TRAVEL_NEAR_U), float(S.TRAVEL_FAR_U)], dtype=float)   # ABI 7


class _Prepared:
    """One compiled candidate's inputs (predict_native / predict_many share them)."""
    __slots__ = ('actions', 'count', 'prefix', 'tail', 'spec', 'extra', 'n_prefix', 'pre', 'q', 'field', 'nfield', 'terrain', 'xdata', 'verify')

    def __init__(self, sim, engine, dynamics, actions, count, plan, stop_on_ambiguity):
        self.actions = actions; self.count = count
        self.prefix, self.tail, self.spec, self.extra = plan
        self.verify = (_verify_twin(sim, engine, dynamics, actions, self.tail, stop_on_ambiguity)
                       if VERIFY_PORT and (self.tail >= 4 or (self.tail == 2 and type(actions).__name__ == 'Proposal')) else None)
        n_prefix = self.n_prefix = min(len(self.prefix), count)
        self.pre = np.ascontiguousarray(np.asarray(self.prefix[:n_prefix], dtype=float).reshape(-1, 9) if n_prefix else np.zeros((1, 9)))
        tail, spec, extra = self.tail, self.spec, self.extra
        self.terrain = None; self.field = self.nfield = None; q = np.zeros(Q_SIZE); self.xdata = extra[0] if extra is not None else None
        if tail in SEQ_TAILS:
            q = seq_array(spec); self.field = np.ascontiguousarray(spec.route.field, dtype=np.float32)
            q[37] = spec._native_brake
            if spec._native_aim is not None:
                q[38] = 1; q[39] = spec._native_aim
            self.nfield = np.ascontiguousarray(spec.next_route.field, dtype=np.float32) if spec.through is not None else None
            self.terrain = spec.route.terrain
        elif tail == 2 or tail == 3:
            q[7:10] = spec[:3]
        elif tail == 5 and extra[1] is not None:
            self.field = extra[1]; self.terrain = extra[2]
        self.q = q

    def writeback(self, engine, rows, status, replies):
        if self.tail in SEQ_TAILS:
            _seq_writeback(self.spec, self.q)
        if self.xdata is not None:
            _family_writeback(self.tail, self.spec, self.xdata)
        if self.verify is not None:
            _verify_compare(self.verify, self.actions, self.tail, rows, status, replies)


def _start(dynamics):
    """(state with its contexts, blast0, inventory, bridge array) of a forecast's starting dynamics."""
    native = dynamics['native']; state = clone_state(native['state'])
    if state['oob'] or state['mode'] != 0:
        raise ValueError('Playable non-OOB state required')
    # the contexts are flat dicts of scalars: state_copy.clone copies them as deepcopy did, in a tenth of the time (profile 11-01)
    blast0 = clone_state(dynamics['blast']) if dynamics.get('blast') is not None else None
    inventory = clone_state(dynamics['inventory']) if dynamics.get('inventory') is not None else None
    if blast0 is not None:
        state['blast_context'] = blast0
    if inventory is not None:
        state['inventory_context'] = inventory
    bridge = np.zeros(18)
    _move_array(native['pending_move'], bridge[0:6]); bridge[6:15] = dynamics['bridge_action']; bridge[15] = int(dynamics['pow_yaw_hold'])
    bridge[16] = float(dynamics['pow_yaw']); bridge[17] = int(native['frame_clock_ms'])
    return state, blast0, inventory, bridge


def _budget(engine):
    """(budget_s for the compiled loop's own clock, max_steps for the virtual clock) from the planner deadline."""
    import time
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    deadline = getattr(engine, 'planning_deadline', None); step_cost = getattr(engine, 'virtual_step_cost_s', None)
    max_steps = -1; budget_s = 0.
    if deadline is not None:
        now = time.perf_counter()
        if now >= deadline:
            raise PlanningBudgetExceeded('Declared planning wall-time budget exceeded')
        if step_cost:
            max_steps = int((deadline - now) / step_cost)
        else:
            budget_s = float(deadline - now)   # C++ adds it to its own clock (perf_counter's origin is per process)
    return budget_s, max_steps


def _finish(dynamics, state, blast0, count, stop_on_ambiguity, rows, replies, bridge_out, status, own_state=False):
    """The forecast result (what predict_small_step returns) from the compiled outputs.
    own_state: the caller's state is a fresh clone nobody else holds (predict_native), so it is state 0 as it is."""
    done_replies, nrows, fail_reply, fail_code, stopped, budget, steps = (status[k] for k in range(7))
    pend_move = dynamics['native']['pending_move']
    attempted = done_replies + (1 if fail_reply >= 0 else 0)
    if LAZY_ROLLOUTS:
        tab = replies[:attempted]
        proposed = []
        for a in tab[:, 0:9].tolist():
            a[4] = int(a[4]); a[6] = int(a[6]); a[8] = int(a[8])
            proposed.append(a)
        n_done = fail_reply if fail_reply >= 0 else attempted
        done = tab[:, 31].astype(np.int64)
        idx = np.concatenate([[0], np.cumsum(done[:n_done])]).astype(np.int64)
        state0 = state if own_state else clone_state(state)
        native_rows = NativeRows(rows[:nrows], state0)
        states = ReplyStates(native_rows, idx, state0, tab[:n_done, 36:38].copy() if blast0 is not None else None)
        builder = _ControlsBuilder(tab.copy(), pend_move, n_done, fail_reply)
        if n_done:
            amb = (tab[:n_done, 32:36] != 0) & (np.arange(4)[None, :] < done[:n_done, None])
            ambiguous = np.flatnonzero(amb.any(axis=1)).tolist()
        else:
            ambiguous = []
        failure = None
        if fail_reply >= 0:
            failure = dict(step=fail_reply, reason=ERRORS_EXTRA.get(fail_code) or ERRORS.get(fail_code, 'native sim error %d' % fail_code))
        bridge_resume = None
        if failure is None and not stopped and n_done == count:
            bridge_resume = dict(reply_count=n_done, pending_move=_move_dict(tab[n_done-1, 15:21]) if n_done else clone_state(pend_move),
                bridge_action=list(proposed[-1]) if n_done else list(dynamics['bridge_action']),
                pow_yaw_hold=int(status[7]), pow_yaw=float(bridge_out[0]))
        return dict(states=states, controls=_BuiltSeq(builder, 0, n_done, int((tab[:n_done, 22] == 1).sum())),
                    events=_BuiltSeq(builder, 1, n_done), proposed=proposed,
                    failure=failure, native_states=native_rows, native_controls=_BuiltSeq(builder, 2, int(done.sum())),
                    time_s=[i * .064 for i in range(n_done + 1)], ambiguous_intervals=ambiguous,
                    requested_replies=count, stopped_on_ambiguity=bool(stopped), physics_frame_ms=16, decision_ms=64,
                    bridge_resume=bridge_resume, reply_table=builder.table,
                    assumptions='Measured local16ms physics/64ms decisions (compiled forecast nav/native), static geometry, '
                                'ordinary movement/jump and native powers1-5, observed regular/item-charged Blast.')
    states = [clone_state(state)]; controls = []; events = []; native_controls = []
    table = replies[:attempted].tolist()   # one conversion; row access on Python lists is fast
    proposed = []
    for a in table:
        a = a[0:9]; a[4] = int(a[4]); a[6] = int(a[6]); a[8] = int(a[8])
        proposed.append(a)
    template = states[0]; row = 1
    for i in range(attempted):
        R = table[i]; done = int(R[31])
        move0 = copy.deepcopy(pend_move) if i == 0 else _move_dict(table[i - 1][15:21])
        move1 = _move_dict(R[15:21])
        camera = [dict(kind='camera_yaw', value=float(R[21]))]
        impulse = None
        if R[22] == 1:
            impulse = dict(kind='blast_impulse', impulse=R[23:26], delta_velocity=R[26:29], meter_before=R[29], special=bool(R[30]))
        frames = []
        for k in range(done):
            control = dict(move=dict(move0 if k == 0 else move1), events=camera if k == 0 else [])
            if k == 0 and impulse is not None:
                control['script_events'] = [impulse]
            frames.append(control); native_controls.append(control)
        row += done
        if i == fail_reply:
            break
        macro = dict(move=dict(frames[0]['move']), events=[dict(e) for e in frames[0]['events']], subframes=frames)
        if impulse is not None:
            macro['script_events'] = [impulse]
        st = NativeSim_unpack(rows[row - 1], template)
        if blast0 is not None:
            b = dict(st['blast_context']); b['value'] = float(R[36]); b['special'] = bool(R[37]); st['blast_context'] = b
        states.append(st); controls.append(macro); template = st
        events.append([dict(kind='ambiguous_contact', subframe=k) for k in range(done) if R[32 + k]])
    failure = None
    if fail_reply >= 0:
        failure = dict(step=fail_reply, reason=ERRORS_EXTRA.get(fail_code) or ERRORS.get(fail_code, 'native sim error %d' % fail_code))
    return dict(states=states, controls=controls, events=events, proposed=proposed, failure=failure,
                native_states=NativeRows(rows[:nrows], states[0]), native_controls=native_controls,
                time_s=[i * .064 for i in range(len(states))],
                ambiguous_intervals=[i for i, ev in enumerate(events) if ev],
                requested_replies=count, stopped_on_ambiguity=bool(stopped), physics_frame_ms=16, decision_ms=64,
                reply_table=np.asarray(replies[:attempted]).copy(),
                assumptions='Measured local16ms physics/64ms decisions (compiled forecast nav/native), static geometry, '
                            'ordinary movement/jump and native powers1-5, observed regular/item-charged Blast.')


def NativeSim_unpack(row, template):
    from nav.native.marble_sim import NativeSim
    return NativeSim.unpack(row, template)


def predict_native(sim, engine, dynamics, actions, stop_on_ambiguity=False):
    count = len(actions) if not callable(actions) else int(actions.steps)
    plan = _controller(sim, actions, count)
    if plan is None or getattr(engine, 'native_per_reply', False):
        return _predict_per_reply(sim, engine, dynamics, actions, stop_on_ambiguity)
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    P = _Prepared(sim, engine, dynamics, actions, count, plan, stop_on_ambiguity)
    state, blast0, inventory, bridge = _start(dynamics)
    budget_s, max_steps = _budget(engine)
    rows = np.empty((4 * count + 1, S_SIZE)); replies = np.zeros((max(count, 1), R_SIZE)); bridge_out = np.zeros(1); blast_out = np.zeros(13)
    status = (ctypes.c_int * 8)()
    start_row = np.empty(S_SIZE); sim.pack(state, start_row)
    th = sim.terrain(P.terrain) if P.terrain is not None else None
    sim.lib.ms_forecast_x(sim.handle, th, _ptr(start_row), _ptr(bridge), _ptr(_blast_array(blast0)), _ptr(_inventory_array(inventory)),
                          count, 1 if stop_on_ambiguity else 0, P.n_prefix, _ptr(P.pre), P.tail, _ptr(P.q),
                          _ptr(P.field) if P.field is not None else None, _ptr(P.nfield) if P.nfield is not None else None, _ptr(_consts()),
                          budget_s, max_steps, _ptr(rows), _ptr(replies), _ptr(bridge_out), _ptr(blast_out), status,
                          _ptr(P.xdata) if P.xdata is not None else None)
    hook = getattr(engine, 'native_step_hook', None)
    if hook is not None:
        hook(status[6])
    P.writeback(engine, rows, status, replies)
    if status[5]:
        raise PlanningBudgetExceeded('Declared planning wall-time budget exceeded')
    return _finish(dynamics, state, blast0, count, stop_on_ambiguity, rows, replies, bridge_out, status, own_state=True)


# ------------------------------------------------------------------------------------ many candidates, one start (ABI 15)
def predict_many(engine, dynamics, proposals, stop_on_ambiguity=False):
    """The forecasts of several proposals from one starting dynamics, in order, as predict_replies would return them one by one
    (the same numbers, the same virtual-clock accounting): runs of compiled candidates go to C++ in one call (ms_forecast_many),
    the others run through predict_replies. A candidate past the planning deadline is a PlanningBudgetExceeded instance in the
    result list, and every later one too. For the broader searches that expand many continuations from one state."""
    import nav.marble_dynamics.bridge as B
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    sim = getattr(engine, 'native_sim', None)
    results = [None] * len(proposals)
    if sim is None or getattr(engine, 'native_per_reply', False) or getattr(engine, 'bridge_physics_ms', 64) != 16:
        for i, p in enumerate(proposals):
            try:
                results[i] = B.predict_replies(engine, dynamics, p, stop_on_ambiguity=stop_on_ambiguity)
            except PlanningBudgetExceeded as exc:
                results[i] = exc
        return results
    # as bridge.predict_replies: ambiguity accepted (forecast run through, the intervals reported as accepted), Mega via Python
    accept = bool(B.ACCEPT_AMBIGUITY); stop = False if accept else stop_on_ambiguity
    i = 0; n = len(proposals)
    while i < n:
        count = len(proposals[i]) if not callable(proposals[i]) else int(proposals[i].steps)
        plan = None if B.mega_relevant(engine, dynamics, proposals[i]) else _controller(sim, proposals[i], count)
        if plan is None:
            try:
                results[i] = B.predict_replies(engine, dynamics, proposals[i], stop_on_ambiguity=stop_on_ambiguity)
            except PlanningBudgetExceeded as exc:
                results[i] = exc
            i += 1; continue
        batch = [(i, count, plan)]; j = i + 1
        while j < n:
            c = len(proposals[j]) if not callable(proposals[j]) else int(proposals[j].steps)
            pl = None if B.mega_relevant(engine, dynamics, proposals[j]) else _controller(sim, proposals[j], c)
            if pl is None:
                break
            batch.append((j, c, pl)); j += 1
        for k, res in zip([b[0] for b in batch], _run_many(sim, engine, dynamics, [proposals[b[0]] for b in batch], [b[1] for b in batch],
                                                             [b[2] for b in batch], stop)):
            if accept and not isinstance(res, Exception) and res.get('ambiguous_intervals'):
                res['ambiguous_accepted'] = res['ambiguous_intervals']; res['ambiguous_intervals'] = []
            results[k] = res
        i = j
    return results


def _run_many(sim, engine, dynamics, actions_list, counts, plans, stop_on_ambiguity):
    import time
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    n = len(actions_list)
    preps = [_Prepared(sim, engine, dynamics, a, c, pl, stop_on_ambiguity) for a, c, pl in zip(actions_list, counts, plans)]
    state, blast0, inventory, bridge = _start(dynamics)
    start_row = np.empty(S_SIZE); sim.pack(state, start_row)
    # the deadline: the compiled loop's own clock (live) or the virtual clock (the sim), advanced per candidate as the hook would
    deadline = getattr(engine, 'planning_deadline', None); step_cost = getattr(engine, 'virtual_step_cost_s', None)
    budget_s = 0.; vnow = vdeadline = 0.; vcost = 0.
    if deadline is not None:
        now = time.perf_counter()
        if now >= deadline:
            raise PlanningBudgetExceeded('Declared planning wall-time budget exceeded')
        if step_cost:
            vnow, vdeadline, vcost = float(now), float(deadline), float(step_cost)
        else:
            budget_s = float(deadline - now)
    cand = np.zeros(n * 8, dtype=np.int32); seq_all = np.zeros((n, Q_SIZE)); pre_parts = []; x_parts = []; fields = []; terrains = []
    rows_off = np.zeros(n, dtype=np.int64); reply_off = np.zeros(n, dtype=np.int64); pre_rows = 0; x_len = 0; nrows = 0; nrep = 0
    def field_index(arr):
        if arr is None:
            return -1
        fields.append(arr); return len(fields) - 1
    for k, P in enumerate(preps):
        d = cand[k * 8:k * 8 + 8]
        d[0] = P.count; d[1] = 1 if stop_on_ambiguity else 0; d[2] = P.n_prefix; d[3] = pre_rows; d[4] = P.tail
        pre_parts.append(P.pre); pre_rows += P.pre.shape[0]
        seq_all[k] = P.q
        d[5] = field_index(P.field); d[6] = field_index(P.nfield)
        if P.xdata is not None:
            d[7] = x_len; x_parts.append(P.xdata); x_len += len(P.xdata)
        else:
            d[7] = -1
        terrains.append(sim.terrain(P.terrain) if P.terrain is not None else None)
        rows_off[k] = nrows; reply_off[k] = nrep; nrows += 4 * P.count + 1; nrep += max(P.count, 1)
    pre_all = np.ascontiguousarray(np.concatenate(pre_parts, axis=0)) if pre_parts else np.zeros((1, 9))
    x_all = np.ascontiguousarray(np.concatenate(x_parts)) if x_parts else np.zeros(1)
    rows_all = np.empty((nrows, S_SIZE)); reply_all = np.zeros((nrep, R_SIZE)); bridge_out = np.zeros(n); blast_out = np.zeros((n, 13))
    status_all = (ctypes.c_int * (8 * n))()
    f_arr = (ctypes.c_void_p * max(1, len(fields)))(*[f.ctypes.data for f in fields]) if fields else (ctypes.c_void_p * 1)()
    t_arr = (ctypes.c_void_p * n)(*terrains)
    sim.lib.ms_forecast_many(sim.handle, _ptr(start_row), _ptr(bridge), _ptr(_blast_array(blast0)), _ptr(_inventory_array(inventory)),
                             n, cand.ctypes.data_as(ctypes.c_void_p), _ptr(pre_all), _ptr(seq_all), f_arr, t_arr, _ptr(x_all), _ptr(_consts()),
                             budget_s, vnow, vdeadline, vcost, _ptr(rows_all), _ptr(rows_off), _ptr(reply_all), _ptr(reply_off),
                             _ptr(bridge_out), _ptr(blast_out), status_all)
    # the spec and family arrays were copied into the shared buffers: copy the written-back state to each candidate's own arrays
    for k, P in enumerate(preps):
        P.q[:] = seq_all[k]
        if P.xdata is not None:
            off = cand[k * 8 + 7]; P.xdata[:] = x_all[off:off + len(P.xdata)]
    hook = getattr(engine, 'native_step_hook', None)
    out = []; exceeded = False
    for k, P in enumerate(preps):
        status = status_all[8 * k:8 * k + 8]
        if hook is not None and not exceeded:
            hook(status[6])
        rows = rows_all[rows_off[k]:rows_off[k] + 4 * P.count + 1]; replies = reply_all[reply_off[k]:reply_off[k] + max(P.count, 1)]
        if not exceeded:
            P.writeback(engine, rows, status, replies)
        if exceeded or status[5]:
            exceeded = True
            out.append(PlanningBudgetExceeded('Declared planning wall-time budget exceeded')); continue
        out.append(_finish(dynamics, state, blast0, P.count, stop_on_ambiguity, rows, replies, bridge_out[k:k + 1], status))
    return out


def resume_at(dynamics, rollout, k):
    """The dynamics at reply k (1 <= k <= replies) of a compiled forecast, as viability.resume_forecast builds them for its end:
    the k-th state, the bridge state after reply k - 1's translation (the reply table's hold / aim, ABI 15), the clock, the Blast
    and inventory contexts. A continuation from the middle of a plan (branching searches)."""
    table = rollout.get('reply_table')
    if table is None or rollout['failure']:
        raise ValueError('A compiled forecast without failure is required to resume mid-plan')
    states = rollout['states']; proposed = rollout['proposed']
    if not 1 <= k <= len(proposed) or k >= len(states):
        raise ValueError('Reply index out of the forecast')
    if any(i < k for i in rollout['ambiguous_intervals']):
        raise ValueError('Cannot continue a contact-ambiguous forecast')
    result = clone_state(dynamics)
    state = clone_state(states[k])
    result['native']['state'] = state
    result['native']['frame_clock_ms'] = int(dynamics['native']['frame_clock_ms']) + 64 * k
    result['native']['pending_move'] = _move_dict(table[k - 1][15:21])
    result['bridge_action'] = list(proposed[k - 1]); result['pow_yaw_hold'] = int(table[k - 1][38]); result['pow_yaw'] = float(table[k - 1][39])
    for source, key in [('blast_context', 'blast'), ('inventory_context', 'inventory')]:
        if source in state:
            result[key] = clone_state(state[source])
    return result
