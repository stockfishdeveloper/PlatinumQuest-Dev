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


def _predict_per_reply(sim, engine, dynamics, actions, stop_on_ambiguity=False):
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
        rows = native.rows; ar = np.arange(self._n)
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
Q_SIZE = 61; R_SIZE = 38
ERRORS_EXTRA = {7: 'Blast context unavailable in this snapshot', 8: 'Triple Blast, Fireball or cannon actuation is not verified',
                10: 'Positive native mass required for Blast', 11: 'Positive Blast charge time required',
                12: 'Timed Mega activation/expiry requires a verified script-event model',
                13: 'Script/native radius transition is not verified', 14: 'Script-held Mega activation is not verified',
                20: 'Unsupported compiled controller'}


def seq_array(spec):
    """search.SequenceProposal -> the 61-value layout of marble_sim.cpp (route / through state included; the brake
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
    return q


def _seq_writeback(spec, q):
    spec.route.waypoint = tuple(float(v) for v in q[4:7]) if q[3] else None
    if spec.through is not None:
        spec.next_route.waypoint = tuple(float(v) for v in q[23:26]) if q[22] else None
    spec.passed = bool(q[32]); spec.previous = np.asarray(q[34:37], dtype=float) if q[33] else None


def _goal_key(v):
    return tuple(round(float(x), 3) for x in np.asarray(v)[:3])


def _controller(sim, actions, count):
    """(prefix actions, tail kind, SequenceProposal or brake goal) for controllers the compiled loop runs, else None."""
    import nav.marble_dynamics.search as S
    from nav.marble_dynamics.ballistic import NoInput
    if not callable(actions):
        return [list(a) for a in actions], 0, None
    kind = type(actions)
    if kind is NoInput:
        return [actions(None, i) for i in range(count)], 0, None
    if kind is S.BrakeContinuation:
        return [list(a) for a in actions.prefix], 2, np.asarray(actions.gem, dtype=float)
    from nav.marble_dynamics.landing import LandingContinuation as SupportLanding
    if kind is SupportLanding:
        # the fixed prefix, then Proposal(goal) (no braking) toward the sampled support point
        return [list(a) for a in actions.prefix], 3, np.asarray(actions.goal, dtype=float)
    from nav.marble_dynamics.settle import SettleProposal
    from nav.marble_dynamics.recharge import RechargeProposal
    from nav.marble_dynamics.phased_search import AimProposal
    brake, aim = 0, None
    if kind is S.LandingContinuation:
        prefix, spec = [list(a) for a in actions.prefix], actions.follow
    elif kind is S.SequenceProposal:
        if getattr(actions, 'land_short', False):
            return None   # 2026-10-26 search.LAND_SHORT (under test): not ported, the per-reply forecast runs it
        prefix, spec = [], actions
    elif kind is AimProposal:
        prefix, spec, aim = [], actions, float(actions.aim_offset)
    elif kind is SettleProposal or kind is RechargeProposal:
        # brake toward the gem (Proposal(gem, brake=True), index 0), then the sequence at index - prefix
        prefix, spec = [], actions.after
        brake = int(actions.prefix if kind is SettleProposal else actions.wait_replies)
        if not np.array_equal(np.asarray(actions.gem, dtype=float)[:3], np.asarray(spec.gem, dtype=float)[:3]):
            return None
    else:
        return None
    if spec.route.field is None or spec.route.key != _goal_key(spec.gem):
        return None
    if spec.through is not None:
        nr = spec.next_route
        if nr.field is None or nr.key != _goal_key(spec.through[1]) or nr.terrain is not spec.route.terrain:
            return None
    spec._native_brake, spec._native_aim = brake, aim
    return prefix, 1, spec


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


def predict_native(sim, engine, dynamics, actions, stop_on_ambiguity=False):
    count = len(actions) if not callable(actions) else int(actions.steps)
    plan = _controller(sim, actions, count)
    if plan is None or getattr(engine, 'native_per_reply', False):
        return _predict_per_reply(sim, engine, dynamics, actions, stop_on_ambiguity)
    import time
    import nav.marble_dynamics.search as S
    import nav.marble_dynamics.gem_demo as GD
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    prefix, tail, spec = plan
    native = dynamics['native']; state = clone_state(native['state'])
    if state['oob'] or state['mode'] != 0:
        raise ValueError('Playable non-OOB state required')
    blast0 = copy.deepcopy(dynamics['blast']) if dynamics.get('blast') is not None else None
    inventory = copy.deepcopy(dynamics['inventory']) if dynamics.get('inventory') is not None else None
    if blast0 is not None:
        state['blast_context'] = blast0
    if inventory is not None:
        state['inventory_context'] = inventory
    pend_move = native['pending_move']
    bridge = np.zeros(18)
    _move_array(pend_move, bridge[0:6]); bridge[6:15] = dynamics['bridge_action']; bridge[15] = int(dynamics['pow_yaw_hold'])
    bridge[16] = float(dynamics['pow_yaw']); bridge[17] = int(native['frame_clock_ms'])
    n_prefix = min(len(prefix), count)
    pre = np.ascontiguousarray(np.asarray(prefix[:n_prefix], dtype=float).reshape(-1, 9) if n_prefix else np.zeros((1, 9)))
    th = None; field = nfield = None; q = np.zeros(Q_SIZE)
    if tail == 1:
        q = seq_array(spec); field = np.ascontiguousarray(spec.route.field, dtype=np.float32)
        q[37] = spec._native_brake
        if spec._native_aim is not None:
            q[38] = 1; q[39] = spec._native_aim
        nfield = np.ascontiguousarray(spec.next_route.field, dtype=np.float32) if spec.through is not None else None
        th = sim.terrain(spec.route.terrain)
    elif tail == 2 or tail == 3:
        q[7:10] = spec[:3]
    consts = np.asarray([GD.TARGET_SPEED_CAP, GD.BRAKE_DECEL, GD.THROTTLE_GAIN, GD.THROTTLE_CAP, S.PICKUP_MIN_SPEED,
                         S.THROUGH_SWITCH_U, 1. if S.ROUTE_SPEED_LAW else 0., float(S.OPEN_TOP or 0.), float(S.OPEN_NEAR_U),
                         float(S.OPEN_FAR_U), float(S.TRAVEL_TOP or 0.), float(S.TRAVEL_NEAR_U), float(S.TRAVEL_FAR_U)], dtype=float)   # ABI 7
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
    rows = np.empty((4 * count + 1, S_SIZE)); replies = np.zeros((max(count, 1), R_SIZE)); bridge_out = np.zeros(1); blast_out = np.zeros(13)
    status = (ctypes.c_int * 8)()
    start_row = np.empty(S_SIZE); sim.pack(state, start_row)
    sim.lib.ms_forecast(sim.handle, th, _ptr(start_row), _ptr(bridge), _ptr(_blast_array(blast0)), _ptr(_inventory_array(inventory)),
                        count, 1 if stop_on_ambiguity else 0, n_prefix, _ptr(pre), tail, _ptr(q),
                        _ptr(field) if field is not None else None, _ptr(nfield) if nfield is not None else None, _ptr(consts),
                        budget_s, max_steps, _ptr(rows), _ptr(replies), _ptr(bridge_out), _ptr(blast_out), status)
    done_replies, nrows, fail_reply, fail_code, stopped, budget, steps = (status[k] for k in range(7))
    hook = getattr(engine, 'native_step_hook', None)
    if hook is not None:
        hook(steps)
    if tail == 1:
        _seq_writeback(spec, q)
    if budget:
        raise PlanningBudgetExceeded('Declared planning wall-time budget exceeded')
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
        state0 = clone_state(state)
        native_rows = NativeRows(rows[:nrows], state0)
        states = ReplyStates(native_rows, idx, state0, tab[:n_done, 36:38].copy() if blast0 is not None else None)
        builder = _ControlsBuilder(tab.copy(), pend_move, n_done, fail_reply)
        ambiguous = [i for i in range(n_done) if bool((tab[i, 32:32 + int(done[i])] != 0).any())]
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
                    bridge_resume=bridge_resume,
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
        st = sim.unpack(rows[row - 1], template)
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
                assumptions='Measured local16ms physics/64ms decisions (compiled forecast nav/native), static geometry, '
                            'ordinary movement/jump and native powers1-5, observed regular/item-charged Blast.')
