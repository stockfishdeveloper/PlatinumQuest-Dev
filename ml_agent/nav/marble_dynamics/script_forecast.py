"""Tick-ordered item events around the compiled native marble mechanics.

The ordinary forecast stays fast. Only a window that touches a visible powerup
needs the event path: pickup can replace inventory or be followed by use within
the same reply. No future item spawns or native observations are consulted.
"""
import numpy as np

from nav.marble_dynamics.state_copy import clone
from nav.marble_dynamics.bridge import predict_replies, translated, observe_proposal_ticks
from nav.marble_dynamics.blast import event as blast_event, advance as advance_blast
from nav.marble_dynamics.inventory import check as check_inventory
from nav.marble_dynamics.rollout import apply_control
from nav.marble_dynamics.planning_budget import check as check_budget
from nav.marble_dynamics.gem_demo import possible_pickup


def held_type(state):
    inv = state.get('inventory_context') or {}
    return 6 if inv.get('held_type') == 6 else int(state['held'])


def _touches(a, b, item):
    p, q = np.asarray(a['position']), np.asarray(b['position'])
    bounds = np.asarray(item['bounds'])
    reach = float(b['radius']) + .2
    return bool(np.all(np.maximum(p, q) + reach >= bounds[:3])
                and np.all(np.minimum(p, q) - reach <= bounds[3:]))


def _collect(state, previous, items, tick):
    """Script onPickup after one physics tick. Return remaining items and events.

    A different powerup REPLACES the held one; an identical type stays visible.
    Simultaneous different inventory pickups need native container query order,
    which the snapshot does not expose: refuse that ambiguous transition.
    """
    hits = [item for item in items if _touches(previous, state, item)]
    powers = [item for item in hits if not item['gem'] and 1 <= int(item['powerup']) <= 6]
    # A held-type item may become eligible AFTER another item replaces it.
    # Even same-type duplicates need query order to identify the consumed item.
    if len(powers) > 1 and any(int(item['powerup']) != held_type(state) for item in powers):
        raise ValueError('Simultaneous inventory pickups need native query order')
    events = []
    consumed = set()
    for item in hits:
        kind = int(item['powerup'])
        inv = state.get('inventory_context')
        if item['gem']:
            pass
        elif 1 <= kind <= 6:
            if held_type(state) == kind:
                continue
            state['held'] = kind if kind <= 5 else 0
            if inv is None and kind == 6:
                raise ValueError('Mega pickup requires observed script inventory context')
            if inv is not None:
                inv['held_type'] = kind
        elif kind == 7:
            blast = state.get('blast_context')
            if blast is None:
                raise ValueError('Blast pickup requires observed Blast context')
            if not blast['enabled']:
                continue
            blast['value'] = 1.
            blast['special'] = True
        elif kind > 7:
            raise ValueError('Unmodeled script powerup pickup: %s' % kind)
        else:
            continue
        consumed.add(id(item))
        events.append(dict(item=item, tick=tick, interval=(tick-1)//4))
    return [item for item in items if id(item) not in consumed], events


def _interior_below(mesh, position):
    start = np.asarray(position, dtype=float)
    end = start - [0., 0., 1.]
    for _ in range(32):
        hit = mesh.raycast(start, end, cull_backfaces=False)
        if hit is None:
            return False
        if mesh.objects[hit['object_id']]['class'] == 'InteriorInstance':
            return True
        start = np.asarray(hit['point']) - [0., 0., 1e-5]
        if start[2] <= end[2]:
            return False
    raise ValueError('Mega ground ray exceeded bounded static intersections')


def _mega_prepare(state, clock):
    from nav.native.marble_sim import PARAM_KEYS
    inv = state.get('inventory_context')
    if inv is None:
        raise ValueError('Mega requires observed inventory context')
    if inv.get('permanent_mega'):
        raise ValueError('Timed Mega use on a permanently Mega map is unverified')
    for key in ('normal_parameters', 'mega_parameters'):
        profile = inv.get(key)
        if not isinstance(profile, dict) or set(PARAM_KEYS)-set(profile):
            raise ValueError('Mega requires both observed physics profiles')
        inv[key] = {k:float(np.float32(profile[k])) for k in PARAM_KEYS}
    if inv.get('mega_active') and '_mega_end' not in inv:
        if not inv.get('mega_timer_known'):
            raise ValueError('Timed Mega expiry is unknown')
        # The snapshot's script clock can lag Platform's next physics clock by
        # a few milliseconds. getEventTimeLeft uses the former (recorded on
        # Concentric: a 2 ms offset otherwise postpones shrink by a whole tick).
        inv['_mega_end'] = int(state['clock_ms']) + int(inv['mega_left_ms'])
    return inv


def _mega_begin(engine, state, pending, move, clock):
    inv = _mega_prepare(state, clock)
    latched = inv.get('use_latched', bool(move.get('use')))
    if pending[6] and inv['held_type'] == 6 and not latched:
        if inv.get('mega_radius', 0.) <= 0 or inv.get('mega_duration_ms', 0) <= 0:
            raise ValueError('Mega radius or duration unavailable')
        if not inv['mega_active']:
            inv['_mega_growth'] = clock+16
            if not np.allclose(state['gravity_direction'], [0., 0., -1.], atol=1e-7, rtol=0):
                raise ValueError('Mega impulse with rotated gravity is unverified')
            if _interior_below(engine.mesh, state['position']):
                inv['_mega_rise'] = clock+32
        inv['held_type'] = 0
        state['held'] = 0
        inv['mega_active'] = True
        inv['mega_timer_known'] = True
        inv['_mega_end'] = clock+int(inv['mega_duration_ms'])
    inv['use_latched'] = bool(pending[6])


def _mega_tick(state, clock):
    inv = state['inventory_context']
    if inv.get('_mega_growth', float('inf')) <= clock:
        state['radius'] = float(np.float32(inv['mega_radius']))
        state['parameters'] = dict(state['parameters'], **inv['mega_parameters'])
        del inv['_mega_growth']
    if inv.get('_mega_end', float('inf')) <= clock:
        state['radius'] = float(np.float32(inv['normal_radius']))
        state['parameters'] = dict(state['parameters'], **inv['normal_parameters'])
        inv['mega_active'] = False
        inv['mega_timer_known'] = False
        inv['_mega_shrink_impulse'] = clock+32
        del inv['_mega_end']
    for key, amount in (('_mega_rise', 6.), ('_mega_shrink_impulse', -2.)):
        if inv.get(key, float('inf')) <= clock:
            # Native applyImpulse converts the entire velocity to Point3F.
            delta = np.asarray([0., 0., amount], np.float32) / np.float32(state['parameters']['mass'])
            state['velocity'] = (np.asarray(state['velocity'], np.float32) + delta).astype(float).tolist()
            del inv[key]
    inv['mega_left_ms'] = max(0, int(inv.get('_mega_end', clock))-clock)


def _restore_tick(state, clock):
    inv = state.get('inventory_context')
    if not inv or '_restore_end' not in inv:
        return
    if clock >= inv['_restore_end']:
        kind = int(inv['restore_type'])
        inv['held_type'] = kind
        state['held'] = kind if kind <= 5 else 0
        inv['restore_type'] = 0; inv['restore_left_ms'] = -1
        del inv['_restore_end']
    else:
        inv['restore_left_ms'] = max(0, inv['_restore_end']-16-clock)


def predict_events(engine, dynamics, actions, *, collect_items=True, model_mega=False):
    """Same bridge/control timing as predict_small_step, with visible item events.

    Uses the compiled *physics step* and copies only the bounded state window.
    Script inventory and Blast changes happen before the next tick can use them.
    """
    if getattr(engine, 'bridge_physics_ms', 64) != 16:
        raise ValueError('Item event forecast requires verified 16 ms physics')
    sim = getattr(engine, 'native_sim', None)
    if sim is None:
        raise ValueError('Item event forecast requires compiled native physics')
    native = dynamics['native']
    state = clone(native['state'])
    if state['oob'] or state['mode'] != 0:
        raise ValueError('Playable non-OOB state required')
    for key, source in (('blast_context', 'blast'), ('inventory_context', 'inventory')):
        if dynamics.get(source) is not None:
            state[key] = clone(dynamics[source])
    clock = int(native['frame_clock_ms'])
    inv = state.get('inventory_context') or {}
    if inv.get('restore_type', 0) and inv.get('restore_left_ms', -1) >= 0 and '_restore_end' not in inv:
        # Saved inventory is a scheduled server callback; its native ghost
        # changes on the following frame (measured legal quick respawn).
        inv['_restore_end'] = int(state['clock_ms'])+int(inv['restore_left_ms'])+16
    move = clone(native['pending_move'])
    pending = list(dynamics['bridge_action'])
    hold, aim = int(dynamics['pow_yaw_hold']), float(dynamics['pow_yaw'])
    items = list(dynamics.get('items', [])) if collect_items else []
    count = len(actions) if not callable(actions) else int(actions.steps)
    states = [clone(state)]; ticks = [clone(state)]
    controls = []; native_controls = []; events = []; proposed = []; pickups = []
    failure = None
    for i in range(count):
        check_budget(engine)
        action = actions(state, i) if callable(actions) else actions[i]
        proposed.append(list(action))
        next_move, camera, hold, aim = translated(pending, hold, aim)
        frames = []; macro_events = []
        try:
            inv = state.get('inventory_context') or {}
            mega = model_mega and (inv.get('held_type') == 6 or inv.get('mega_active')
                                  or any(k.startswith('_mega_') for k in inv))
            if mega:
                _mega_begin(engine, state, pending, move, clock+i*64)
            else:
                check_inventory(inv or None, state, pending)
            for k in range(4):
                check_budget(engine)
                state['clock_ms'] = clock+i*64+k*16
                _restore_tick(state, state['clock_ms'])
                if mega:
                    _mega_tick(state, state['clock_ms'])
                control = dict(move=clone(move if k == 0 else next_move), events=camera if k == 0 else [])
                impulse = blast_event(state.get('blast_context'), state, bool(pending[8])) if k == 0 else None
                if impulse is not None:
                    control['script_events'] = [impulse]
                apply_control(state, control)
                previous = state
                state, ev = sim.step(state, control['move'], 16)
                advance_blast(state.get('blast_context'), impulse is not None, 16)
                inv = state.get('inventory_context')
                if inv is not None and inv.get('held_type') != 6:
                    inv['held_type'] = int(state['held'])
                if mega:
                    inv['mega_left_ms'] = max(0, int(inv.get('_mega_end', state['clock_ms']))-state['clock_ms']-16)
                items, collected = _collect(state, previous, items, i*4+k+1)
                pickups.extend(collected)
                frames.append(control); native_controls.append(control)
                ticks.append(clone(state)); macro_events.extend(ev)
        except ValueError as exc:
            failure = dict(step=i, reason=str(exc))
            break
        observe_proposal_ticks(actions,ticks[-len(frames)-1:])
        macro = dict(move=clone(frames[0]['move']), events=clone(frames[0]['events']), subframes=frames)
        script_events = [ev for frame in frames for ev in frame.get('script_events', [])]
        if script_events:
            macro['script_events'] = script_events
        states.append(clone(state)); controls.append(macro); events.append(macro_events)
        move = next_move; pending = list(action)
    return dict(states=states, controls=controls, events=events, proposed=proposed, failure=failure,
        native_states=ticks, native_controls=native_controls, items=items, item_events=pickups,
        time_s=[i*.064 for i in range(len(states))], ambiguous_intervals=[],
        ambiguous_accepted=[i for i, ev in enumerate(events) if any(e['kind'] == 'ambiguous_contact' for e in ev)],
        requested_replies=count, stopped_on_ambiguity=False, physics_frame_ms=16, decision_ms=64,
        assumptions='Compiled native physics plus tick-ordered visible pickups; no new item spawns')


def forecast_items(engine, dynamics, actions):
    """Fast ordinary forecast, replaying with events when a powerup is touched."""
    inv = dynamics.get('inventory') or {}
    if inv.get('restore_type', 0) and inv.get('restore_left_ms', -1) >= 0:
        return predict_events(engine, dynamics, actions, model_mega=getattr(engine, 'mega_enabled', False))
    if getattr(engine, 'mega_enabled', False) and (inv.get('held_type') == 6 or inv.get('mega_active')):
        return predict_events(engine, dynamics, actions, model_mega=True)
    result = predict_replies(engine, dynamics, actions, stop_on_ambiguity=False)
    if result['failure']:
        return result
    items = list(dynamics.get('items', []))
    contacts = []
    for item in items:
        k = possible_pickup(result['states'], item, result.get('native_states'))
        if k is not None:
            contacts.append((k, item))
    if any(not item['gem'] and int(item['powerup']) > 0 for k, item in contacts):
        return predict_events(engine, dynamics, actions, model_mega=getattr(engine, 'mega_enabled', False))
    consumed = {id(item) for k, item in contacts if item['gem']}
    result['items'] = [item for item in items if id(item) not in consumed]
    result['item_events'] = [dict(item=item, interval=k-1, tick=None) for k, item in contacts if item['gem']]
    return result
