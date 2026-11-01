"""Whole Hunt rounds in the offline simulator (2026-10-29, phase 1 of docs/IMPROVEMENT_LOOP_PLAN_2026-10-29.md).

The real ContinuousController plays a complete round against the engine port: the opening is the recorded round's GO
state (same first group, so a sim round pairs with its native round), every later group is drawn by the ported hunt spawn
rules (tools/sim_bench/hunt_world.py) from a spawn seed, powerups respawn 7 s after a pickup, a fall respawns the marble at
the spawn trigger the game would pick (then the planner is restarted as after the pools' legal OOB click), the pools' 8 s
no-progress rescue applies, and the round ends when the clock reaches zero. Deterministic: virtual search clock
(sim_world), fixed spawn seed.

usage (repo root): PYTHONPATH=ml_agent python ml_agent/tools/sim_bench/full_round.py --pool overnight_20261026/qr3_explore_baseline
          --seed 482017593 --map Sprawl_Hunt --spawn-seed 1 --out rounds.jsonl [--cap-s 60] [--set module.NAME=value] [--trace]
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_world as W
import hunt_world as H

from nav.protocol import (RAW_GEMS, RAW_SCORE, RAW_GEMS_REMAINING, RAW_POW_ITEMS, RAW_TIME_LEFT, RAW_TIME_REMAINING, RAW_SPIN,
                          RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL, RAW_POW_MEGA, RAW_POW_MEGA_LEFT)

DECISION_MS = 64
RESCUE_WINDOW_S = 8.


def go_row(rows, round_ms):
    """The first playable decision at or after GO (the countdown rows have the full round time left and a neutral label)."""
    for r in rows:
        st = r['before']['native']['state']
        if st.get('mode', 0) != 0 or st.get('oob') or r['choice'].get('label') == 'countdown_neutral':
            continue
        if float(r['raw_observation'][RAW_TIME_LEFT]) > round_ms:
            continue
        return r
    raise ValueError('no playable GO decision')


class HuntWorld:
    """Items, clock, spawner and respawns of one simulated round."""

    def __init__(self, mission, start_row, spawn_seed, round_ms):
        self.m = mission
        self.spawner = H.HuntSpawner(mission, spawn_seed)
        self.rng = np.random.default_rng(spawn_seed + 7919)
        d0 = start_row['before']
        self.clock_ms = int(d0['native']['frame_clock_ms'])
        self.time_left_ms = int(float(start_row['raw_observation'][RAW_TIME_LEFT]))
        self.round_ms = round_ms
        self.powerups = [copy.deepcopy(it) for it in d0.get('items', []) if not it.get('gem')]   # exact game boxes
        self.pow_respawn = {}   # item id -> clock_ms when it is back
        # the recorded first group, mapped onto the mission's gem indices
        self.gems = []
        for it in d0.get('items', []):
            if it.get('gem'):
                k = int(np.argmin(np.linalg.norm(mission.gem_positions - np.asarray(it['position']), axis=1)))
                if np.linalg.norm(mission.gem_positions[k] - np.asarray(it['position'])) > 0.25:
                    raise ValueError('recorded gem not in the mission: %s' % it['position'])
                g = copy.deepcopy(it); g['index'] = k; self.gems.append(g)
        if not self.gems:
            raise ValueError('no gems at GO')
        # the recorded group's center is unknown: the gem whose farthest group mate is nearest (the spawn radius rule)
        P = np.asarray([mission.gem_positions[g['index']] for g in self.gems])
        self.spawner.last_center = self.gems[int(np.argmin(np.max(np.linalg.norm(P[:, None] - P[None], axis=2), axis=1)))]['index']
        self.last_pick_index = None
        self.points = 0.; self.picks = []; self.spawns = 0; self.falls = 0; self.rescues = 0
        self.last_trigger = None; self.last_trigger_ms = -10 ** 9
        self.gems_remaining0 = float(start_row['raw_observation'][RAW_GEMS_REMAINING]); self.collected = 0

    def items(self):
        return self.gems + [p for p in self.powerups if p['id'] not in self.pow_respawn]

    def advance_items(self, roll, reply_clock_ms):
        """Apply one reply's pickups: points, gems removed, powerups scheduled to respawn, respawned powerups restored."""
        import nav.marble_dynamics.gem_demo as GD
        from tools.sim_bench.world_events import consumed_powerups
        gained = 0.; keep = []; hits = []
        for g in self.gems:
            k = GD.tick_pickup(roll['native_states'], g)
            if k is not None:
                hits.append((k, g))
            else:
                keep.append(g)
        for k, g in sorted(hits, key=lambda x: x[0]):
            gained += float(g.get('value') or 1.); self.collected += 1
            self.picks.append(dict(t_ms=self.round_ms - self.time_left_ms + k * DECISION_MS, value=float(g.get('value') or 1.), index=g['index']))
            self.last_pick_index = g['index']
        self.gems = keep; self.points += gained
        for pid in consumed_powerups(roll):
            self.pow_respawn[pid] = reply_clock_ms + H.POW_RESPAWN_MS
        for pid in [p for p, t in self.pow_respawn.items() if t <= reply_clock_ms + DECISION_MS]:
            del self.pow_respawn[pid]
        return gained

    def maybe_spawn(self):
        if self.gems:
            return False
        for g, item in self.spawner.spawn_items(exclude=self.last_pick_index):
            item['index'] = g; self.gems.append(item)
        self.spawns += 1
        return True

    def respawn(self, dynamics, fall_position, last_touch, clock_ms):
        since = (clock_ms - self.last_trigger_ms) / 1000.
        vis = [(g['position'][0], g['position'][1], g['position'][2], g.get('value') or 1.) for g in self.gems]
        k, pos = H.respawn_point(self.m, self.rng, fall_position, vis, last_touch, self.last_trigger, since)
        self.last_trigger = k; self.last_trigger_ms = clock_ms
        return H.respawned_dynamics(dynamics, pos, clock_ms)

    def raw(self, template, dynamics):
        """The observer's raw vector for the current world (observer.cs: 5 nearest gems, 3 nearest powerup items with
        their respawn timers, the clock, score and the marble's own state)."""
        st = dynamics['native']['state']; p = np.asarray(st['position'], dtype=float)
        raw = template.copy()
        raw[0:3] = st['position']; raw[3:6] = st['velocity']; raw[RAW_SPIN] = st['spin']
        slots = []
        for g in self.gems:
            rel = np.asarray(g['position'], dtype=float) - p
            slots.append([rel[0], rel[1], rel[2], float(g.get('value') or 1.), float(np.linalg.norm(rel))])
        slots.sort(key=lambda s: s[4]); slots = slots[:5]
        slots += [[-999.] * 5] * (5 - len(slots))
        raw[RAW_GEMS] = np.asarray(slots, dtype=np.float32).ravel()
        pw = []
        for it in self.powerups:
            rel = np.asarray(it['position'], dtype=float) - p
            left = max(0., (self.pow_respawn.get(it['id'], self.clock_ms) - self.clock_ms) / 1000.)
            pw.append((float(np.linalg.norm(rel)), [float(it['powerup']), rel[0], rel[1], rel[2], left]))
        pw.sort(key=lambda s: s[0]); pw = [s[1] for s in pw[:3]]
        pw += [[-999.] * 5] * (3 - len(pw))
        raw[RAW_POW_ITEMS] = np.asarray(pw, dtype=np.float32).ravel()
        raw[RAW_TIME_LEFT] = self.time_left_ms; raw[RAW_TIME_REMAINING] = self.round_ms - self.time_left_ms
        raw[RAW_SCORE] = self.points; raw[RAW_GEMS_REMAINING] = max(0., self.gems_remaining0 - self.collected)
        inv = dynamics.get('inventory') or {}
        raw[RAW_POW_HELD] = 6 if inv.get('held_type') == 6 else int(st.get('held', 0))
        b = dynamics.get('blast') or st.get('blast_context')
        if b is not None:
            raw[RAW_POW_BLAST] = float(b.get('value', 0.)); raw[RAW_POW_SPECIAL] = int(bool(b.get('special')))
        raw[RAW_POW_MEGA] = int(bool(inv.get('mega_active'))); raw[RAW_POW_MEGA_LEFT] = float(inv.get('mega_left_ms', 0) or 0) / 1000.
        raw[43:46] = [st['effects'][i][1] / 1000. if st['effects'][i][0] else 0. for i in (3, 4, 5)]
        return raw


def play_round(a, terrain, engine, mission, rows, spawn_seed, o, label='', start=None, max_decisions=None, root_at=None,
               root_actions=None, record_root=False, future_seed=None):
    """A whole round from the recorded GO state, or (branch benchmark) a segment: `start` = any playable recorded row,
    `max_decisions` the segment length, `root_at` the decision index of the root choice; with `record_root` every search of
    that decision is recorded (candidate list), with `root_actions` the root decision and the following ones execute those
    bridge actions open loop (the branch) before the controller takes over again."""
    import nav.marble_dynamics.reliability_eval as RE
    from nav.recovery import ProgressMonitor
    from nav.routing import goal_key
    from nav.gems import visible_gems
    from nav.marble_dynamics.fast_nav import fully_outside_inbounds
    from tools.sim_bench.world_events import advance_world
    start = start if start is not None else go_row(rows, mission.round_ms)
    world = HuntWorld(mission, start, spawn_seed, mission.round_ms)
    d = copy.deepcopy(start['before']); template = np.asarray(start['raw_observation'], dtype=np.float32)
    ctrl = RE.ContinuousController(terrain, engine, a.checkpoint, a, 'physics_plan')
    if getattr(engine, 'parallel', None) is not None:
        engine.parallel.bind(dict(terrain=terrain, geometry=a.geometry_terrain, mesh=engine.mesh, sim=engine.native_sim,
                                  model=ctrl.nav.model, crossings=engine.crossings, contact_order=getattr(engine, 'contact_order', None)))
    monitor = ProgressMonitor(RESCUE_WINDOW_S)
    labels = Counter(); why = 'round_end'; t = 0; failures = 0; blasts = 0; uses = Counter(); speeds = []
    cap = int(o.cap_s / .064) if o.cap_s else 10 ** 9
    if max_decisions:
        cap = min(cap, int(max_decisions))
    root_points = None; root_picks = None; recorded = []; open_loop = list(root_actions or []); root_context = None
    last_touch = list(d['native']['state']['position']); exited_before = False
    d['items'] = world.items()
    raw = world.raw(template, d); msg = W.Msg(raw, d)
    t0 = time.process_time(); s0 = W.STEPS[0]
    while world.time_left_ms > 0 and t < cap:
        before = raw.copy(); remaining = max(1, int(world.time_left_ms / DECISION_MS))
        if root_at is not None and t == root_at:
            root_points = world.points; root_picks = len(world.picks); root_t_ms = world.round_ms - world.time_left_ms
            # the root protocol, the same for every arm: the plan in progress is dropped so the root decision is a fresh
            # search (a kept plan runs no search, so there would be no candidate list to branch from)
            ctrl.invalidate('pilot_root', preserve_task=True)
            _st = d['native']['state']; _inv = d.get('inventory') or {}; _b = d.get('blast') or {}
            root_context = dict(position=[float(v) for v in _st['position']], velocity=[float(v) for v in _st['velocity']],
                                held=int(_st.get('held', 0)), held_type=int(_inv.get('held_type', 0) or 0), blast=float(_b.get('value', 0.)),
                                restore_type=int(_inv.get('restore_type', 0) or 0), time_left_ms=int(world.time_left_ms),
                                effects=[[bool(e[0]), int(e[1])] for e in _st.get('effects', [])],
                                gems=[[float(v) for v in g['position']] + [float(g.get('value') or 1.)] for g in world.gems],
                                powerups=[[int(it['powerup'])] + [float(v) for v in it['position']] for it in world.items() if not it.get('gem')])
            root_falls = world.falls; root_rescues = world.rescues
            if future_seed is not None:   # the futures of the branch benchmark: common prefix, future-specific spawns from the root on
                world.spawner.rng = np.random.default_rng(future_seed); world.rng = np.random.default_rng(future_seed + 7919)
        if root_at is not None and t >= root_at and open_loop:
            # the branch: the candidate's own replies, open loop, the controller observing (its memory stays consistent)
            action = list(open_loop.pop(0)); info = dict(label='branch_open_loop', reason=None)
            if not open_loop:
                ctrl.invalidate('branch_handoff', preserve_task=True)
        elif record_root and root_at is not None and t == root_at:
            import nav.marble_dynamics.reliability_eval as RE
            real = RE.search

            def recorder(engine_, route_, dynamics_, raw_, gem_, *args, **kw):
                chosen, pairs = real(engine_, route_, dynamics_, raw_, gem_, *args, **kw)
                recorded.append(dict(gem=[float(v) for v in np.asarray(gem_)[:3]], target_item=kw.get('target_item'), pairs=list(pairs),
                                     chosen=chosen, route=route_, dynamics=copy.deepcopy(dynamics_), raw=np.array(raw_, copy=True)))
                return chosen, pairs
            RE.search = recorder
            try:
                action, info = ctrl.action(msg, remaining)
            finally:
                RE.search = real
        else:
            action, info = ctrl.action(msg, remaining)
        labels[info.get('label') or '?'] += 1
        if action[6] or action[8]:
            uses[int(before[RAW_POW_HELD])] += 1
        engine.planning_deadline = None
        try:
            roll = advance_world(engine, d, action)
        except ValueError as exc:
            why = 'sim_error:' + str(exc)[:60]; break
        if roll['failure']:
            failures += 1; why = 'sim_failure'; break
        blasts += sum(1 for c in roll.get('controls', []) for e in c.get('script_events', []) if e.get('kind') == 'blast_impulse')
        nd = W.resume(d, roll); st = nd['native']['state']
        world.clock_ms += DECISION_MS; world.time_left_ms -= DECISION_MS
        gained = world.advance_items(roll, world.clock_ms)
        if st.get('ground_ms', 0) > 0:
            last_touch = list(st['position'])
        # the game's InBounds trigger polls, so its OOB flag lags the geometric exit by 0-2 decisions (43 recorded falls of
        # 10-29: 12 / 21 / 9 / 1 at lags 0 / 1 / 2 / 3); the sim declares the fall one decision after the exit
        exited = bool(engine.mission_bounds is not None and fully_outside_inbounds(st, engine.mission_bounds))
        fell = exited and exited_before; exited_before = exited
        ended = world.time_left_ms <= 0
        speeds.append(float(np.linalg.norm(np.asarray(st['velocity'])[:2])))
        if o.trace:
            print('   t %6.2f %-28s %-22s v %5.1f pos %s gems %d pts %g%s' % ((world.round_ms - world.time_left_ms) / 1000., str(info.get('label'))[:28],
                  str(info.get('reason'))[:22], speeds[-1], np.round(st['position'], 1).tolist(), len(world.gems), world.points, ' FELL' if fell else ''), flush=True)
        spawned = (not fell) and world.maybe_spawn()
        nd['items'] = world.items()
        raw = world.raw(template, nd)
        outcome = dict(fell=fell, gem_delta=gained, round_ended=ended, reconnected=False)
        msg = W.Msg(raw, nd); ctrl.observe(before, msg, outcome); d = nd; t += 1
        if ended:
            break
        if fell:
            # the game: one decision with the marble out of bounds (the planner answers neutral), then the respawn
            world.falls += 1
            ctrl.invalidate('legal_oob_click', reset_actor=True)
            st['oob'] = True
            try:
                ctrl.action(W.Msg(raw, nd), remaining)
            except Exception:
                pass
            world.clock_ms += DECISION_MS; world.time_left_ms -= DECISION_MS; t += 1
            d = world.respawn(nd, st['position'], last_touch, world.clock_ms); exited_before = False
            d['items'] = world.items(); raw = world.raw(template, d); msg = W.Msg(raw, d)
            monitor.rescued((world.round_ms - world.time_left_ms) / 1000., float('inf')); monitor.rescues -= 1
            continue
        now_s = (world.round_ms - world.time_left_ms) / 1000.
        target = ctrl.task_goal; distance = float('inf')
        if target is not None:
            ctrl.progress_route.navigate(raw[:3], target[:3]); distance = ctrl.progress_route.distance(raw[:3])
        idle = monitor.observe(now_s, goal_key(target) if target is not None else None, distance, picked=gained, fell=False,
                               gems=[goal_key(g) for g in visible_gems(raw)], groups_completed=world.spawns)
        if idle >= monitor.window_s - 1e-9 and visible_gems(raw):
            world.rescues += 1
            ctrl.invalidate('legal_stuck_rescue', reset_actor=True)
            d = world.respawn(d, d['native']['state']['position'], last_touch, world.clock_ms)
            d['items'] = world.items(); raw = world.raw(template, d); msg = W.Msg(raw, d)
            monitor.rescued(now_s, float('inf'))
    after = None
    if root_points is not None:
        after = dict(points=world.points - root_points, picks=[p for p in world.picks[root_picks:]], decisions=t - root_at,
                     root_t_ms=root_t_ms, falls=world.falls - root_falls, rescues=world.rescues - root_rescues)
    return dict(label=label, map=mission.name, spawn_seed=spawn_seed, points=world.points, picks=world.picks, spawns=world.spawns,
                after_root=after, root_candidates=recorded, root_context=root_context,
                falls=world.falls, rescues=world.rescues, decisions=t, played_s=(world.round_ms - max(0, world.time_left_ms)) / 1000.,
                why=why, failures=failures, blasts=blasts, uses=dict(uses), labels=dict(labels.most_common(12)),
                mean_speed=float(np.mean(speeds)) if speeds else 0., cpu_s=time.process_time() - t0, steps=W.STEPS[0] - s0,
                fallback=int(ctrl.counts.get('fallback_decisions', 0)), safe_holds=int(ctrl.counts.get('safe_holds', 0)),
                searches=int(ctrl.counts.get('full_searches', 0)), stalled_s=float(monitor.stalled_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--seed', required=True); ap.add_argument('--map', required=True)
    ap.add_argument('--spawn-seed', type=int, nargs='+', default=[1])
    ap.add_argument('--out', required=True); ap.add_argument('--label', default='base')
    ap.add_argument('--cap-s', type=float, default=0.)
    ap.add_argument('--set', action='append', default=[]); ap.add_argument('--pflag', action='append', default=[])
    ap.add_argument('--module', action='append', default=[])
    ap.add_argument('--budget', type=float, default=2.)
    ap.add_argument('--native-step-us', type=float, default=0.)
    ap.add_argument('--legacy-flags', action='store_true'); ap.add_argument('--trace', action='store_true')
    ap.add_argument('--normal-priority', action='store_true')
    o = ap.parse_args()
    if not o.normal_priority:
        import ctypes
        _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p
        _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; _k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
    for spec in o.module:
        import importlib.util
        name, path = spec.split('=', 1); full = 'nav.marble_dynamics.' + name
        s = importlib.util.spec_from_file_location(full, path); mod = importlib.util.module_from_spec(s)
        sys.modules[full] = mod; s.loader.exec_module(mod)
    W.install_virtual_clock()
    W.apply_settings(o.set)
    a = W.planner_args(o.map, budget_s=o.budget, pflags=o.pflag, legacy=o.legacy_flags)
    terrain, engine = W.setup(a, o.map, native=True, native_step_us=o.native_step_us)
    W.apply_settings(o.set)
    mission = H.MissionHunt(o.map)
    rows = W.load_journal(W.journal_path(o.pool, o.map, o.seed))
    with open(o.out, 'a', encoding='utf-8') as out:
        for ss in o.spawn_seed:
            res = play_round(a, terrain, engine, mission, rows, ss, o, label=o.label)
            res.update(pool=o.pool, seed=o.seed, settings=o.set, budget=o.budget)
            out.write(json.dumps(res) + '\n'); out.flush()
            print('%s %-18s spawn %d -> %5.1f pts in %5.1f s | picks %3d spawns %2d falls %d rescues %d | fb %d holds %d speed %.1f | %s cpu %.0fs' % (
                o.label, o.map[:18], ss, res['points'], res['played_s'], len(res['picks']), res['spawns'], res['falls'], res['rescues'],
                res['fallback'], res['safe_holds'], res['mean_speed'], res['why'], res['cpu_s']), flush=True)


if __name__ == '__main__':
    main()
