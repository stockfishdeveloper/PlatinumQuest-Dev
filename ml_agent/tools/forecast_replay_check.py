"""Check look-ahead predictions against recorded native decision windows.

Windows include regular samples and every observed inventory transition. Actual
future actions are replayed only for validation, never supplied to live planning.
New gem groups are reported separately; respawn/handshake gaps are excluded by
native clock continuity. All prediction failures and disagreements are retained.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import atomic_json
from nav.marble_dynamics import realtime as rt
from nav.marble_dynamics.record import sha

TOLERANCES = dict(position=1e-4, velocity=1e-3, spin=1e-3, radius=1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool', type=Path)
    source.add_argument('--mega-traces', '--script-traces', type=Path, help='Replay native script-event trials using only each starting observation')
    p.add_argument('--maps', nargs='+')
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--stride', type=int, default=17)
    p.add_argument('--max-windows', type=int, default=400)
    p.add_argument('--event-model', action='store_true', help='Direct tick-ordered item forecast before live integration')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite prior evidence')
    if a.mega_traces:
        mega_replay(a)
        return
    if not a.maps:
        p.error('--maps is required with --pool')
    rt.SEARCH_WORKERS = 0
    all_maps = []
    for name in a.maps:
        world = rt.World(name)
        cases = []
        skipped = Counter()
        paths = sorted((a.pool / name).rglob('decisions.jsonl.gz'))
        if not paths:
            raise ValueError('No journal for ' + name)
        for path in paths:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]
            indices = set(range(0, len(rows)-a.horizon, a.stride))
            events = set()
            for i in range(1, len(rows)):
                x, y = rows[i-1]['before'], rows[i]['before']
                if (x.get('inventory') != y.get('inventory') or
                        x['native']['state']['held'] != y['native']['state']['held']):
                    events.update(range(max(0, i-a.horizon), min(i, len(rows)-a.horizon)))
            indices = sorted(events) + sorted(indices-events)
            for i in indices[:a.max_windows]:
                selected = rows[i:i+a.horizon+1]
                contexts = [r['before'] for r in selected]
                if any(d['native']['state']['mode'] != 0 or d['native']['state']['oob'] for d in contexts):
                    skipped['non_playable'] += 1
                    continue
                clocks = [d['native']['frame_clock_ms'] for d in contexts]
                if any(y-x != 64 for x, y in zip(clocks, clocks[1:])):
                    skipped['native_clock_gap'] += 1
                    continue
                actual = selected[-1]['before']['native']['state']
                start = rt.Msg(np.asarray(rows[i]['raw_observation'], np.float32), contexts[0], rows[i]['tick'])
                case = dict(decision=rows[i]['decision'], journal=str(path), inventory_event=i in events)
                t = time.perf_counter()
                try:
                    actions = [r['action'] for r in selected[:-1]]
                    if a.event_model:
                        from nav.marble_dynamics.script_forecast import forecast_items
                        from nav.marble_dynamics.viability import resume_forecast
                        forecast = forecast_items(world.engine, contexts[0], actions)
                        if forecast['failure']:
                            raise ValueError(str(forecast['failure']))
                        got = rt.Msg(start.obs, resume_forecast(contexts[0], forecast))
                        pickups = {}
                    else:
                        got, pickups = rt.predict(world.engine, start, actions)
                    state = got.dynamics['native']['state']
                    errors = {k:float(np.max(np.abs(np.asarray(state[k])-np.asarray(actual[k]))))
                              for k in ('position', 'velocity', 'spin', 'radius')}
                    held = lambda d: 6 if d.get('inventory', {}).get('held_type') == 6 else int(d['native']['state']['held'])
                    case.update(errors=errors, held_match=held(got.dynamics) == held(contexts[-1]),
                                predicted_held=held(got.dynamics), actual_held=held(contexts[-1]), pickups=pickups,
                                strict_physics_match=max(errors.values()) <= 1e-6,
                                passed=all(errors[k] <= limit for k, limit in TOLERANCES.items())
                                and held(got.dynamics) == held(contexts[-1]))
                except ValueError as exc:
                    case.update(passed=False, failure=str(exc))
                case['wall_s'] = time.perf_counter()-t
                cases.append(case)
        world.close()
        result = dict(map=name, cases=cases, windows=len(cases), passed=sum(c['passed'] for c in cases),
                      inventory_windows=sum(c['inventory_event'] for c in cases), skipped=dict(skipped))
        all_maps.append(result)
        print(json.dumps({k:v for k,v in result.items() if k != 'cases'}), flush=True)
        atomic_json(a.output, dict(maps=all_maps, complete=False))
    atomic_json(a.output, dict(maps=all_maps, complete=True,
        realtime_sha256=sha(Path(rt.__file__)), pool=str(a.pool), horizon=a.horizon, stride=a.stride,
        max_windows=a.max_windows, event_model=a.event_model, tolerances=TOLERANCES,
        contract='Recorded native futures for validation only; no policy training'))


def mega_replay(args):
    from nav.marble_dynamics.script_forecast import predict_events
    rt.SEARCH_WORKERS = 0
    worlds = rt.WorldManager()
    cases = []
    try:
        for path in sorted(args.mega_traces.rglob('observations.jsonl.gz')):
            parts = path.relative_to(args.mega_traces).parts
            name = parts[0]
            trial = parts[1] if len(parts)>2 else 'restore'
            if args.maps and name not in args.maps:
                continue
            world = worlds.get(name)
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]
            indices = set(range(0, len(rows)-args.horizon, args.stride))
            for i in range(1, len(rows)):
                x, y = (rows[j]['dynamics'] for j in (i-1, i))
                keys = ('held_type', 'mega_active')
                if (any(x['inventory'][k] != y['inventory'][k] for k in keys)
                        or x['native']['state']['radius'] != y['native']['state']['radius']):
                    indices.update(range(max(0, i-args.horizon), min(i+2, len(rows)-args.horizon)))
            for i in sorted(indices):
                window = rows[i:i+args.horizon+1]
                start = window[0]['dynamics']; actual = window[-1]['dynamics']
                clocks = [r['dynamics']['native']['frame_clock_ms'] for r in window]
                if any(b-a != 64 for a,b in zip(clocks, clocks[1:])):
                    continue
                case = dict(map=name, trial=trial, start=i)
                try:
                    predicted = predict_events(world.engine, start, [r['action'] for r in window[:-1]],
                                               collect_items=True, model_mega=True)
                    if predicted['failure']:
                        raise ValueError(str(predicted['failure']))
                    got = predicted['states'][-1]; observed = actual['native']['state']
                    errors = {k:float(np.max(np.abs(np.asarray(got[k])-np.asarray(observed[k]))))
                              for k in TOLERANCES}
                    parameter_error = max(abs(got['parameters'][k]-v) for k,v in observed['parameters'].items()
                                          if isinstance(v, (float,int)))
                    inventory_match = all(got['inventory_context'][k] == actual['inventory'][k]
                                          for k in ('held_type','mega_active'))
                    predicted_points = sum(float(e['item']['value']) for e in predicted['item_events'] if e['item']['gem'])
                    actual_points = sum(max(0., float(r['outcome'].get('gem_delta') or 0)) for r in window[:-1])
                    case.update(errors=errors, parameter_error=parameter_error, inventory_match=inventory_match,
                        predicted_pickup_points=predicted_points, actual_pickup_points=actual_points,
                        pickup_match=predicted_points == actual_points,
                        passed=all(errors[k] <= limit for k,limit in TOLERANCES.items())
                               and parameter_error <= 1e-5 and inventory_match and predicted_points == actual_points)
                except ValueError as exc:
                    case.update(passed=False, failure=str(exc))
                cases.append(case)
            print(json.dumps(dict(map=name, trial=trial,
                windows=len(indices), failed=sum(not c['passed'] for c in cases if c['map']==name and c['trial']==trial))), flush=True)
    finally:
        worlds.close()
    if not cases:
        raise ValueError('No native Mega windows found')
    atomic_json(args.output, dict(complete=True, cases=cases, windows=len(cases), passed=sum(c['passed'] for c in cases),
        tolerances=TOLERANCES, horizon=args.horizon, input=str(args.mega_traces),
        contract='Only starting observed profiles, inventory, clocks and native state used; actual subsequent actions for validation'))


if __name__ == '__main__':
    main()
