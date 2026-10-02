import json, math, collections, numpy as np
R = 'logs/learned_nav/rounds/'
def load(tag):
    return [json.loads(l) for l in open(R + tag + '_KingOfTheMarble_Hunt.jsonl')]
for tag in ('s6_nav', 's6_short', 's6_route'):
    rows = load(tag)
    print('=====', tag, 'rounds', len(rows), 'points', [r['points'] for r in rows], 'mean', np.mean([r['points'] for r in rows]))
    legs = [l for r in rows for l in r['legs']]
    print(' legs', len(legs), 'planner-driven', sum(l.get('planner', False) for l in legs), 'route', sum(l.get('route', False) for l in legs), 'shortcut', sum(l.get('shortcut', False) for l in legs))
    # route legs: takeover state
    rl = [l for l in legs if l.get('route')]
    if rl:
        print(' route legs (t from takeover, straight0, walk0, speed0, i0 - prev pickup):')
        for l in rl[:40]:
            print('   t %.2f  straight %.1f walk %.1f speed0 %.1f  t_route %s' % (l['t'], l['straight0'], l['walk0'], l['speed0'], l.get('t_route')))
        print('  median t', np.median([l['t'] for l in rl]), 'mean speed0', np.mean([l['speed0'] for l in rl]))
    # probes
    pr = [p for r in rows for p in r['probe_log']]
    if pr:
        kinds = collections.Counter(p['kind'] for p in pr)
        print(' probes', len(pr), 'ok', sum(p['ok'] for p in pr), 'kinds', dict(kinds))
        pk = [p for p in pr if p['kind'] == 'pickup']
        print('  pickup-kind probes', len(pk), 'with jump', sum(p['jump_at'] >= 0 for p in pk),
              'faster than nav by margin', sum(p['ok'] for p in pk))
        for p in pk[:15]:
            print('   speed %.1f straight %.1f walk %.1f t_nav %.2f t_plan %.2f jump_at %d p %.2f ok %s' % (p['speed'], p['straight'], p['walk'], p['t_nav'], p['t_close']*0.064, p['jump_at'], p['p_succ'], p['ok']))
    # nav legs: detour legs (walk - straight >= 3) times
    nl = [l for l in legs if not l.get('planner') and 'pick_i' in l]
    det = [l for l in nl if l['walk0'] - l['straight0'] >= 3.0]
    print(' nav legs', len(nl), 'detour legs (walk-straight>=3u)', len(det), 'per round', len(det)/len(rows),
          'mean t detour %.2f s, mean straight %.1f, walk %.1f' % (np.mean([l['t'] for l in det]), np.mean([l['straight0'] for l in det]), np.mean([l['walk0'] for l in det])) if det else '')
    # falls
    print(' falls nav', sum(r['falls_nav'] for r in rows), 'planner', sum(r['falls_planner'] for r in rows), 'after handback', sum(r['falls_after_handback'] for r in rows), 'planner decisions/round', np.mean([r['planner_decisions'] for r in rows]), 'stuck', sum(r['stuck_breaks'] for r in rows))
