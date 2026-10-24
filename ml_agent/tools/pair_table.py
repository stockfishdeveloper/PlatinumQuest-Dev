import json, os, gzip, glob, sys
os.chdir(r'C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\dynamics_reports\handoff_safety_20261020')
tag = sys.argv[1]
tab = {}
for arm in [tag + '_baseline', tag + '_candidate']:
    s = json.load(open(arm + '/summary.json'))
    print(arm, 'branches', len(s['branches']), 'failures', len(s.get('failures', [])), 'audits',
          json.load(open(arm + '/start_layout_audit.json')).get('passed'), json.load(open(arm + '/input_integrity.json')).get('passed'))
    for b in s['branches']:
        tab.setdefault((b['map'], b['seed']), {})['cand' if 'candidate' in arm else 'base'] = dict(pts=b.get('actual_points'), falls=b.get('falls'))
for arm in [tag + '_baseline', tag + '_candidate']:
    for f in glob.glob(arm + '/*/*/physics_plan/1/decisions.jsonl.gz'):
        m = os.path.normpath(f).split(os.sep)[-5]; s = int(os.path.normpath(f).split(os.sep)[-4]); edge = blast = 0
        for l in gzip.open(f, 'rt'):
            r = json.loads(l); lab = (r.get('choice') or {}).get('label') or ''
            edge += lab.startswith('edge_hit'); blast += bool(r['action'][8])
        tab[(m, s)]['cand' if 'candidate' in arm else 'base'].update(edge=edge, blast=blast)
print('%-22s %8s | off pts(falls) blasts | on pts(falls) blasts edge' % ('map', 'seed'))
tb = tc = fb = fc = bb = bc = 0; permap = {}
for (m, s), v in sorted(tab.items()):
    b = v['base']; c = v['cand']; tb += b['pts']; tc += c['pts']; fb += b['falls']; fc += c['falls']; bb += b['blast']; bc += c['blast']
    permap.setdefault(m, [0, 0]); permap[m][0] += b['pts'] / 2; permap[m][1] += c['pts'] / 2
    print('%-22s %8d | %6.0f (%d) %3d | %6.0f (%d) %3d %4d' % (m, s, b['pts'], b['falls'], b['blast'], c['pts'], c['falls'], c['blast'], c['edge']))
print('totals off %d on %d  falls %d -> %d  blasts %d -> %d  equal-map mean %.2f -> %.2f' % (
    tb, tc, fb, fc, bb, bc, sum(v[0] for v in permap.values()) / 10, sum(v[1] for v in permap.values()) / 10))
json.dump(dict(pairs={f'{m}/{s}': v for (m, s), v in tab.items()}, totals=dict(off=tb, on=tc, falls_off=fb, falls_on=fc, blasts_off=bb, blasts_on=bc)),
          open(tag + '_tenmap_comparison.json', 'w'), indent=1)
