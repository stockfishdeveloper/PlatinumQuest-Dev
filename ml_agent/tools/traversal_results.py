"""Summarize traversal screens or matched, valid native full-round score pairs."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics


def start_motion(context):
    """Compare motion, queued controls and relative active timers across processes."""
    native=context['native'];state=dict(native['state'])
    if state.get('script_context') is not None:
        state['script_context']=copy.deepcopy(state['script_context'])
        # Server authority and client ghost instance numbers differ across owned games.
        # Preserve their other context fields; only these process-local identities vary.
        for key in ('authority','target'):
            if isinstance(state['script_context'].get(key),dict):
                state['script_context'][key].pop('object_id',None)
    clock=state.pop('clock_ms')
    ends=state.pop('effect_end_clock_ms',None)
    if ends is not None:
        # Disabled effects can carry unused native timer storage; it has no physical meaning.
        state['active_effect_time_left']=[end-clock if effect[0] else None
                                          for end,effect in zip(ends,state['effects'])]
    return dict(state=state,pending_move=native['pending_move'],
                **{key:context.get(key) for key in ('bridge_action','pow_yaw_hold','pow_yaw','inventory','blast')})


def screen(folder, base):
    rows = defaultdict(dict)
    for path in folder.glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                rows[(r['map'], r['pool'], r['decision'])][r['variant']] = r
    out = {}
    for variant in sorted({v for item in rows.values() for v in item} - {base}):
        pairs = [(r[base], r[variant]) for r in rows.values() if base in r and variant in r]
        finished = [(a, b) for a, b in pairs if a['why'] in ('done', 'trip_done') and b['why'] in ('done', 'trip_done')]
        # The next group's first pickup can have a different point value. A
        # slower trip to a higher-value gem is not a like-for-like time loss.
        # Retain every result in points/failure totals; compare elapsed times
        # only for equal-point completions, and current-group clears for all.
        same_points = [(a,b) for a,b in finished if abs(a['points']-b['points'])<.01]
        ratios = [b['t_s']/a['t_s'] for a, b in same_points if a['t_s'] > 0]
        trips = [(a['t_trip'], b['t_trip']) for a, b in same_points if a.get('t_trip') and b.get('t_trip')]
        groups = [(a['t_spawn'], b['t_spawn']) for a, b in finished if a.get('t_spawn') and b.get('t_spawn')]
        result = dict(pairs=len(pairs), both_completed=len(finished),
                      matched_point_completions=len(same_points),
                      point_mismatch_completions=len(finished)-len(same_points),
                      time_ratio=math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                      mean_delta_s=statistics.mean(b['t_s']-a['t_s'] for a,b in same_points) if same_points else None,
                      trip_ratio=math.exp(statistics.mean(math.log(b/a) for a,b in trips)) if trips else None,
                      group_ratio=math.exp(statistics.mean(math.log(b/a) for a,b in groups)) if groups else None,
                      base_points=sum(a['points'] for a,b in pairs), points=sum(b['points'] for a,b in pairs),
                      base_failures=sum(a['why'] not in ('done','trip_done') for a,b in pairs),
                      failures=sum(b['why'] not in ('done','trip_done') for a,b in pairs))
        out[variant] = result
        print(variant, json.dumps(result))
    return out


def native_pair(base, candidate, *additional, unique_openings=False):
    if len(additional)%2:raise ValueError('Supply baseline/candidate paths in pairs')
    paths=[base,candidate,*additional]
    tables=[]
    complete=True;invalid=0
    for arm in (0,1):
        table={}
        for path in paths[arm::2]:
            data=json.loads((path/'summary.json').read_text())
            complete=complete and data.get('complete',False)
            invalid+=len(data.get('failures',[]))
            valid=[r for r in data['branches'] if r.get('score_valid') and r.get('completed')
                   and r.get('declared_inputs_valid') and r.get('protected_checkpoint_unchanged',True)]
            for r in valid:
                key=(r['map'],r['seed'],r.get('round',1))
                if key in table:raise ValueError('Duplicate matched opening: '+str(key))
                table[key]=r
        tables.append(table)
    shared=sorted(set(tables[0]) & set(tables[1]))
    excluded=[]
    signatures=set()
    selected=[]
    for key in shared:
        a,b=(table[key] for table in tables)
        if a.get('start_layout') != b.get('start_layout'):
            raise ValueError('Matched opening/RNG audit failed: '+str(key))
        if a.get('start_context') is None or b.get('start_context') is None:
            raise ValueError('Missing starting motion context: '+str(key))
        if start_motion(a['start_context']) != start_motion(b['start_context']):
            raise ValueError('Matched starting motion/input audit failed: '+str(key))
        signature=(key[0],(a.get('start_layout') or {}).get('gem_sha256'))
        if unique_openings and signature[1] is None:
            raise ValueError('Missing opening signature: '+str(key))
        if unique_openings and signature in signatures:
            excluded.append(dict(map=key[0],seed=key[1],round=key[2],reason='Repeated opening; retain lowest seed'))
            continue
        signatures.add(signature);selected.append(key)
    shared=selected
    per=defaultdict(list)
    pairs=[]
    for key in shared:
        a,b=(table[key] for table in tables)
        row=dict(map=key[0],seed=key[1],round=key[2],base=a['actual_points'],candidate=b['actual_points'],
                 delta=b['actual_points']-a['actual_points'])
        pairs.append(row);per[key[0]].append(row)
    means={m:dict(n=len(rs),base=statistics.mean(r['base'] for r in rs),
                  candidate=statistics.mean(r['candidate'] for r in rs),delta=statistics.mean(r['delta'] for r in rs)) for m,rs in per.items()}
    for m,row in sorted(means.items()):print(m,json.dumps(row))
    result=dict(pairs=pairs,per_map=means,valid_pairs=len(pairs),
                complete=complete,invalid_jobs=invalid,
                unique_openings=unique_openings,excluded_openings=excluded,
                base=statistics.mean(r['base'] for r in means.values()) if means else None,
                candidate=statistics.mean(r['candidate'] for r in means.values()) if means else None,
                delta=statistics.mean(r['delta'] for r in means.values()) if means else None,
                unmatched_base=len(set(tables[0])-set(tables[1])),unmatched_candidate=len(set(tables[1])-set(tables[0])))
    if len(pairs)>1 and len({r['n'] for r in means.values()})==1:
        se=statistics.stdev(r['delta'] for r in pairs)/math.sqrt(len(pairs))
        result['paired_se']=se
        result['approximate_ci95']=[result['delta']-1.96*se,result['delta']+1.96*se]
    elif means and all(len(rs)>1 for rs in per.values()):
        se=math.sqrt(sum(statistics.variance(r['delta'] for r in rs)/len(rs) for rs in per.values()))/len(per)
        result['stratified_paired_se']=se
        result['approximate_ci95']=[result['delta']-1.96*se,result['delta']+1.96*se]
    print('ALL',json.dumps({k:v for k,v in result.items() if k not in ('pairs','per_map')}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('screen','pair'))
    p.add_argument('paths',nargs='+',type=Path);p.add_argument('--base',default='base');p.add_argument('--out',type=Path)
    p.add_argument('--unique-openings',action='store_true',help='Keep lowest declared seed for each map/opening layout, without using scores')
    a=p.parse_args()
    result=screen(a.paths[0],a.base) if a.kind=='screen' else native_pair(*a.paths,unique_openings=a.unique_openings)
    if a.out:a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
