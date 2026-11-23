"""Run a declared paired-window matrix, preserving complete baseline histories.

One process per arm/root keeps caches, recurrent state and module settings isolated.
The matrix is JSON: cases (map, pool, seed, spawn_seed, root_s, window_s), arms
(name, settings), and optionally budget. Results and the source manifest live beside it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def source_manifest():
    paths = sorted((REPO/'ml_agent/nav').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    hashes = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(files=hashes,sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def run_case(folder, case, arm, budget):
    key = f"{case['id']}_{arm['name']}"; output = folder/(key+'.jsonl')
    cmd = [sys.executable,'-m','tools.sim_bench.full_round','--pool',case['pool'],'--seed',str(case['seed']),
        '--map',case['map'],'--spawn-seed',str(case['spawn_seed']),'--label',arm['name'],
        '--cap-s',str(case['root_s']+case['window_s']),'--root-s',str(case['root_s']),
        '--set','connected.MODE=off','--budget',str(budget),'--out',str(output)]
    for setting in arm['settings']:cmd.extend(['--root-set',setting])
    if output.exists():raise RuntimeError('Refusing to mix runs: '+str(output))
    started=time.perf_counter()
    with (folder/(key+'.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.run(cmd,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT,
            timeout=max(300,20*(case['root_s']+case['window_s'])))
    if proc.returncode:raise RuntimeError(f'{key} failed ({proc.returncode}); inspect its log')
    rows=[json.loads(s) for s in output.read_text().splitlines() if s.strip()]
    if len(rows)!=1 or rows[0]['after_root'] is None:raise RuntimeError(f'{key}: missing window')
    row=rows[0];row.update(case_id=case['id'],wall_s=time.perf_counter()-started)
    return row


def summarize(matrix, rows):
    reference=matrix['arms'][0]['name']; indexed={(r['case_id'],r['label']):r for r in rows}
    out={}
    for arm in matrix['arms'][1:]:
        pairs=[]
        for case in matrix['cases']:
            b=indexed[(case['id'],reference)];r=indexed[(case['id'],arm['name'])]
            if b['root_context']['sha256']!=r['root_context']['sha256']:
                raise RuntimeError('Unequal root state: '+case['id'])
            x,y=b['after_root'],r['after_root']
            pairs.append(dict(case=case['id'],map=case['map'],baseline=x['points'],candidate=y['points'],
                delta=y['points']-x['points'],baseline_cpu=x['cpu_s'],candidate_cpu=y['cpu_s'],
                baseline_steps=x['steps'],candidate_steps=y['steps'],falls_delta=y['falls']-x['falls'],
                handoffs=y['controller_counts'].get('connected_pickup_handoff',0),connected=y['connected']))
        ds=np.array([p['delta'] for p in pairs]); maps=sorted({p['map'] for p in pairs})
        per_map={m:float(np.mean([p['delta'] for p in pairs if p['map']==m])) for m in maps}
        rng=np.random.default_rng(91107)
        samples=np.array(list(per_map.values()))
        # Map-clustered uncertainty: multiple roots from one map are not independent maps.
        boots=rng.choice(samples,(10000,len(samples)),replace=True).mean(axis=1)
        base=sum(p['baseline'] for p in pairs);cand=sum(p['candidate'] for p in pairs)
        out[arm['name']]=dict(n=len(pairs),baseline_mean=base/len(pairs),candidate_mean=cand/len(pairs),
            delta_mean=float(ds.mean()),percent=100.*(cand/base-1) if base else None,
            map_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist(),per_map=per_map,
            cpu_ratio=sum(p['candidate_cpu'] for p in pairs)/max(1e-9,sum(p['baseline_cpu'] for p in pairs)),
            forecast_step_ratio=sum(p['candidate_steps'] for p in pairs)/max(1,sum(p['baseline_steps'] for p in pairs)),
            handoffs=sum(p['handoffs'] for p in pairs),falls_delta=sum(p['falls_delta'] for p in pairs),pairs=pairs)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('matrix',type=Path);ap.add_argument('--workers',type=int,default=8)
    o=ap.parse_args();matrix=json.loads(o.matrix.read_text());folder=o.matrix.resolve().parent
    manifest=source_manifest();(folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in manifest['files']:archive.write(REPO/name,name)
    rows=[];started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=o.workers) as pool:
        jobs=[pool.submit(run_case,folder,c,a,matrix.get('budget',2.)) for c in matrix['cases'] for a in matrix['arms']]
        for future in as_completed(jobs):
            row=future.result();rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['case_id']} {row['label']}: {row['after_root']['points']} points",flush=True)
    if source_manifest()['sha256']!=manifest['sha256']:raise RuntimeError('Source changed during evaluation')
    summary=dict(results=summarize(matrix,rows),elapsed_s=time.perf_counter()-started,
        total_cpu_s=sum(r['cpu_s'] for r in rows),total_steps=sum(r['steps'] for r in rows),
        source_sha256=manifest['sha256'],matrix_sha256=hashlib.sha256(o.matrix.read_bytes()).hexdigest())
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
