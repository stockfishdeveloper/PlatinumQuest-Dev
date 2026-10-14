"""Audit native score/control evidence and replay recorded live command windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.events import external_changes


def audit(folder,output):
    results=[];by_map={};geometry_hashes={}
    for dest in sorted(folder.iterdir()):
        if not dest.is_dir() or not (dest/'report.json').exists():continue
        report=json.loads((dest/'report.json').read_text());logs=json.loads((dest/'decisions.json').read_text())
        rows=load_rows(dest/'native.jsonl.gz');by={r['sim_ms']:r for r in rows}
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(report['map']+'.npz')))
        checks=[];mismatches=[];excluded=[];trajectories=[]
        for start in range(max(0,len(logs)-4)):
            window=logs[start:start+5]
            if any(external_changes(l['predicted_next'],l['actual_next']) or l['outcome']['fell'] for l in window):
                excluded.append(start);continue
            pred=predict_replies(engine,window[0]['dynamics'],[l['action'] for l in window])
            clock=window[0]['dynamics']['native']['frame_clock_ms']
            actual=[window[0]['dynamics']['native']['state']]+[l['actual_next'] for l in window]
            for i,control in enumerate(pred['controls']):
                row=by[clock+i*64]
                if control['move']!=row['move']:
                    mismatches.append(dict(start=start,offset=i,predicted=control['move'],actual=row['move']))
            error=errors(pred['states'][-1],actual[len(pred['states'])-1]);checks.append(error)
            per=[errors(p,a) for p,a in zip(pred['states'],actual)]
            trajectories.append(dict(id=start,family='live_command_window',powerup='observed',split='demo',
                time_s=pred['time_s'],actual_position=[s['position'] for s in actual[:len(pred['states'])]],
                predicted_position=[s['position'] for s in pred['states']],
                actual_velocity=[s['velocity'] for s in actual[:len(pred['states'])]],
                predicted_velocity=[s['velocity'] for s in pred['states']],
                actual_spin=[s['spin'] for s in actual[:len(pred['states'])]],predicted_spin=[s['spin'] for s in pred['states']],
                position_error=[e[0] for e in per],max_position_error_u=max(e[0] for e in per),
                final_errors=error,truncated_external_samples=0,rollout_failure=pred['failure'],
                control_basis='Five-step open-loop replay of executed bridge replies from one initial state; no future physical/camera state',
                native_use_events=[],ambiguous_contact_intervals=pred['ambiguous_intervals']))
        # Keep a few beginnings/pickup windows for an inspectable viewer, not
        # hundreds of near-identical overlapping traces.
        keep={0,len(trajectories)-1}
        for pickup in report['actual_pickup_events']:
            keep.update(i for i,t in enumerate(trajectories) if t['id']<=pickup['decision']<=t['id']+4)
        selected=[t for i,t in enumerate(trajectories) if i in keep]
        for case in selected:case['source_dataset']=report.get('branch',folder.name)
        by_map.setdefault(report['map'],[]).extend(selected);geometry_hashes[report['map']]=report['geometry_sha256']
        values=np.asarray(checks)
        result=dict(map=report['map'],branch=report.get('branch'),actual_points=report['actual_points'],pickup_events=len(report['actual_pickup_events']),
                    score_difference=report['final_score']-report['initial_score'],
                    logged_pickup_points=sum(p['points'] for p in report['actual_pickup_events']),
                    stop_reason=report['stop_reason'],native_transitions=len(rows),
                    five_step_windows=len(checks),excluded_external_windows=excluded,
                    control_mismatches=mismatches,
                    five_step_errors_p95=np.percentile(values,95,axis=0).tolist() if len(values) else None,
                    five_step_errors_max=values.max(axis=0).tolist() if len(values) else None,
                    median_planning_s=float(np.median(report['planning_s'])) if report['planning_s'] else None,
                    decision_log_sha256=hashlib.sha256((dest/'decisions.json').read_bytes()).hexdigest(),
                    policy_training_started=False)
        results.append(result)
    for name,trajectories in by_map.items():
        atomic_json(output/(name+'.trajectories.json'),dict(map=name,geometry_sha256=geometry_hashes[name],trajectories=trajectories))
    atomic_json(output/'audit.json',dict(source=str(folder),maps=results,global_planner_approved=False))
    for result in results:print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.folder,a.output)
