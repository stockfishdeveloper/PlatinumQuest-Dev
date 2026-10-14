"""Whole open-loop mechanics forecasts against native16ms motion on full maps."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs
from nav.collision.mesh import CollisionMap,atomic_json
from nav.collision.viewer import build_map,write_atomic
from nav.marble_dynamics.evaluate import load_rows,errors
from nav.marble_dynamics.record import ML,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);traces={};provenance=[]
    trace_folder=a.output/'trajectories';trace_folder.mkdir(exist_ok=True)
    names={1:'Super Jump',2:'Super Speed',3:'Super Bounce',4:'Shock Absorber',5:'Helicopter'}
    for root,resource in [(a.ordinary,False),(a.resources,True)]:
        for case in json.loads((root/'summary.json').read_text())['cases']:
            folder=root/(str(case['power']) if resource else case['map'])
            source=folder/('observations.json' if resource else 'decisions.json')
            logs=json.loads(source.read_text());first=logs[0]['dynamics']['native']['frame_clock_ms']
            last=logs[-1]['actual_next']['native']['frame_clock_ms'] if resource else logs[-1]['next_frame_clock_ms']
            rows=[r for r in load_rows(folder/'native.jsonl.gz') if first<=r['sim_ms']<last]
            forecast_path=folder/'small_step_forecast_r2.json' if resource else root/(case['map']+'.small_step_forecast.json')
            forecast=json.loads(forecast_path.read_text());actual=[logs[0]['dynamics']['native']['state']]+[r['after'] for r in rows]
            predicted=forecast['native_states']
            if len(actual)!=len(predicted):raise ValueError('Incomplete forecast cannot be presented as full trajectory')
            measures=[errors(p,n) for p,n in zip(predicted,actual)]
            power=case.get('power');label=names.get(power,str(power)) if resource else 'Ordinary rolling/jump'
            grant=case.get('fixture_only_grant');setup='Fixture-only setup grant '+grant if grant else ('Existing-item pickup setup' if isinstance(power,int) or power=='special_blast' else 'Natural spawn/meter')
            activations=[dict(interval=i,powerup=label) for i,r in enumerate(rows) if r['move']['use'] and r['before']['held'] and not r['after']['held']]
            if power in ('regular_blast','special_blast'):
                activations += [dict(interval=i,powerup=label) for i,c in enumerate(forecast['native_controls']) if c.get('script_events')]
            meter=[None]*len(actual)
            for i,log in enumerate(logs):meter[i*4]=log['dynamics']['blast']['value']
            if resource:meter[-1]=logs[-1]['actual_next']['blast']['value']
            value=lambda states,key:[s[key] for s in states]
            trace=dict(id=len(traces.get(case['map'],[])),source_dataset=label,family='whole_native16ms_mechanics',powerup=label,
                split='controlled_mechanics_fixture',time_s=[i*.016 for i in range(len(actual))],
                actual_position=value(actual,'position'),predicted_position=value(predicted,'position'),
                actual_velocity=value(actual,'velocity'),predicted_velocity=value(predicted,'velocity'),
                actual_spin=value(actual,'spin'),predicted_spin=value(predicted,'spin'),position_error=[e[0] for e in measures],
                max_position_error_u=max(e[0] for e in measures),final_errors=measures[-1],truncated_external_samples=0,
                rollout_failure=forecast['failure'],native_use_events=[v['interval'] for v in activations],
                native_powerup_activations=activations,ambiguous_contact_intervals=[4*i for i in forecast['ambiguous_intervals']],
                held_powerup=[names.get(s['held'],'None' if not s['held'] else str(s['held'])) for s in actual],
                blast_meter=meter,outcomes=dict(actual_points=case['actual_points'],falls=case['falls'],replies=case['replies'],setup=setup),
                control_basis=setup+'. One complete proposed-input forecast from the initial snapshot; NEVER reanchored to future native state. Native physics16ms, commands64ms. Cyan native/coral prediction. Meter is observed only at64ms snapshots (intermediate values blank). Mechanics only; no gem collection or learned-policy improvement claim.')
            traces.setdefault(case['map'],[]).append(trace)
            provenance.append(dict(map=case['map'],case=label,setup=setup,native_sha256=sha(folder/'native.jsonl.gz'),
                forecast=str(forecast_path),forecast_sha256=sha(forecast_path),observations_sha256=sha(source)))
    for name,items in traces.items():
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'))
        atomic_json(trace_folder/(name+'.trajectories.json'),dict(map=name,geometry_sha256=mesh.info['mesh_sha256'],trajectories=items))
    write_atomic(a.output/'plotly.min.js',get_plotlyjs())
    maps=list(traces);views=[build_map(ML/'collision_maps'/(name+'.npz'),a.output,maps,[trace_folder]) for name in maps]
    links=''.join('<li><a href="'+name+'.html">'+html.escape(name)+'</a></li>' for name in maps)
    write_atomic(a.output/'index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><title>16ms mechanics verification</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:32px;line-height:1.6}a{color:#79c8ff}p{max-width:1000px}</style>
<h1>16ms marble mechanics: whole forecasts versus the game</h1>
<p>Select Recording, Focus trajectory, then scrub Time; orbit, pan and zoom around the complete reviewed map geometry. Cyan is measured motion and coral is one full open-loop prediction from the initial state and proposed replies. Prediction receives no future measured states or native control tape. Physics advances every16ms; commands arrive every64ms.</p>
<p>These are mechanics fixtures, not scored navigation or training. Jump, Speed, Helicopter and item-charged Blast use actual existing items, with teleport labeled as setup before recording. Regular Blast uses the natural meter. Bounce and Shock Absorber do not occur in the selected ten-map hunt pool, so their tests explicitly grant inventory before recording. No grants, resets or rescue occur during the recorded continuation.</p>
<p>Ordinary rolling, jumps, spin, impacts and effect expiry are represented by the same general physics predictor. The selected trajectories do not certify every long-horizon landing; Mega, triple Blast, Fireball, cannons and later item pickups remain unsupported external transitions. No policy, map encoder or dynamics model was fitted.</p><ul>'''+links+'''</ul><p><a href="../planning_progress_20261012/index.html">Gem-planning experiments and limitations</a></p></html>''')
    atomic_json(a.output/'summary.json',dict(maps=views,sources=provenance,training=False,scored_navigation=False))
    print(a.output/'index.html')


if __name__=='__main__':main()
