"""Complete ten-map matched mechanics review, with native and predicted paths."""
import argparse
import html
import json
from pathlib import Path

from plotly.offline import get_plotlyjs
from nav.collision.viewer import build_map,write_atomic
from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ML/'collision_maps/viewer/matched_mechanics_all_maps_20261012')
    p.add_argument('--reports',type=Path,nargs='+',default=[ML/'dynamics_reports/mechanics_small_step_cutoff_recheck_20261012',
        ML/'dynamics_reports/mechanics_pool_coverage_20261012'])
    p.add_argument('--script-preflight',type=Path,help='Render native-only script/power observability responses, including measured body radius')
    a=p.parse_args();output=a.output
    if a.script_preflight:return script_response_viewer(a.script_preflight,output)
    if output.exists():raise ValueError('Keep prior review evidence; choose a new directory')
    output.mkdir();write_atomic(output/'plotly.min.js',get_plotlyjs())
    reports=a.reports
    names=json.loads((ML/'configs/real_maps_20261009.json').read_text())['maps']
    maps=[];sources=[];rows=[]
    for name in names:
        directory=next((r for r in reports if (r/(name+'.trajectories.json')).exists()),None)
        if directory is None:raise ValueError('No controlled coverage for '+name)
        path=directory/(name+'.trajectories.json');traces=json.loads(path.read_text())['trajectories']
        maps.append(build_map(ML/'collision_maps'/(name+'.npz'),output,names,[directory]))
        sources.append(dict(map=name,report=str(path),sha256=sha(path),trials=len(traces)))
        rows.append('<tr><td><a href="'+name+'.html">'+html.escape(name)+'</a></td><td>'+str(len(traces))+
            '</td><td>'+str(sum(bool(t['ambiguous_contact_intervals']) for t in traces))+
            '</td><td>'+format(max(t['max_position_error_u'] for t in traces),'.6g')+'</td></tr>')
    description='This gallery contains'+str(sum(s['trials'] for s in sources))+' controlled trajectories from '+html.escape(', '.join(r.name for r in reports))+'.'
    write_atomic(output/'index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><title>General mechanics on all ten maps</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:32px;line-height:1.6}a{color:#79c8ff}p{max-width:1100px}td,th{padding:10px;border-bottom:1px solid #314353;text-align:left}</style>
<h1>General marble mechanics on all ten maps</h1>
<p>Open a map, select Recording, then use Focus trajectory and Time. Orbit, pan, zoom, inspect materials or view from underneath. Every reviewed collision triangle remains displayed. Cyan is native marble motion and coral is the general analytical predictor.</p>
<p>Each forecast begins from one recorded physical state and receives the complete recorded external Move/camera command sequence. It never receives future measured position, spin, camera state or contact state. These are controlled mechanics comparisons, including alternate initial spins, jump timings and power modifiers. They are not forecasts from proposed policy replies, scored gem-collection runs or learned-model results. Physics and observations both advance every16ms.</p>
<p>'''+description+''' Matched comparisons are identified in the native source manifests; the rich use/no-use experiments hold inventory and measured initial position, velocity and spin identical. Requested setup kind does not prove the marble remained supported or airborne until activation. Setup teleports, respawns and inventory grants are disclosed; none occur during a recorded continuation. Script transitions delimit physical forecasts and omitted samples remain counted. Falls and larger errors remain available.</p>
<p>Flagged contact-sensitive trajectories are retained, including larger errors. The table shows the largest error over each displayed trajectory, not a calibrated prediction bound. First landings can agree while a later rebound diverges. This review does not certify safety or broad generalization.</p>
<table><thead><tr><th>Map</th><th>Trials</th><th>Flagged trajectories</th><th>Maximum trajectory position error (u)</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p><a href="../planning_progress_20261012/index.html">Gem planning, native comparisons and limitations</a></p></html>''')
    atomic_json(output/'summary.json',dict(maps=maps,sources=sources,training=False,scored_navigation=False))
    print(output/'index.html',flush=True)


def script_response_viewer(source,output):
    from nav.collision.mesh import CollisionMap
    from nav.marble_dynamics.evaluate import load_rows
    from nav.source_archive import snapshot_sources
    result=json.loads((source/'result.json').read_text())
    if not result['complete'] or not json.loads((source/'independent_audit.json').read_text())['all_six_contracts_pass']:
        raise ValueError('Requires independently verified native script responses')
    if output.exists():raise ValueError('Keep previous review evidence')
    output.mkdir(parents=True);directory=output/'trajectories';directory.mkdir();traces=[];provenance=[]
    for case in result['cases']:
        folder=source/case['power'];record=json.loads((folder/'result.json').read_text());rows=load_rows(folder/'native.jsonl.gz')
        states=[rows[0]['before']];times=[0.];stages=['before native frame0'];moves=[rows[0]['move']]
        replies=[record['observations'][0]['dynamics']['bridge_action']];uses=[];boundaries=[]
        for i,row in enumerate(rows):
            before=row['before'];previous=states[-1]
            if i and any(previous[k]!=before[k] for k in ('radius','datablock','parameters')):
                states.append(before);times.append(i*.016);stages.append('after script/body update; before native frame'+str(i))
                moves.append(row['move']);replies.append(record['observations'][i]['dynamics']['bridge_action']);boundaries.append(len(states)-1)
            states.append(row['after']);times.append((i+1)*.016);stages.append('after native frame'+str(i))
            moves.append(row['move']);replies.append(record['observations'][i]['dynamics']['bridge_action'])
            if row['move']['use']:uses.append(len(states)-1)
        count=len(states);values=lambda key:[s[key] for s in states];empty=[[None,None,None] for _ in states]
        label=case['power'].removesuffix('Item');authority=lambda s:s['script_context']['authority']
        trace=dict(id=len(traces),source_dataset=label,family='native_script_response',powerup=label,split='dev_engineering',
            time_s=times,actual_position=values('position'),actual_velocity=values('velocity'),actual_spin=values('spin'),
            predicted_position=empty,predicted_velocity=empty,predicted_spin=empty,position_error=[None]*count,
            prediction_available=[False]*count,max_position_error_u=None,final_errors=None,truncated_external_samples=0,
            rollout_failure=None,actual_radius=values('radius'),sample_stage=stages,native_move=moves,queued_reply_before_frame=replies,
            held_powerup=[authority(s)['held_item'] or 'None' for s in states],script_held_type=[authority(s)['held_item_power_id'] for s in states],
            native_effect_active=[[active for active,left in s['effects']] for s in states],
            native_effect_remaining_ms=[[left for active,left in s['effects']] for s in states],
            native_use_events=uses,native_powerup_activations=[dict(sample=i,powerup=label) for i in uses],
            outcomes=dict(seed=case['seed'],native_frames=len(rows),observed_body_boundary_samples=boundaries,
                fixture_only_grant=case['power'],falls=sum(o['outcome']['fell'] for o in record['observations'])),
            control_basis='Measured native response only. Setup grant/descending spinning pose; one requested use. Cyan centers and translucent sphere use exact observed position/radius. Duplicate elapsed times retain the Mega body event before physics; sample stage is explicit. Queued reply is the preceding live observation metadata. No forecast or learned skill result.')
        traces.append(trace);provenance.append(dict(power=case['power'],native_path=str(folder/'native.jsonl.gz'),native_sha256=sha(folder/'native.jsonl.gz'),
            result_path=str(folder/'result.json'),result_sha256=sha(folder/'result.json')))
    mesh=CollisionMap(ML/'collision_maps/CubeIsle_Hunt.npz')
    atomic_json(directory/'CubeIsle_Hunt.trajectories.json',dict(map='CubeIsle_Hunt',geometry_sha256=mesh.info['mesh_sha256'],trajectories=traces))
    write_atomic(output/'plotly.min.js',get_plotlyjs());view=build_map(ML/'collision_maps/CubeIsle_Hunt.npz',output,['CubeIsle_Hunt'],[directory])
    write_atomic(output/'index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><title>Recorded power and body responses</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:32px;line-height:1.6}a{color:#79c8ff}p{max-width:1000px}td,th{padding:9px;border-bottom:1px solid #314353;text-align:left}</style>
<h1>Recorded power and body responses</h1><p>Select Recording, Focus trajectory and Time to orbit, pan or zoom around the native paths. The complete verified Cube collision mesh is retained. Cyan follows measured marble centers; the translucent sphere has the measured radius in world units. Readouts show velocity, spin, inventory, active effects and actual native controls.</p>
<p>Six independent short descending/spinning setup fixtures, one item/use per process. These are native response and observability checks. Mega's script growth is a separate sample at the same elapsed time before its next physics frame; the phase label makes that boundary explicit. No predicted trajectory is available for these recordings.</p>
<table><thead><tr><th>Map</th><th>Collision triangles</th><th>Native recordings</th></tr></thead><tbody><tr><td><a href="CubeIsle_Hunt.html">Cube Isle</a></td><td>1950</td><td>6</td></tr></tbody></table>
<p><a href="../../../docs/RESULT_SCRIPT_STATE_OBSERVABILITY_2026-10-13.md">Methods, event alignment and limitations</a> · <a href="../planning_progress_20261012/index.html">Overall planning progress</a></p></html>''')
    snapshot_sources(output,[Path(__file__),ML/'nav/collision/viewer.py'])
    atomic_json(output/'summary.json',dict(maps=[view],sources=provenance,training=False,scored_navigation=False,
        native_only=True,source_result_sha256=sha(source/'result.json')))
    print(output/'index.html',flush=True)


if __name__=='__main__':main()
