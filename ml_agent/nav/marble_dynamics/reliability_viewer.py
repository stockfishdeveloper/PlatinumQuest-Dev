"""Full native episode paths, explicitly reanchored predictions and actual scores."""
import argparse
import html
import json
import gzip
from pathlib import Path

from plotly.offline import get_plotlyjs
from nav.collision.mesh import atomic_json
from nav.collision.viewer import build_map,write_atomic
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.evaluate import load_rows


def round_trace_evidence(logs):
    """Before-state evidence only; discontinuities break lines, not observations."""
    breaks=[];reasons=[];goals=[];kinds=[]
    for i,row in enumerate(logs):
        why=[];before=row['before']
        if i:
            previous=logs[i-1];prior=previous['before']
            if before['native']['frame_clock_ms']-prior['native']['frame_clock_ms']!=64:why.append('unobserved_control_interval')
            if any(previous['outcome'].get(k,False) for k in ('fell','round_ended','reconnected')):why.append('fall_or_round_boundary')
            a=prior.get('game_context',{});b=before.get('game_context',{})
            if a.get('available') and b.get('available') and a.get('respawns')!=b.get('respawns'):why.append('native_respawn')
            if prior['native'].get('object_id')!=before['native'].get('object_id'):why.append('body_changed')
        breaks.append(bool(why));reasons.append(why)
        context=row['choice'].get('goal_context')
        if context is not None:
            goals.append(context['destination']);kinds.append(context['kind']);continue
        # Older recordings prove the checked item's identity at the same
        # BEFORE observation, but do not contain the actor/fallback waypoint.
        item_id=row['choice'].get('goal_item_id') if row.get('predicted') is not None else None
        matches=[item for item in before.get('items',[]) if item_id is not None and item['id']==item_id]
        if len(matches)==1:
            goals.append(matches[0]['position']);kinds.append('checked_observed_gem' if matches[0]['gem'] else 'checked_observed_powerup')
        else:goals.append([None,None,None]);kinds.append('unknown')
    return breaks,reasons,goals,kinds


def full_round_viewer(data,output,summary,prepared,include_incomplete=False):
    """Show captured macro observations, not invented goals or whole-round forecasts."""
    folder=data/'viewer_trajectories';folder.mkdir(exist_ok=True)
    maps={};table=[]
    for report in summary['branches']:
        if not report.get('score_valid'):
            partial=data/report['map']/str(report['seed'])/report['branch']/str(report['round'])/'report.json'
            if not include_incomplete or not partial.exists():continue
            report=json.loads(partial.read_text(encoding='utf-8'))
            if report.get('completed') or report.get('accumulated_points') is None:continue
        if report.get('declared_inputs_valid') is False:continue
        name=report['map'];branch=report['branch'];seed=report['seed'];round_index=report['round']
        path=data/name/str(seed)/branch/str(round_index)/'decisions.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf-8') as source:logs=[json.loads(line) for line in source]
        if not logs:continue
        breaks,break_reasons,goals,goal_kinds=round_trace_evidence(logs)
        actual=[r['before']['native']['state'] for r in logs]
        # The previous command forecasts the next observation. Control-word
        # boundaries and falls have explicit gaps; no final next-round pose.
        predicted=[None]+[r['predicted'] if not (r['outcome']['fell'] or r['outcome']['round_ended']
            or r['outcome']['reconnected']) else None for r in logs[:-1]]
        for i,gap in enumerate(breaks):
            if gap:predicted[i]=None
        empty=[None,None,None]
        vectors=lambda states,key:[s[key] if s is not None else empty for s in states]
        errors=[sum((a-b)**2 for a,b in zip(p['position'],q['position']))**.5 if q else None
                for p,q in zip(actual,predicted)]
        first=logs[0]['before']['native']['frame_clock_ms']
        clock=[r['before']['native']['frame_clock_ms'] for r in logs]
        # A control/rescue can advance an extra observation between commands.
        # Such gaps must not be presented as a one-reply prediction comparison.
        for i in range(1,len(logs)):
            if clock[i]-clock[i-1]!=64:predicted[i]=None;errors[i]=None
        available=[s is not None for s in predicted]
        score=[r['raw_observation'][33] for r in logs]
        trace=dict(id=str(seed)+'-round'+str(round_index),source_dataset=branch,
            family='complete_native_hunt_round' if report.get('score_valid') else 'incomplete_native_hunt_attempt',
            powerup='observed',split='familiar_map_fresh_launch',
            time_s=[(c-first)/1000. for c in clock],actual_position=vectors(actual,'position'),
            predicted_position=vectors(predicted,'position'),actual_velocity=vectors(actual,'velocity'),
            predicted_velocity=vectors(predicted,'velocity'),actual_spin=vectors(actual,'spin'),
            predicted_spin=vectors(predicted,'spin'),prediction_available=available,position_error=errors,
            actual_radius=[s['radius'] for s in actual],
            max_position_error_u=max((e for e in errors if e is not None),default=None),
            final_errors=None,truncated_external_samples=0,rollout_failure=None,
            native_use_events=[],native_powerup_activations=[],ambiguous_contact_intervals=[],
            path_break_before=breaks,path_break_reasons=break_reasons,
            goal_position=goals,goal_kind=goal_kinds,
            task_goal_position=[r['choice'].get('task_goal_context',{}).get('destination') or empty for r in logs],
            actor_goal_position=[r['choice'].get('actor_goal_context',{}).get('destination') or empty for r in logs],
            held_powerup=[str(int(r['raw_observation'][38])) for r in logs],score=score,
            pickup_decisions=[i for i in range(1,len(score)) if score[i]>score[i-1]],
            blast_meter=[r['raw_observation'][39] for r in logs],
            control_basis='Full native round,64ms macro observations. Cyan observed; coral previous checked one-reply predictions, reanchored during gameplay, not a full-round open-loop forecast. Lines break at native respawns and unobserved control intervals; every measured point remains available. Destinations come from explicit before-action logging or older checked item IDs resolved in the same before observation. Other goals/predictions remain unknown. Command requests and native inventory/effect consumption evidence remain distinct. Recording time includes countdown/control intervals; official duration and points come from native end score. Next-round reset pose omitted.',
            outcomes={k:report.get(k) for k in ('actual_points','accumulated_points','completed','score_valid','played_seconds',
                'falls','round_seconds','stop_reason','elapsed_wall_s','declared_inputs_valid','automatic_oob_key_requests')})
        if not report.get('score_valid'):
            trace['control_basis']='INCOMPLETE ATTEMPT: no full-round point score. Observed partial motion and score only. '+trace['control_basis']
        audit_path=path.parent/'activation_audit.json'
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            trace['native_use_events']=audit['commands']
            trace['native_powerup_activations']=audit['confirmed_native_inventory_consumptions']
        maps.setdefault(name,[]).append(trace)
        counts=report['counts'];planned=counts.get('planner_decisions',0);fallback=counts.get('fallback_decisions',0)
        points=report['actual_points'] if report.get('score_valid') else 'INCOMPLETE ('+str(report['accumulated_points'])+' accumulated)'
        table.append([name,seed,branch,points,report['falls'],report['recovery']['rescues'],
            str(planned)+'/'+str(fallback),round(report['elapsed_wall_s'],2),
            'passed' if report.get('declared_inputs_valid') else 'unknown legacy audit'])
    output.mkdir(parents=True,exist_ok=True);write_atomic(output/'plotly.min.js',get_plotlyjs())
    names=list(maps);rows=[]
    for name,traces in maps.items():
        from nav.collision.mesh import CollisionMap
        fingerprint=CollisionMap(ML/'collision_maps'/(name+'.npz')).info['mesh_sha256']
        atomic_json(folder/(name+'.trajectories.json'),dict(map=name,geometry_sha256=fingerprint,trajectories=traces))
        rows.append(build_map(ML/'collision_maps'/(name+'.npz'),output,names,[folder]))
    links=''.join('<li><a href="'+name+'.html">'+html.escape(name)+'</a></li>' for name in names)
    cells=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in table)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete Hunt round review</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:40px;line-height:1.6}a{color:#79c8ff}p{max-width:1050px}td,th{padding:8px;border-bottom:1px solid #314353;text-align:left}</style>
<h1>Complete native Hunt rounds</h1><p>Original-duration rounds, competitive off, legal OOB click and8s time-cost rescue. Physics16ms and decisions64ms. Controller labels distinguish combined analytical planning with counted A2 fallback, geometry routing without search where present, and legacy routing. All use the same frozen A2 weights; its local observation builder is unchanged. Independent native launches; supplied seeds do not certify identical starting contexts. No policy promotion or learned-student integration.</p>
<p>Choose Recording, Focus trajectory and Time. Cyan is measured motion; coral is previous checked one-reply forecasts, with unknown intervals blank. Lines break at respawns/control gaps; every measured point remains available. Purple destinations come from before-action logging or the checked item resolved in that same observation; unrecorded fallback goals stay unknown. Command requests and native consumption evidence are listed separately where audited. Score is observed at each captured observation; the table uses the authoritative native end score. The recording clock includes countdown/control time and differs from the official Hunt timer. Explicitly input-invalid rounds are excluded; legacy missing audits remain labelled unknown.</p><ul>'''+links+'''</ul><table><thead><tr><th>Map</th><th>Seed</th><th>Controller</th><th>Native points</th><th>Falls</th><th>Rescues</th><th>Planned/fallback decisions</th><th>Wall seconds</th><th>Declared inputs</th></tr></thead><tbody>'''+cells+'</tbody></table></html>'
    if prepared.get('fast_checkpoint'):
        page=page.replace('All use the same frozen A2 weights; its local observation builder is unchanged.',
            'The fast navigator uses a trained complete-map residual policy around frozen A2. Geometry-routed A2 is the reference. Fast decisions use cached geometry attention, without analytical action search.')
        page=page.replace('No policy promotion or learned-student integration.',
            'This is opt-in learned navigation evaluation; the default policy is unchanged. Native points assess gameplay separately from imitation loss.')
    write_atomic(output/'index.html',page.replace('Physics16ms','Physics'+str(prepared['physics_frame_ms'])+'ms'))
    if include_incomplete:
        page=page.replace('<title>Complete Hunt round review</title>','<title>Hunt rounds and incomplete attempts</title>')
        page=page.replace('<h1>Complete native Hunt rounds</h1>',
            '<h1>Native Hunt rounds and incomplete attempts</h1><p>Incomplete recordings are explicitly labeled; their accumulated points are not full-round scores. This gallery does not certify a completed map pool. Purple marks the current physical destination; green marks the logged real task. Actor input destinations are shown in the time readout. Missing historic task/input destinations remain unknown.</p>')
        write_atomic(output/'index.html',page.replace('Physics16ms','Physics'+str(prepared['physics_frame_ms'])+'ms'))
    atomic_json(output/'summary.json',dict(maps=rows,source=str(data),source_summary_sha256=sha(data/'summary.json'),
        full_round=True,incomplete_recordings_included=include_incomplete,policy_training_started=False,promotion=False))
    print('Saved '+str(output/'index.html'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('--output',type=Path,default=ML/'collision_maps/viewer/reliability')
    p.add_argument('--include-incomplete',action='store_true',help='Include audited partial reports with explicit incomplete score labels')
    a=p.parse_args();summary=json.loads((a.data/'summary.json').read_text());by_map={};fingerprints={}
    prepared=json.loads((a.data/'prepared.json').read_text())
    if prepared.get('full_round'):
        full_round_viewer(a.data,a.output,summary,prepared,a.include_incomplete);return
    audit=json.loads((a.data/'audit.json').read_text()) if (a.data/'audit.json').exists() else {'branches':[]}
    trace_folder=a.data/'viewer_trajectories';trace_folder.mkdir(exist_ok=True)
    for report in summary['branches']:
        if report.get('status')=='invalid':continue
        name=report['map'];branch=report['branch'];seed=report['seed']
        path=a.data/name/str(seed)/branch/'decisions.json';logs=json.loads(path.read_text())
        if not logs:continue
        actual=[logs[0]['dynamics']['native']['state']]+[l['actual_next'] for l in logs]
        predicted=[actual[0]]+[l['predicted_next'] for l in logs]
        available=[s is not None for s in predicted];empty=[None,None,None]
        score=[report['initial_score']]
        for l in logs:score.append(score[-1]+l['score_delta'])
        assert abs(score[-1]-report['final_score'])<.01
        per=[0.]+[l['errors'][0] if l['errors'] else None for l in logs]
        clock=logs[0]['dynamics']['native']['frame_clock_ms']
        powerups={1:'Super Jump',2:'Super Speed',3:'Super Bounce',4:'Shock Absorber',5:'Helicopter',6:'Mega Marble'}
        activations=[dict(interval=int((r['sim_ms']-clock)/64),powerup=powerups.get(r['before']['held'],str(r['before']['held'])))
                    for r in load_rows(path.with_name('native.jsonl.gz'))
                    if r['move']['use'] and r['before']['held'] and not r['after']['held']
                    and clock<=r['sim_ms']<clock+len(logs)*64]
        native_use=[event['interval'] for event in activations]
        checked=next((b for b in audit['branches'] if (b['map'],b['seed'],b['branch'])==(name,seed,branch)),{})
        blasts=[dict(interval=b['decision'],powerup='Regular Blast') for b in checked.get('regular_blast_activations',[])]
        blasts += [dict(interval=b['decision'],powerup='Item-charged Blast') for b in checked.get('special_blast_activations',[])]
        activations+=blasts;native_use=sorted(native_use+[b['interval'] for b in blasts])
        goals=[]
        for l in logs:
            goal=l['gem'][:3];kind=l.get('goal_kind','gem')
            if kind in ('gem','resource'):
                assert any(sum((float(a)-float(b))**2 for a,b in zip(item['position'],goal))<1e-6
                    for item in l['dynamics']['items'] if item['gem']==(kind=='gem')),('unobserved goal',name,seed,branch,l['decision'])
            elif kind=='landing':
                assert l.get('landing_triangle_id') is not None,('landing without surface provenance',l['decision'])
            goals.append(goal)
        value=lambda states,key:[s[key] if s else empty for s in states]
        trace=dict(id=seed,source_dataset=branch,family='matched_native_episode',powerup='observed',split='reused_start_regression' if prepared.get('fixture_data') else 'fresh_seed_validation',
                   time_s=[i*.064 for i in range(len(actual))],actual_position=value(actual,'position'),predicted_position=value(predicted,'position'),
                   actual_velocity=value(actual,'velocity'),predicted_velocity=value(predicted,'velocity'),actual_spin=value(actual,'spin'),predicted_spin=value(predicted,'spin'),
                   prediction_available=available,position_error=per,max_position_error_u=max(e for e in per if e is not None),
                   final_errors=logs[-1]['errors'],truncated_external_samples=0,rollout_failure=None,
                   native_use_events=native_use,ambiguous_contact_intervals=[],
                   native_powerup_activations=activations,held_powerup=[powerups.get(s['held'],'None' if not s['held'] else str(s['held'])) for s in actual],
                   score=score,pickup_decisions=[l['decision'] for l in logs if l['score_delta']>0],
                   goal_position=goals+[goals[-1]],goal_kind=[l.get('goal_kind','gem') for l in logs]+[logs[-1].get('goal_kind','gem')],
                   blast_meter=[l['dynamics'].get('blast',{}).get('value') for l in logs]+[None],
                   control_basis='Full measured episode. Predictions are joined short control chunks, reanchored at each search/current observation; NOT a full-episode open-loop forecast. Null gaps mean unmodelled futures. Held-power activations are native held-to-empty transitions; Regular Blast markers require native impulse audit. Goals are observed items or a checked geometric landing point; the last goal is retained for display. Meter is actual observed charge; final meter is unknown. Ambiguity intervals are not summarized in this episode view.',
                   outcomes={k:report[k] for k in ('actual_points','falls','stop_reason','simulated_s','median_planning_s')})
        by_map.setdefault(name,[]).append(trace);fingerprints[name]=report['geometry_sha256']
    names=list(by_map)
    for name,traces in by_map.items():atomic_json(trace_folder/(name+'.trajectories.json'),dict(map=name,geometry_sha256=fingerprints[name],trajectories=traces))
    a.output.mkdir(parents=True,exist_ok=True);write_atomic(a.output/'plotly.min.js',get_plotlyjs())
    rows=[build_map(ML/'collision_maps'/(name+'.npz'),a.output,names,[trace_folder]) for name in names]
    links=''.join('<li><a href="'+name+'.html">'+html.escape(name)+'</a></li>' for name in names)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ('map','seed','branch','actual_points','falls','simulated_s','stop_reason'))+'</tr>' for r in summary['branches'] if r.get('status')!='invalid')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Planning reliability comparison</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:40px;line-height:1.6}a{color:#79c8ff}p{max-width:1050px}td,th{padding:8px;border-bottom:1px solid #314353;text-align:left}</style>
<h1>Planning reliability: planner and frozen A2 navigator</h1>
<p>These are bounded fresh-seed starts on familiar maps. Both branches restore the same settled spawn, game RNG, gems and score before playing. They stop at the first fall or DECISION_LIMIT decisions; no rescue or teleport occurs during play. This is a short-attempt comparison, not a full-round or unseen-map benchmark.</p>
<p>Choose Recording, then Focus trajectory and scrub Time to see the entire measured path and game score. Cyan is native motion. Coral joins short predictions that are reanchored as the controller observes the game; it is not one long open-loop forecast. Gaps explicitly mean prediction unavailable for external events. The predictor is independent analytical mechanics; no policy or dynamics model was trained here.</p>
<p><a href="../index.html">Original ten-map geometry/mechanics viewer</a> · <a href="../gem_demo/index.html">Earlier development attempts</a></p><ul>'''+links+'''</ul>
<table><thead><tr><th>Map</th><th>Seed</th><th>Controller</th><th>Points</th><th>Falls</th><th>Played seconds</th><th>Stop</th></tr></thead><tbody>'''+table+'</tbody></table></html>'
    if prepared.get('fixture_data'):
        page=page.replace('These are bounded fresh-seed starts on familiar maps.','These are bounded development regressions from archived starting worlds, not fresh validation.')
    timing='<p>Native physics steps: '+str(prepared.get('physics_frame_ms',64))+'ms; decisions:64ms. Whole-search wall-time limit: '+str(prepared.get('search_budget_s',0))+'s (0 means unbounded). Zero falls covers only the played interval and does not certify a safe final landing or stopping state.</p>'
    page=page.replace('<table><thead>',timing+'<table><thead>')
    write_atomic(a.output/'index.html',page.replace('DECISION_LIMIT',str(prepared['decisions'])))
    atomic_json(a.output/'summary.json',dict(maps=rows,source=str(a.data),source_summary_sha256=sha(a.data/'summary.json'),policy_training_started=False))
    print('Saved '+str(a.output/'index.html'))


if __name__=='__main__':main()
