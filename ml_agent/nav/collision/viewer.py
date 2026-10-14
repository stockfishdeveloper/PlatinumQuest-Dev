"""Offline orbit/pan/zoom viewers rendered from hash-checked model NPZ files.

python -m nav.collision.viewer --directory ml_agent/collision_maps
No independent render mesh, walkability mask, coordinate rounding or decimation.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from nav.collision.mesh import CollisionMap, atomic_json

PALETTE = ['#98b9cc','#f5b857','#9cdc8a','#c895ef','#e88e83','#73d4ce','#c7c96c']


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(str(tmp),str(path))


def color_material(m):
    if m['force']:
        return '#f096d6'
    if m['friction'] < .5:
        return '#62c6ff'
    if m['friction'] > 1.5:
        return '#7dd698'
    if m['restitution'] > 1:
        return '#c895ef'
    return '#b4c4d3'


def build_map(path, output, all_names, trajectory_directories=()):
    mesh = CollisionMap(path)
    info, fig, records = mesh.info, go.Figure(), []
    validation = json.loads(path.with_name(path.stem+'.validation.json').read_text())
    sweep, physics = validation['sweeps'], validation['physics']
    ambiguities = []
    for case in sweep.get('disagreements', []):
        if case['reason'] != 'material_boundary_tie':
            continue
        ambiguities.append(dict(label='A'+str(len(ambiguities)+1), test_id=case['id'],
                                point=case['engine']['point'], engine=case['engine'], mesh=case['mesh']))
    for k, oid in enumerate(np.unique(mesh.object_ids)):
        ids = np.flatnonzero(mesh.object_ids==oid)
        tri = mesh.triangles[ids]
        xyz = tri.reshape(-1,3)
        obj = mesh.objects[int(oid)]
        label = obj['datablock'] or Path(obj['interior_file'] or obj['shape_name']).name or obj['class']
        mats = [color_material(info['materials'][int(mi)]) for mi in mesh.material_ids[ids]]
        orientation = ['#73d3a4' if n[2]>.3 else '#c395ec' if n[2]<-.3 else '#ecb565' for n in mesh.normals[ids]]
        face = np.arange(len(xyz)).reshape(-1,3)
        fig.add_trace(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=face[:,0],j=face[:,1],k=face[:,2],
                               facecolor=mats,name=label,customdata=np.repeat(ids,3),
                               hovertemplate=html.escape(label)+'<br>triangle %{customdata}<extra></extra>',
                               flatshading=True,showscale=False,lighting=dict(ambient=.65,diffuse=.6,
                               specular=.05,roughness=.9),lightposition=dict(x=100,y=-150,z=200)))
        records.append(dict(ids=ids.tolist(), material_colors=mats, orientation_colors=orientation,
                            object_color=PALETTE[k%len(PALETTE)], object=obj,
                            material_ids=mesh.material_ids[ids].tolist(), normals=mesh.normals[ids].tolist(),
                            surface_keys=mesh.surface_keys[ids].astype(int).tolist(),
                            boundary_edges=mesh.boundary_edges[ids].tolist()))
    mesh_count = len(records)
    # Wireframe retains true polygon boundary edges; fan diagonals are not physical seams.
    wire = []
    for t, flags in zip(mesh.triangles,mesh.boundary_edges):
        for j, boundary in enumerate(flags):
            if boundary:
                wire.extend([t[j].tolist(),t[(j+1)%3].tolist(),[None,None,None]])
    wire = np.asarray(wire,dtype=object)
    fig.add_trace(go.Scatter3d(x=wire[:,0],y=wire[:,1],z=wire[:,2],mode='lines',
                             line=dict(color='#283f55',width=1),name='Polygon boundaries',
                             visible=False,hoverinfo='skip',showlegend=False))
    annotation_indices = []
    groups = {}
    for item in info.get('mission_annotations',[]):
        db = item['datablock']
        if item['class_name']=='Item':
            category = 'Gem definitions' if 'gem' in db.lower() else 'Powerup definitions'
        elif 'spawn' in db.lower():
            category = 'Spawn definitions'
        else:
            continue
        groups.setdefault(category,[]).append(item)
    colors = {'Gem definitions':'#ffe45e','Powerup definitions':'#ff7d89','Spawn definitions':'#66c4ff'}
    for category, items in groups.items():
        xyz = np.asarray([i['position'] for i in items])
        annotation_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                                 marker=dict(size=4,color=colors[category]),name=category,
                                 text=[html.escape(i['datablock']) for i in items],
                                 hovertemplate='%{text}<br>mission location; availability varies<extra></extra>'))
    ambiguity_indices = []
    if ambiguities:
        contact = np.asarray([a['point'] for a in ambiguities])
        label = contact + np.array([0., 0., 4.])
        stems = np.asarray([v for p, q in zip(contact, label) for v in (p.tolist(), q.tolist(), [None]*3)], dtype=object)
        hover = []
        for a in ambiguities:
            engine, material = a['engine'], a['mesh']['material']
            hover.append(a['label']+' / sweep '+str(a['test_id'])+'<br>Contact: '+
                         ', '.join(format(x, '.3f') for x in a['point'])+
                         '<br>Game friction / restitution: '+str(engine['friction'])+' / '+str(engine['restitution'])+
                         '<br>Query friction / restitution: '+str(material['friction'])+' / '+str(material['restitution']))
        ambiguity_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=stems[:,0], y=stems[:,1], z=stems[:,2], mode='lines',
                                  line=dict(color='#ff694b', width=5), name='Ambiguity callouts', hoverinfo='skip'))
        ambiguity_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=contact[:,0], y=contact[:,1], z=contact[:,2], mode='markers',
                                  marker=dict(size=6, color='#ff694b', line=dict(color='#fff0d8', width=2)),
                                  name='Exact ambiguity contacts', text=hover, customdata=list(range(len(ambiguities))),
                                  hovertemplate='%{text}<extra></extra>'))
        ambiguity_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=label[:,0], y=label[:,1], z=label[:,2], mode='markers+text',
                                  marker=dict(size=8, color='#ff694b', symbol='diamond', line=dict(color='#fff0d8', width=2)),
                                  name='Material ambiguities', text=[a['label'] for a in ambiguities],
                                  textfont=dict(color='#fff0d8', size=16), textposition='top center', hovertext=hover,
                                  customdata=list(range(len(ambiguities))), hovertemplate='%{hovertext}<extra></extra>'))
    trajectories=[];trajectory_sources=[]
    for directory in trajectory_directories:
        source=directory/(path.stem+'.trajectories.json')
        if not source.exists():continue
        data=json.loads(source.read_text())
        if data['geometry_sha256']!=info['mesh_sha256']:
            raise ValueError('Trajectory geometry hash differs: '+str(source))
        trajectory_sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for case in data['trajectories']:
            keys=('id','family','powerup','split','time_s','actual_position','predicted_position',
                  'actual_velocity','predicted_velocity','actual_spin','predicted_spin','position_error',
                  'max_position_error_u','final_errors','truncated_external_samples','rollout_failure','control_basis')
            trajectories.append(dict(source=directory.name+('/'+case['source_dataset'] if case.get('source_dataset') else ''),
                                     **{key:case[key] for key in keys}))
            trajectories[-1]['native_use_events']=case.get('native_use_events',[])
            trajectories[-1]['ambiguous_contact_intervals']=case.get('ambiguous_contact_intervals',[])
            trajectories[-1]['outcomes']=case.get('outcomes',{})
            trajectories[-1]['pair_group']=case.get('pair_group')
            trajectories[-1]['intervention']=case.get('intervention')
            for key in ('prediction_available','score','pickup_decisions','held_powerup','native_powerup_activations','goal_position','goal_kind','task_goal_position','actor_goal_position','blast_meter','path_break_before',
                        'actual_radius','sample_stage','script_held_type','native_effect_active','native_effect_remaining_ms','queued_reply_before_frame','native_move'):
                if key in case:trajectories[-1][key]=case[key]
    trajectory_indices=[]
    if trajectories:
        for label,color,mode in [('Actual native motion','#45dbff','lines'),('Predicted motion','#ff6f85','lines'),
                                 ('Actual at selected time','#45dbff','markers'),('Prediction at selected time','#ff6f85','markers')]:
            trajectory_indices.append(len(fig.data))
            fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode=mode,name=label,visible=False,
                                      line=dict(color=color,width=7),marker=dict(color=color,size=6),
                                      hovertemplate=label+'<br>t=%{customdata[0]:.3f}s<br>position error=%{customdata[1]:.5f}u'+
                                      '<br>speed=%{customdata[2]:.3f}u/s<br>spin=%{customdata[3]:.3f}rad/s<extra></extra>'))
        trajectory_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='markers',name='Current goal',visible=False,
                                  marker=dict(color='#d18bff',size=8,symbol='diamond'),
                                  hovertemplate='Current goal (%{customdata})<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'))
        trajectory_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='markers',name='Real task goal',visible=False,
                                  marker=dict(color='#82e89a',size=7,symbol='square'),
                                  hovertemplate='Logged real task<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'))
    radius_trace=None
    if any('actual_radius' in c for c in trajectories):
        radius_trace=len(fig.data)
        fig.add_trace(go.Mesh3d(x=[],y=[],z=[],i=[],j=[],k=[],name='Native marble radius',
            color='#45dbff',opacity=.35,visible=False,hoverinfo='skip'))
    fig.update_layout(height=800,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='#101923',
                      font=dict(color='#cbd6e2'),showlegend=False,
                      scene=dict(bgcolor='#101923',aspectmode='data',dragmode='orbit',
                                 xaxis=dict(title='X (game units)',backgroundcolor='#172633',gridcolor='#34495e',zerolinecolor='#5a7790'),
                                 yaxis=dict(title='Y (game units)',backgroundcolor='#172633',gridcolor='#34495e',zerolinecolor='#5a7790'),
                                 zaxis=dict(title='Z (game units)',backgroundcolor='#172633',gridcolor='#34495e',zerolinecolor='#5a7790'),camera=dict(up=dict(x=0,y=0,z=1),
                                 eye=dict(x=1.4,y=-1.8,z=1.2))),uirevision='map')
    points = np.asarray([x['position'] for x in info.get('mission_annotations',[]) if
                         x['class_name']=='Item' or 'spawn' in x['datablock'].lower()])
    focus = dict(x=0.,y=0.,z=0.)
    factor = 1.
    if len(points):
        world_extent = np.max(mesh.bounds[1]-mesh.bounds[0])
        point_center = (points.min(axis=0)+points.max(axis=0))*.5
        world_center = mesh.bounds.mean(axis=0)
        focus = dict(zip(('x','y','z'), ((point_center-world_center)/world_extent).tolist()))
        factor = float(np.clip(np.max(points.max(axis=0)-points.min(axis=0))/world_extent,.10,1.))
        fig.update_layout(scene_camera=dict(up=dict(x=0,y=0,z=1),center=focus,
                          eye={k:focus[k]+v*factor for k,v in zip(('x','y','z'),(1.4,-1.8,1.2))}))
    fragment = fig.to_html(full_html=False,include_plotlyjs='plotly.min.js',div_id='geometry',
                           config=dict(displaylogo=False,scrollZoom=True,responsive=True))
    links = ''.join('<option value="'+html.escape(n)+'.html"'+(' selected' if n==path.stem else '')+'>'+html.escape(n.replace('_Hunt',''))+'</option>' for n in all_names)
    source_info = dict(mesh_sha256=info['mesh_sha256'],map=info['map'],triangles=info['triangles'],
                       objects=info['objects'],materials=info['materials'],bounds=mesh.bounds.tolist())
    payload = json.dumps(dict(records=records,info=source_info,mesh_count=mesh_count,
                              annotations=annotation_indices,ambiguities=ambiguities,ambiguity_traces=ambiguity_indices,
                              trajectories=trajectories,trajectory_traces=trajectory_indices,radius_trace=radius_trace,
                              focus=focus,focus_factor=factor),separators=(',',':')).replace('</','<\\/')
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — collision geometry</title><style>
*{box-sizing:border-box}body{margin:0;background:#101923;color:#dce5ee;font:14px system-ui,sans-serif}a{color:#79c8ff}
header{padding:16px 22px;border-bottom:1px solid #314353}h1{font-size:21px;margin:0 0 8px}p{margin:6px 0;line-height:1.5}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 310px}.controls{padding:16px;border-left:1px solid #314353;max-height:85vh;overflow:auto}
select,button,input{background:#263747;color:#e5edf5;border:1px solid #577083;border-radius:4px;padding:6px}button{cursor:pointer}
label{display:block;margin:10px 0}#objects label{font-size:12px;margin:6px 0}#objects input{margin-right:5px}pre{white-space:pre-wrap;word-break:break-word;font:12px ui-monospace,monospace}
details{margin-top:16px}summary{cursor:pointer}#geometry{height:calc(100vh - 155px);min-height:500px;width:100%}.muted{color:#9faebc;font-size:12px}
#ambiguity-panel{border:1px solid #ff694b;border-radius:6px;padding:10px;margin:14px 0}#ambiguity-list button{display:block;width:100%;margin:7px 0;text-align:left}#ambiguity-panel strong{color:#ffab8c}
#trajectory{width:100%;min-width:0}#trajectory-panel{margin-top:14px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.controls{border-left:0;max-height:none}#geometry{height:65vh}}
</style><header><h1>__TITLE__: exact collision surfaces</h1>
<select id="map">__LINKS__</select> <a href="index.html">Map gallery</a> · <a href="__MESH_LINK__">Model mesh</a> · <a href="__META_LINK__">Metadata</a> · <a href="__MAP__.obj">OBJ</a>
<p>__COUNT__ triangles. Every exported contact orientation is included. Drag to orbit, scroll to zoom; use the toolbar to pan.</p>
<p class="muted">__VALIDATION__ · __PHYSICS__ live contacts. Markers show mission definitions, not current item availability.</p></header>
<div class="layout"><main>__PLOT__</main><aside class="controls">
<label>Color <select id="color"><option value="material">Physical material</option><option value="orientation">Surface direction</option><option value="object">Object</option></select></label>
<p class="muted">Material: gray default, blue low friction, green high friction, pink force, purple high restitution.<br>Direction: green up, amber side, purple down.</p>
<label>Opacity <input id="opacity" type="range" min="0.12" max="1" step="0.02" value="1"></label>
<label>Drag <select id="drag"><option value="orbit">Orbit</option><option value="pan">Pan</option></select></label>
<label><input id="edges" type="checkbox"> Polygon boundaries</label>
<label><input id="markers" type="checkbox" checked> Gem / powerup / spawn definitions</label>
<button id="reset">Focus game area</button> <button id="all">Fit all geometry</button> <button id="top">Top view</button> <button id="under">From below</button>
<section id="trajectory-panel" hidden><strong>Actual / predicted motion</strong>
<label>Recording <select id="trajectory"><option value="">Choose a trajectory</option></select></label>
<p class="muted" id="trajectory-explanation">Cyan: native marble. Coral: prediction. Matching paths overlap. Lines follow marble centers. Hover for speed and spin.</p>
<button id="trajectory-focus">Focus trajectory</button> <button id="trajectory-worst">Largest error</button>
<label>Time <input id="trajectory-time" type="range" min="0" max="1" step="1" value="0" disabled></label>
<pre id="trajectory-readout"></pre><pre id="trajectory-info"></pre>
<p class="muted">Each recording's notes describe its controls and whether predictions run open loop or restart from observations. Contact uncertainty is not calibrated; these traces do not certify a safe landing.</p></section>
<section id="ambiguity-panel" hidden><strong>Material ambiguities</strong>
<label><input id="ambiguities" type="checkbox" checked> Show orange ambiguity callouts</label>
<p class="muted">The stem ends at the exact game contact. Labels sit 4 units above it. Hover for the friction / restitution difference, or focus a contact below.</p>
<div id="ambiguity-list"></div><a href="__VALIDATION_LINK__">Full validation records</a></section>
<details open><summary>Objects (visibility only)</summary><button id="showall">Show all</button> <div id="objects"></div></details>
<details><summary>Source identity</summary><pre id="identity"></pre></details>
<details open><summary>Selected surface</summary><pre id="selected">Click a surface to inspect its normal and collision material.</pre></details>
<p class="muted">No geometry is deleted by visibility controls. A static map is not a dynamics predictor; moving actors and effects need separate state. Material winners at shared seams can depend on native collision ordering.</p>
</aside></div><script>
const model=__PAYLOAD__;const gd=document.getElementById('geometry');
// Plotly restyles can mutate full-layout axis ranges; camera coordinates use the original world frame.
const cameraFrame={ranges:['x','y','z'].map(axis=>gd._fullLayout.scene[axis+'axis'].range.slice()),aspect:{...gd._fullLayout.scene.aspectratio}};
document.getElementById('map').onchange=e=>location.href=e.target.value;
document.getElementById('identity').textContent='NPZ SHA-256\n'+model.info.mesh_sha256+'\nWorld coordinates: z up, game units.\nViewer arrays come from this hash-checked NPZ.';
const box=document.getElementById('objects');
model.records.forEach((r,i)=>{const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;
input.onchange=()=>Plotly.restyle(gd,{visible:input.checked},[i]);const source=String(r.object.datablock||r.object.interior_file||r.object.shape_name||r.object.id).split(/[\\/]/).pop();label.append(input,document.createTextNode(r.object.class+' '+source+' ('+r.ids.length+')'));box.append(label);});
document.getElementById('showall').onclick=()=>{box.querySelectorAll('input').forEach(x=>x.checked=true);Plotly.restyle(gd,{visible:true},model.records.map((_,i)=>i));};
document.getElementById('color').onchange=e=>model.records.forEach((r,i)=>Plotly.restyle(gd,{facecolor:[e.target.value==='material'?r.material_colors:e.target.value==='orientation'?r.orientation_colors:r.ids.map(()=>r.object_color)]},[i]));
document.getElementById('opacity').oninput=e=>Plotly.restyle(gd,{opacity:Number(e.target.value)},model.records.map((_,i)=>i));
document.getElementById('drag').onchange=e=>Plotly.relayout(gd,{'scene.dragmode':e.target.value});
document.getElementById('edges').onchange=e=>Plotly.restyle(gd,{visible:e.target.checked},[model.mesh_count]);
document.getElementById('markers').onchange=e=>Plotly.restyle(gd,{visible:e.target.checked},model.annotations);
document.getElementById('ambiguities').onchange=e=>Plotly.restyle(gd,{visible:e.target.checked},model.ambiguity_traces);
function camera(eye,up={x:0,y:0,z:1},center={x:0,y:0,z:0}){return Plotly.relayout(gd,{'scene.camera':{eye,up,center}});}
function focusEye(x,y,z){return {x:model.focus.x+x*model.focus_factor,y:model.focus.y+y*model.focus_factor,z:model.focus.z+z*model.focus_factor};}
document.getElementById('reset').onclick=()=>camera(focusEye(1.4,-1.8,1.2),{x:0,y:0,z:1},model.focus);
document.getElementById('all').onclick=()=>camera({x:1.4,y:-1.8,z:1.2});
document.getElementById('top').onclick=()=>camera(focusEye(0,0,2.2),{x:0,y:1,z:0},model.focus);
document.getElementById('under').onclick=()=>camera(focusEye(1.4,-1.8,-1.2),{x:0,y:0,z:1},model.focus);
let activeTrajectory=null;
const trajectorySelect=document.getElementById('trajectory'),timeSlider=document.getElementById('trajectory-time');
function traceArrays(points){return {x:[points.map(p=>p[0])],y:[points.map(p=>p[1])],z:[points.map(p=>p[2])]};}
function motionNorm(v){return v.every(Number.isFinite)?Math.hypot(...v):null;}
function nativeSphere(position,radius){const x=[],y=[],z=[],i=[],j=[],k=[],lat=12,lon=24;
for(let a=0;a<=lat;a++){const p=Math.PI*a/lat;for(let b=0;b<lon;b++){const q=2*Math.PI*b/lon;
x.push(position[0]+radius*Math.sin(p)*Math.cos(q));y.push(position[1]+radius*Math.sin(p)*Math.sin(q));z.push(position[2]+radius*Math.cos(p));}}
for(let a=0;a<lat;a++)for(let b=0;b<lon;b++){const c=a*lon+b,d=a*lon+(b+1)%lon,e=c+lon,f=d+lon;
i.push(c,d);j.push(e,e);k.push(d,f);}return {x:[x],y:[y],z:[z],i:[i],j:[j],k:[k]};}
function trajectoryHover(c,basis){return c.time_s.map((time,i)=>[time,c.position_error[i],motionNorm(c[basis+'_velocity'][i]),motionNorm(c[basis+'_spin'][i])]);}
function trajectoryLine(c,basis){const points=[],hover=[],samples=trajectoryHover(c,basis);
c[basis+'_position'].forEach((p,i)=>{if(c.path_break_before&&c.path_break_before[i]){
points.push([null,null,null]);hover.push([null,null,null,null]);}points.push(p);hover.push(samples[i]);});
return {points,hover};}
async function scrubTrajectory(index){if(!activeTrajectory)return;const c=activeTrajectory;
timeSlider.value=index;document.getElementById('trajectory-readout').textContent=JSON.stringify({time_s:c.time_s[index],
position_error_u:c.position_error[index],game_score:c.score?c.score[index]:undefined,
held_powerup:c.held_powerup?c.held_powerup[index]:undefined,
actual_radius:c.actual_radius?c.actual_radius[index]:undefined,
sample_stage:c.sample_stage?c.sample_stage[index]:undefined,
script_held_type:c.script_held_type?c.script_held_type[index]:undefined,
native_effect_active:c.native_effect_active?c.native_effect_active[index]:undefined,
native_effect_remaining_ms:c.native_effect_remaining_ms?c.native_effect_remaining_ms[index]:undefined,
queued_reply_before_frame:c.queued_reply_before_frame?c.queued_reply_before_frame[index]:undefined,
native_move:c.native_move?c.native_move[index]:undefined,
blast_meter:c.blast_meter?c.blast_meter[index]:undefined,
goal:c.goal_position?c.goal_position[index]:undefined,
task_goal:c.task_goal_position?c.task_goal_position[index]:undefined,
actor_input_goal:c.actor_goal_position?c.actor_goal_position[index]:undefined,
prediction_available:c.prediction_available?c.prediction_available[index]:true,
actual_velocity:c.actual_velocity[index],predicted_velocity:c.predicted_velocity[index],
actual_spin:c.actual_spin[index],predicted_spin:c.predicted_spin[index]},null,2);
for(let j=0;j<2;j++){const basis=j===0?'actual':'predicted';await Plotly.restyle(gd,
{...traceArrays([c[basis+'_position'][index]]),customdata:[[trajectoryHover(c,basis)[index]]],visible:c[basis+'_position'][index].every(Number.isFinite)},[model.trajectory_traces[j+2]]);}
if(model.trajectory_traces.length>4){const goal=c.goal_position?c.goal_position[index]:[null,null,null];
await Plotly.restyle(gd,{...traceArrays([goal]),customdata:[[c.goal_kind?c.goal_kind[index]:'gem']],visible:goal.every(Number.isFinite)},[model.trajectory_traces[4]]);}
if(typeof model.radius_trace==='number'){const radius=c.actual_radius?c.actual_radius[index]:null;
if(Number.isFinite(radius)&&radius>0)await Plotly.restyle(gd,{...nativeSphere(c.actual_position[index],radius),visible:true},[model.radius_trace]);
else await Plotly.restyle(gd,{visible:false},[model.radius_trace]);}
if(model.trajectory_traces.length>5){const task=c.task_goal_position?c.task_goal_position[index]:[null,null,null];
await Plotly.restyle(gd,{...traceArrays([task]),visible:task.every(Number.isFinite)},[model.trajectory_traces[5]]);}}
async function selectTrajectory(index){if(index===''){activeTrajectory=null;timeSlider.disabled=true;
document.getElementById('trajectory-info').textContent='';document.getElementById('trajectory-readout').textContent='';
if(model.trajectory_traces.length)await Plotly.restyle(gd,{visible:false},model.trajectory_traces);
if(typeof model.radius_trace==='number')await Plotly.restyle(gd,{visible:false},[model.radius_trace]);return;}
const c=model.trajectories[Number(index)];activeTrajectory=c;timeSlider.disabled=false;timeSlider.max=c.time_s.length-1;
const recordedOnly=c.prediction_available && !c.prediction_available.some(Boolean);
document.querySelector('#trajectory-panel strong').textContent=recordedOnly?'Recorded native response':'Actual / predicted motion';
document.getElementById('trajectory-explanation').textContent=recordedOnly?'Cyan: recorded marble centers. The translucent sphere uses measured radius. Hover for speed and spin; scrub for inventory and effects.':'Cyan: native marble. Coral: prediction. Matching paths overlap. Lines follow marble centers. Hover for speed and spin.';
document.getElementById('trajectory-info').textContent=JSON.stringify({source:c.source,trial:c.id,family:c.family,requested_powerup:c.powerup,split:c.split,
duration_s:c.time_s[c.time_s.length-1],max_position_error_u:c.max_position_error_u,final_errors:c.final_errors,
truncated_external_samples:c.truncated_external_samples,rollout_failure:c.rollout_failure,native_use_events:c.native_use_events,
ambiguous_contact_intervals:c.ambiguous_contact_intervals,pair_group:c.pair_group,intervention:c.intervention,
outcomes:c.outcomes,pickup_decisions:c.pickup_decisions,
native_powerup_activations:c.native_powerup_activations,
control_basis:c.control_basis},null,2);
for(let j=0;j<2;j++){const basis=j===0?'actual':'predicted',line=trajectoryLine(c,basis);await Plotly.restyle(gd,
{...traceArrays(line.points),customdata:[line.hover],visible:true},[model.trajectory_traces[j]]);}
await scrubTrajectory(0);}
trajectorySelect.onchange=e=>selectTrajectory(e.target.value);
timeSlider.oninput=e=>scrubTrajectory(Number(e.target.value));
document.getElementById('trajectory-worst').onclick=()=>{if(!activeTrajectory)return;const errors=activeTrajectory.position_error;
const known=errors.filter(Number.isFinite);if(!known.length)return;
scrubTrajectory(errors.indexOf(Math.max(...known)));};
document.getElementById('trajectory-focus').onclick=()=>{if(!activeTrajectory)return;
const points=activeTrajectory.actual_position.concat(activeTrajectory.predicted_position).filter(p=>p.every(Number.isFinite)),center={};let factor=.04;
['x','y','z'].forEach((axis,i)=>{const values=points.map(p=>p[i]),lo=Math.min(...values),hi=Math.max(...values),range=cameraFrame.ranges[i];
center[axis]=((lo+hi)*.5-(range[0]+range[1])*.5)/(range[1]-range[0])*cameraFrame.aspect[axis];
factor=Math.max(factor,(hi-lo)/(range[1]-range[0])*cameraFrame.aspect[axis]);});
camera({x:center.x+1.3*factor,y:center.y-1.6*factor,z:center.z+factor},{x:0,y:0,z:1},center);};
if(model.trajectories.length){document.getElementById('trajectory-panel').hidden=false;
model.trajectories.forEach((c,i)=>{const option=document.createElement('option');option.value=i;
option.textContent=c.source+' / '+c.family+' #'+c.id+' / '+c.powerup.replace('Item','')+(c.intervention?' / '+JSON.stringify(c.intervention):'')+' / max '+(Number.isFinite(c.max_position_error_u)?c.max_position_error_u.toFixed(3)+'u':'unknown');
option.title=option.textContent;
trajectorySelect.append(option);});}
function selectAmbiguity(index){const a=model.ambiguities[index];
document.getElementById('selected').textContent=JSON.stringify({label:a.label,test_id:a.test_id,contact:a.point,
game_material:{friction:a.engine.friction,restitution:a.engine.restitution,force:a.engine.force},
query_material:a.mesh.material,triangle_id:a.mesh.triangle_id,object_id:a.mesh.object_id},null,2);}
async function focusAmbiguity(index){const a=model.ambiguities[index],center={};
['x','y','z'].forEach((axis,i)=>{const range=cameraFrame.ranges[i];
center[axis]=(a.point[i]+(axis==='z'?1:0)-(range[0]+range[1])*.5)/(range[1]-range[0])*cameraFrame.aspect[axis];});
document.getElementById('ambiguities').checked=true;
if(model.ambiguity_traces.some(i=>gd.data[i].visible===false))await Plotly.restyle(gd,{visible:true},model.ambiguity_traces);
await camera({x:center.x+.22,y:center.y-.28,z:center.z+.22},{x:0,y:0,z:1},center);selectAmbiguity(index);}
if(model.ambiguities.length){document.getElementById('ambiguity-panel').hidden=false;
const list=document.getElementById('ambiguity-list');model.ambiguities.forEach((a,i)=>{const button=document.createElement('button');
button.textContent=a.label+' · '+a.point.map(x=>x.toFixed(2)).join(', ');button.onclick=()=>focusAmbiguity(i);list.append(button);});}
gd.on('plotly_click',e=>{const p=e.points[0];if(model.ambiguity_traces.includes(p.curveNumber)&&p.customdata!==undefined){selectAmbiguity(p.customdata);return;}
const r=model.records[p.curveNumber];if(!r)return;
const id=p.customdata===undefined?r.ids[Math.floor(p.pointNumber/3)]:p.customdata,j=r.ids.indexOf(id);if(j<0)return;
document.getElementById('selected').textContent=JSON.stringify({triangle_id:id,object_id:r.object.id,class:r.object.class,source:r.object.interior_file||r.object.shape_name||r.object.shape_file,datablock:r.object.datablock,normal:r.normals[j],material:model.info.materials[r.material_ids[j]],surface_key:r.surface_keys[j]},null,2);});
</script></html>'''
    replacements = dict(TITLE=html.escape(path.stem.replace('_Hunt','')),LINKS=links,MAP=path.stem,
                        MESH_LINK=html.escape(Path(os.path.relpath(path.resolve(),output.resolve())).as_posix(),quote=True),
                        META_LINK=html.escape(Path(os.path.relpath(path.with_suffix('.json').resolve(),output.resolve())).as_posix(),quote=True),
                        VALIDATION_LINK=html.escape(Path(os.path.relpath(path.with_name(path.stem+'.validation.json').resolve(),output.resolve())).as_posix(),quote=True),
                        COUNT=f'{len(mesh.triangles):,}',VALIDATION=f"{sweep['sweeps']:,} native swept-sphere checks",
                        PHYSICS=f"{physics['contacts']}/{physics['accepted']}",PLOT=fragment,PAYLOAD=payload)
    for key,value in replacements.items():
        page = page.replace('__'+key+'__',value)
    dest = output/(path.stem+'.html')
    write_atomic(dest,page)
    obj = ['# Exact model collision mesh; z-up, game units', '# NPZ SHA256 '+info['mesh_sha256']]
    for i,t in enumerate(mesh.triangles):
        if i==0 or mesh.object_ids[i]!=mesh.object_ids[i-1]:
            obj.append('g object_'+str(mesh.object_ids[i]))
        for v in t:
            obj.append('v '+' '.join(format(float(x),'.17g') for x in v))
        obj.append('f '+' '.join(str(i*3+j+1) for j in range(3)))
    write_atomic(output/(path.stem+'.obj'),'\n'.join(obj)+'\n')
    # Separate evidence makes coordinate/identity checks possible without a browser.
    atomic_json(output/(path.stem+'.viewer.json'),dict(mesh_sha256=info['mesh_sha256'],
                 html_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                 displayed_triangles=sum(len(r['ids']) for r in records),
                 geometry_sha256=hashlib.sha256(mesh.triangles.astype('<f8').tobytes()).hexdigest(),
                 max_native_plane_projection_u=mesh.max_plane_projection_u,
                 material_ambiguities=ambiguities,
                 trajectory_count=len(trajectories),trajectory_sources=trajectory_sources,
                 coordinate_frame=info['coordinate_frame'],decimation=False))
    return dict(map=path.stem,triangles=len(mesh.triangles),sweeps=sweep['sweeps'],
                contacts=physics['contacts'],trials=physics['accepted'],
                exceptions={k:v for k,v in sweep['counts'].items() if k not in ('hit','miss')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('ml_agent/collision_maps'))
    parser.add_argument('--trajectories',nargs='+',type=Path,default=[],help='Optional dynamics-report directories; all geometry hashes must match')
    args = parser.parse_args()
    names = json.loads((Path(__file__).resolve().parents[2]/'configs/real_maps_20261009.json').read_text())['maps']
    out = args.directory/'viewer'
    out.mkdir(parents=True,exist_ok=True)
    write_atomic(out/'plotly.min.js',get_plotlyjs())
    rows = [build_map(args.directory/(name+'.npz'),out,names,args.trajectories) for name in names]
    body = ''.join('<tr><td><a href="'+r['map']+'.html">'+html.escape(r['map'].replace('_Hunt',''))+'</a></td><td>'+format(r['triangles'],',')+'</td><td>'+format(r['sweeps'],',')+'</td><td>'+str(r['contacts'])+'/'+str(r['trials'])+'</td><td>'+('<a href="'+r['map']+'.html">'+html.escape(str(r['exceptions']))+' — view orange markers</a>' if r['exceptions'] else 'none')+'</td></tr>' for r in rows)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Ten-map collision review</title><style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:40px;line-height:1.6}a{color:#79c8ff}td,th{padding:9px 18px;text-align:left;border-bottom:1px solid #314353}table{border-collapse:collapse}h1{margin-bottom:8px}p{max-width:900px}</style><h1>Ten-map collision review</h1><p>Open a map to orbit, zoom, pan, view from below, show polygon boundaries, change material colors, or hide individual objects. The viewer uses the exact saved model mesh, including glass, walls and undersides.</p><p>Use “Recording” to compare native motion (cyan) with predicted motion (coral), then “Focus trajectory” and the time slider. “Largest error” shows where a prediction fails. On Concentric, Cube Isle, Sprawl and Bowl, intervention recordings compare matched spins, jump timings and powerups. Cube Isle also includes alternate rebound hypotheses. These are mechanics experiments; the policy has not learned them yet.</p><p><a href="fast_navigator/index.html">Fast learned navigator: completed ten-map native trajectories and reference comparison</a>. Candidate remains experimental; consult its handoff for current checkpoints.</p><table><thead><tr><th>Map</th><th>Triangles</th><th>Native sweeps</th><th>Live contacts</th><th>Sweep exceptions</th></tr></thead><tbody>'+body+'</tbody></table><p>All triangles were tested with native marble swept-sphere collisions; additional checks use marble-size spheres and edges. Exception records are preserved in each map’s validation JSON. Sight-ray/BSP differences are recorded separately and do not certify contact collision. These geometry files are used by the experimental fast full-map navigator. Its learned attention queries all saved triangles; general playing strength is still being tested.</p><p>Everything loads locally, with no internet connection or server needed. Each map also provides its model NPZ, source metadata and OBJ geometry.</p></html>'
    write_atomic(out/'index.html',page)
    atomic_json(out/'summary.json',dict(maps=rows,training_started=False))
    print('Saved '+str(out/'index.html'))


if __name__=='__main__':
    main()
