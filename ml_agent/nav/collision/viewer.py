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


def build_map(path, output, all_names):
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
@media(max-width:800px){.layout{grid-template-columns:1fr}.controls{border-left:0;max-height:none}#geometry{height:65vh}}
</style><header><h1>__TITLE__: exact collision surfaces</h1>
<select id="map">__LINKS__</select> <a href="index.html">All ten maps</a> · <a href="../__MAP__.npz">Model mesh</a> · <a href="../__MAP__.json">Metadata</a> · <a href="__MAP__.obj">OBJ</a>
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
<section id="ambiguity-panel" hidden><strong>Material ambiguities</strong>
<label><input id="ambiguities" type="checkbox" checked> Show orange ambiguity callouts</label>
<p class="muted">The stem ends at the exact game contact. Labels sit 4 units above it. Hover for the friction / restitution difference, or focus a contact below.</p>
<div id="ambiguity-list"></div><a href="../__MAP__.validation.json">Full validation records</a></section>
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
                 coordinate_frame=info['coordinate_frame'],decimation=False))
    return dict(map=path.stem,triangles=len(mesh.triangles),sweeps=sweep['sweeps'],
                contacts=physics['contacts'],trials=physics['accepted'],
                exceptions={k:v for k,v in sweep['counts'].items() if k not in ('hit','miss')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('ml_agent/collision_maps'))
    args = parser.parse_args()
    names = json.loads((Path(__file__).resolve().parents[2]/'configs/real_maps_20261009.json').read_text())['maps']
    out = args.directory/'viewer'
    out.mkdir(parents=True,exist_ok=True)
    write_atomic(out/'plotly.min.js',get_plotlyjs())
    rows = [build_map(args.directory/(name+'.npz'),out,names) for name in names]
    body = ''.join('<tr><td><a href="'+r['map']+'.html">'+html.escape(r['map'].replace('_Hunt',''))+'</a></td><td>'+format(r['triangles'],',')+'</td><td>'+format(r['sweeps'],',')+'</td><td>'+str(r['contacts'])+'/'+str(r['trials'])+'</td><td>'+('<a href="'+r['map']+'.html">'+html.escape(str(r['exceptions']))+' — view orange markers</a>' if r['exceptions'] else 'none')+'</td></tr>' for r in rows)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Ten-map collision review</title><style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:40px;line-height:1.6}a{color:#79c8ff}td,th{padding:9px 18px;text-align:left;border-bottom:1px solid #314353}table{border-collapse:collapse}h1{margin-bottom:8px}p{max-width:900px}</style><h1>Ten-map collision review</h1><p>Open a map to orbit, zoom, pan, view from below, show polygon boundaries, change material colors, or hide individual objects. The viewer uses the exact saved model mesh, including glass, walls and undersides.</p><table><thead><tr><th>Map</th><th>Triangles</th><th>Native sweeps</th><th>Live contacts</th><th>Sweep exceptions</th></tr></thead><tbody>'+body+'</tbody></table><p>All triangles were tested with native marble swept-sphere collisions; additional checks use marble-size spheres and edges. Exception records are preserved in each map’s validation JSON. Sight-ray/BSP differences are recorded separately and do not certify contact collision. These geometry files are ready for review and queries; the trained policy has not yet been connected to them.</p><p>Everything loads locally, with no internet connection or server needed. Each map also provides its model NPZ, source metadata and OBJ geometry.</p></html>'
    write_atomic(out/'index.html',page)
    atomic_json(out/'summary.json',dict(maps=rows,training_started=False))
    print('Saved '+str(out/'index.html'))


if __name__=='__main__':
    main()
