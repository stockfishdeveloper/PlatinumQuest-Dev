"""Build a self-contained, 1x viewer of recorded native trajectories (not rendered game video)."""
import argparse
import gzip
import json

import numpy as np

from .common import NIGHT,engine_for,local,rotation

HTML=r'''<!doctype html><meta charset="utf-8"><title>Edge-hit skill review</title>
<style>body{font:16px system-ui;background:#101826;color:#e6eefb;margin:24px}button,select,input{font:inherit;margin:5px;padding:6px;background:#20314a;color:white;border:1px solid #57749a;border-radius:5px}canvas{width:100%;max-width:1200px;background:#142136;border:1px solid #426084}#note{max-width:1100px;color:#b6c9e5}#status{margin:12px 0}a{color:#85ceff}</style>
<h1>Edge-hit skill: native trial review</h1>
<p id="note">Recorded game positions and collision geometry. Playback follows the native simulation clock at 1x; this is a trace viewer, not game footage. Cyan is the marble, green is the landing region, orange marks the verified lip contact. The selected file's trials are listed below. Complete trial logs, including declines and failures, remain alongside the report.</p>
<select id="case"></select><button id="play">Play at 1x</button><button id="slow">0.25x</button><button id="restart">Restart</button>
<input id="time" type="range" min="0" max="256" value="0" style="width:35%"><span id="clock"></span>
<div id="status"></div><canvas id="view" width="1200" height="740"></canvas>
<script>const trials=__DATA__;const sel=document.getElementById('case'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),slider=document.getElementById('time');let trial=trials[0],t=0,playing=false,speed=1,last=0;
trials.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${v.id} — ${v.success?'SUCCESS':'MISS'}${v.accepted?' (accepted)':' (declined)'}`;sel.add(o)});
function choose(){trial=trials[+sel.value];t=0;playing=false;slider.max=trial.p.length-1;document.getElementById('status').textContent=`${trial.mode} approach | ${trial.success?'Passed local contact + landing + exit checks':'Did not pass all checks'} | start ${trial.distance.toFixed(1)}m away at ${trial.speed.toFixed(1)}m/s | ${trial.replans} plan changes | root: ${trial.accepted?'accepted':'declined'} | game fall: ${trial.fell?'yes':'no'} | later departure below landing: ${trial.drop?'yes':'no'}`;draw()}
sel.onchange=choose;document.getElementById('play').onclick=()=>{speed=1;playing=!playing};document.getElementById('slow').onclick=()=>{speed=.25;playing=true};document.getElementById('restart').onclick=()=>{t=0;draw()};slider.oninput=()=>{t=+slider.value*.016;playing=false;draw()};
function panel(top,axis,min,max,title){const left=55,w=1100,h=280,xmin=trial.bounds[0][0],xmax=trial.bounds[1][0];min=trial.bounds[0][axis];max=trial.bounds[1][axis];const project=p=>[left+(p[0]-xmin)/(xmax-xmin)*w,top+h-(p[axis]-min)/(max-min)*h];ctx.fillStyle='#b6c9e5';ctx.font='17px system-ui';ctx.fillText(title,left,top-12);ctx.strokeStyle='#293c55';ctx.lineWidth=1;
for(let x=Math.ceil(xmin/5)*5;x<=xmax;x+=5){ctx.beginPath();ctx.moveTo(...project([x,axis===1?min:0,axis===2?min:0]));ctx.lineTo(...project([x,axis===1?max:0,axis===2?max:0]));ctx.stroke()}
ctx.save();ctx.beginPath();ctx.rect(left,top,w,h);ctx.clip();
for(const tri of trial.mesh){ctx.beginPath();tri.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath();ctx.fillStyle='#324763';ctx.fill();ctx.strokeStyle='#567399';ctx.stroke()}
const goal=project(trial.goal);ctx.strokeStyle='#78e0ac';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(goal[0],goal[1],.85*w/(xmax-xmin),axis===1?.85*h/(max-min):.19*h/(max-min),0,0,Math.PI*2);ctx.stroke();
ctx.strokeStyle='#4b89a0';ctx.lineWidth=1.5;ctx.beginPath();trial.p.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
if(trial.contact){const q=project(trial.contact);ctx.fillStyle='#ffa652';ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fill()}
let index=Math.min(trial.p.length-1,Math.floor(t/.016));const p=trial.p[index],q=project(p);ctx.fillStyle='#7ee3ff';ctx.beginPath();ctx.ellipse(...q,.18975*w/(xmax-xmin),.18975*h/(max-min),0,0,Math.PI*2);ctx.fill();ctx.restore();}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);panel(40,2,-1,7,'Side view · height and contact');panel(420,1,-6,6,'Top view · approach and landing direction');slider.value=Math.floor(t/.016);document.getElementById('clock').textContent=t.toFixed(2)+' s';}
function frame(now){if(playing){t+=Math.min((now-last)/1000,.1)*speed;if(t>=(trial.p.length-1)*.016){t=(trial.p.length-1)*.016;playing=false}draw()}last=now;requestAnimationFrame(frame)}choose();requestAnimationFrame(frame);
</script>'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',nargs='+',required=True);ap.add_argument('--out',default='review.html')
    ap.add_argument('--showcase',type=int,default=0,help='Select varied successful high-lip examples; full logs remain available.')
    a=ap.parse_args()
    trials=[];engines={}
    for path in [p for folder in a.input for p in sorted((NIGHT/folder).glob('*.json.gz'))]:
        r=json.load(gzip.open(path,'rt'));c=r['case']['cell'];name=c['map']
        if name not in engines:engines[name]=engine_for(name)
        p=local(c,[s['position'] for s in r['native_ticks']]);comp=r['comparison'];audit=comp['result'].get('contact_audit',{});hit=audit.get('first_lip')
        mesh=local(c,engines[name].mesh.triangles)
        # Frame the actual approach and landing, including long run-ups and
        # lateral/sloped routes. Extreme failed falls are still visible in data.
        lo=np.maximum(p.min(0)-[2,2,1],[-42,-20,-3]);hi=np.minimum(p.max(0)+[3,2,1],[45,20,18])
        lo=np.minimum(lo,[-2,-3,-.5]);hi=np.maximum(hi,np.asarray(r['case']['goal'])+[4,3,1])
        keep=np.all((mesh.max(1)>lo)&(mesh.min(1)<hi),axis=1)
        finish=comp['result']['finish_tick'];slope=np.asarray(c.get('landing_slope',[0,0]));goal=np.asarray(r['case']['goal'])
        height=goal[2]+(p[:,:2]-goal[:2])@slope
        drop=bool(comp['result']['success'] and finish is not None and np.any(p[finish:,2]<height[finish:]-.10))
        trials.append(dict(id=r['case']['id'],mode=comp['result'].get('contact_mode') or r['case']['start_kind'],goal=r['case']['goal'],mesh=np.round(mesh[keep],5).tolist(),
                           p=np.round(p,6).tolist(),success=comp['result']['success'],accepted=comp['admission'],replans=len(comp.get('refreshes',[])),
                           fell=comp['fell'],drop=drop,contact=local(c,hit['point']).tolist() if hit else None,bounds=[lo.tolist(),hi.tolist()],
                           map=name,cell=c['id'],family=c['family'],tilted=bool(c.get('lip_slope_y')),start_kind=r['case']['start_kind'],
                           distance=float(-p[0,0]),speed=float(np.linalg.norm(r['native_ticks'][0]['velocity'])),source=str(path.relative_to(NIGHT))))
    if not trials:raise ValueError('No native trial records')
    if a.showcase:
        remaining=[r for r in trials if r['success'] and r['accepted'] and r['family']!='low'];selected=[];seen=set()
        def tags(r):return {('map',r['map']),('cell',r['cell']),('mode',r['mode']),('start',r['start_kind']),
                           ('tilted',r['tilted']),('distance',int(r['distance']//5)),('speed',int(r['speed']//5))}
        while remaining and len(selected)<a.showcase:
            best=max(remaining,key=lambda r:len(tags(r)-seen));remaining.remove(best);selected.append(best);seen|=tags(best)
        trials=selected
    # Representative successful modes first, followed by every other trial.
    first=[]
    for mode in ('rise','descend','roll'):
        matches=[r for r in trials if r['mode']==mode and r['success']]
        if matches:first.append(matches[0])
    trials=first+[r for r in trials if r not in first]
    (NIGHT/a.out).write_text(HTML.replace('__DATA__',json.dumps(trials,separators=(',',':'))),encoding='utf-8')
    print('Wrote',NIGHT/a.out,'with',len(trials),'native trials')


if __name__=='__main__':main()
