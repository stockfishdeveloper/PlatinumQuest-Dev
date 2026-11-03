"""Private native skill controller; the production simulator DLL is untouched."""
import ctypes
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import ROOT,RUNTIME


@lru_cache(maxsize=1)
def library_path():
    source=Path(__file__).with_name('edge_execution.cpp');native=ROOT/'ml_agent/nav/native'
    digest=hashlib.sha256(source.read_bytes()+(native/'marble_sim.cpp').read_bytes()+(native/'controllers.inc').read_bytes()).hexdigest()[:16]
    return RUNTIME/'compiled'/digest/'edge_execution.dll'


def build():
    out=library_path()
    if out.exists():return out
    out.parent.mkdir(parents=True,exist_ok=True)
    setup=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat')
    source=Path(__file__).with_name('edge_execution.cpp')
    script=out.with_suffix('.cmd')
    script.write_text('@echo off\ncall "'+str(setup)+'" -arch=x64 -host_arch=x64 >nul\nif errorlevel 1 exit /b 1\n'
                      'cl /nologo /LD /O2 /fp:strict /std:c++17 /EHsc "'+str(source)+'" /Foedge_execution.obj /Feedge_execution.dll\n',encoding='utf-8')
    env=dict(os.environ);env['PATH']='C:/Program Files (x86)/Microsoft Visual Studio/Installer;'+env.get('PATH','')
    result=subprocess.run(['cmd.exe','/c',str(script)],cwd=str(out.parent),capture_output=True,text=True,env=env,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    (out.parent/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode or not out.exists():raise RuntimeError(result.stdout+result.stderr)
    return out


def attach(engine):
    from nav.native.marble_sim import NativeSim
    path=library_path()
    if getattr(engine,'edge_library',None)==str(path):return engine.native_sim
    if not path.exists():raise FileNotFoundError('Build the private edge controller first: '+str(path))
    sim=NativeSim(engine,dll=path);L=sim.lib;P=ctypes.c_void_p;I=ctypes.c_int
    L.ms_edge_many.argtypes=[P]*8+[I,I,I]+[P]*5;L.ms_edge_many.restype=I
    L.ms_edge_contacts.argtypes=[P]*4+[I,P,I,P,P];L.ms_edge_contacts.restype=I
    engine.native_sim=sim;engine.edge_library=str(path)
    return sim


def geometry(case):
    c=case['cell'];yaw=math.radians(c['yaw_deg']);theta=c['landing_yaw']
    inverse=c.get('landing_inverse',[[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    return np.asarray([*c['origin'],math.cos(yaw),math.sin(yaw),c['lip_z'],c['width'],
                       *c['landing_center'],*c['landing_size'],math.cos(theta),math.sin(theta),
                       *case['goal'],*case['exit_velocity'],.189750001,case.get('goal_radius',.9),case.get('start_floor',0.),
                       float(bool(case.get('airborne') or case.get('jump_committed'))),float(case.get('start_kind')=='roll'),
                       float(case.get('stable_inputs',False)),c.get('lip_slope_y',0.),*c.get('landing_slope',[0.,0.]),
                       *np.asarray(inverse).reshape(-1)],dtype=float)


def predict(engine,case,d,params,steps=128,early_stop=False):
    from nav.native.forecast import _start,_blast_array,_inventory_array,_consts,_finish
    from nav.native.marble_sim import _ptr,S_SIZE
    sim=attach(engine);state,blast,inventory,bridge=_start(d);s=np.empty(S_SIZE);sim.pack(state,s)
    par=np.asarray([list(p)+[0.]*(10-len(p)) for p in params],dtype=float);n=len(par)
    rows=np.empty((n,steps*4+1,S_SIZE));replies=np.zeros((n,steps,40));bo=np.zeros(n)
    statuses=np.zeros((n,8),dtype=np.int32);memory=np.zeros((n,7))
    sim.lib.ms_edge_many(sim.handle,_ptr(s),_ptr(bridge),_ptr(_blast_array(blast)),_ptr(_inventory_array(inventory)),
                         _ptr(_consts()),_ptr(geometry(case)),_ptr(par),n,steps,int(early_stop),_ptr(rows),_ptr(replies),_ptr(bo),_ptr(statuses),_ptr(memory))
    out=[]
    for i in range(n):
        count=int(statuses[i,0]) if early_stop and statuses[i,2]<0 else steps
        # Each retained candidate owns its rows; keeping one winner must not
        # retain the entire search batch's large allocation.
        packed=rows[i,:statuses[i,1]].copy()
        r=_finish(d,state,blast,count,False,packed,replies[i],bo[i:i+1],statuses[i]);r['requested_replies']=steps
        r['contract_truncated']=count<steps;out.append(r)
    return out,memory


def contact_events(engine,d,rollout,until=None):
    from nav.native.forecast import _start
    from nav.native.marble_sim import _ptr,S_SIZE
    sim=attach(engine);state,_,_,bridge=_start(d);packed=np.empty(S_SIZE);sim.pack(state,packed)
    ticks=min(len(rollout['native_states'])-1,until if until is not None else 1<<30)
    events=np.empty((max(1,ticks)*32,20));states=np.empty((ticks,S_SIZE));count=ctypes.c_int()
    code=sim.lib.ms_edge_contacts(sim.handle,_ptr(packed),_ptr(bridge),_ptr(rollout['reply_table']),ticks,
                                 _ptr(events),len(events),ctypes.byref(count),_ptr(states))
    if code:raise ValueError('Independent edge contact replay failed: %d'%code)
    return events[:count.value],states


if __name__=='__main__':print(build())
