"""Fingerprint-bound, exact local caches for repeated static-map work.

Cache payloads are generated locally, not imported external artifacts. A single
atomic envelope binds the input fingerprint and payload hash. Native states and
predicted futures are never cached as interchangeable observations.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import time
import uuid

import numpy as np

from nav.marble_dynamics.record import ML, sha


@contextmanager
def file_lock(path, timeout=1800.):
    """Serialize shared cache builds and native launches across worker processes."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        started=time.perf_counter()
        while True:
            try:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if time.perf_counter()-started>=timeout:raise TimeoutError('Shared lock: '+str(path))
                time.sleep(.1)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def fingerprint(files, **settings):
    value=dict(schema='exact_static_map_cache_v1',files={str(Path(p).resolve()):sha(p) for p in [*files,Path(__file__)]},settings=settings)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write_cache(path, key, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=pickle.dumps(value,protocol=5)
    header=json.dumps(dict(key=key,sha256=hashlib.sha256(payload).hexdigest())).encode()
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_bytes(b'PQSTATIC1\n'+struct.pack('<I',len(header))+header+payload)
        for attempt in range(12):
            try:os.replace(str(tmp),str(path));break
            except PermissionError:
                if attempt==11:raise
                time.sleep(.05)
    finally:
        if tmp.exists():tmp.unlink()


def read_cache(path, key):
    path=Path(path)
    if not path.exists():return None
    data=path.read_bytes()
    if data[:10]!=b'PQSTATIC1\n' or len(data)<14:raise ValueError('Corrupt cache envelope: '+str(path))
    n=struct.unpack('<I',data[10:14])[0]
    if n>1048576 or 14+n>len(data):raise ValueError('Invalid cache header: '+str(path))
    header=json.loads(data[14:14+n]);payload=data[14+n:]
    if header['key']!=key:return None
    if hashlib.sha256(payload).hexdigest()!=header['sha256']:raise ValueError('Corrupt cache payload: '+str(path))
    return pickle.loads(payload)


def terrain_key(path):
    from nav.terrain import WALK_RES,GOALS_FROM_SPAWNS
    from generate_terrain_map import resolve_mission
    from nav.terrain import TerrainMap
    with np.load(path,allow_pickle=False) as z:name=str(z['map_name'])
    files=[path,ML/'nav/terrain.py',ML/'terrain_obs.py',ML/'nav/physics.py']
    if GOALS_FROM_SPAWNS:files.append(resolve_mission(name))
    return fingerprint(files,walk_res=WALK_RES,goals_from_spawns=GOALS_FROM_SPAWNS)


def cached_terrain(path, folder):
    from nav.terrain import TerrainGrid
    key=terrain_key(path);dest=Path(folder)/('terrain_'+key+'.bin');started=time.perf_counter()
    with file_lock(dest.with_suffix('.lock')):
        terrain=read_cache(dest,key);hit=terrain is not None
        if not hit:
            terrain=TerrainGrid(path);write_cache(dest,key,terrain)
    terrain.static_cache_key=key
    terrain.static_cache_report=dict(hit=hit,elapsed_s=time.perf_counter()-started,path=str(dest))
    return terrain


def mesh_key(mesh, dll):
    return fingerprint([mesh.path,Path(mesh.path).with_suffix('.json'),dll,
        ML/'nav/collision/mesh.py',ML/'nav/collision/experimental_sweep.py',
        ML/'nav/collision/sweep_kernel.cpp',ML/'nav/marble_dynamics/query_cache.py'],
        cell=float(mesh.cache_cell),entries=int(mesh.cache_entries))


def load_queries(mesh, dll, folder):
    key=mesh_key(mesh,dll);mesh.persistent_query_file=Path(folder)/('queries_'+key+'.bin')
    mesh.persistent_query_key=key
    with file_lock(mesh.persistent_query_file.with_suffix('.lock')):
        saved=read_cache(mesh.persistent_query_file,key)
    if saved is not None:
        for ids in saved.values():
            if np.asarray(ids).dtype!=np.int64 or np.any(ids<0) or np.any(ids>=len(mesh.triangles)):
                raise ValueError('Invalid saved triangle index')
        mesh.query_regions=saved
    return saved is not None


def save_queries(mesh):
    if not hasattr(mesh,'persistent_query_file'):return
    path=mesh.persistent_query_file;key=mesh.persistent_query_key
    with file_lock(path.with_suffix('.lock')):
        saved=read_cache(path,key)
        if saved is None:saved=copy.copy(mesh.query_regions)
        else:
            saved.update(mesh.query_regions)
            while len(saved)>mesh.cache_entries:saved.popitem(last=False)
        write_cache(path,key,saved)


class GoalFieldMemo:
    """Exact memo of TerrainGrid.goal_field for one static walking terrain (2026-10-16).

    The field depends only on the snapped source node and the graph, so goals that snap to the same node share one
    read-only array (an in-place write would raise instead of corrupting the cache). The resource search switches
    the route between the gem and every powerup item: v41 Bowl recomputed 153 fields (30 ms each) in one decision.
    """
    def __init__(self, terrain, entries=16):
        self.terrain=terrain;self.compute=terrain.goal_field;self.entries=int(entries)
        from collections import OrderedDict
        self.fields=OrderedDict();self.hits=0;self.misses=0

    def __call__(self, x, y, jumps=True, z=None):
        t=self.terrain
        if t._graph is None:return self.compute(x,y,jumps=jumps,z=z)
        j,i=t.cell_of(x,y);src=t._nearest_walkable(j,i,radius=3,z=z)
        graph=t._graph if (jumps or t._graph_nojump is None) else t._graph_nojump
        key=(src,id(graph));field=self.fields.get(key)
        if field is None:
            field=self.compute(x,y,jumps=jumps,z=z);field.setflags(write=False)
            self.fields[key]=field;self.misses+=1
            while len(self.fields)>self.entries:self.fields.popitem(last=False)
        else:
            self.fields.move_to_end(key);self.hits+=1
        return field


def cached_walking(terrain, mesh, dll, folder, memo=True):
    from nav.marble_dynamics.geometry_route import GroundClearance,walking_terrain
    from nav.marble_dynamics.ground_cache import MemoGroundClearance
    import nav.marble_dynamics.geometry_route as _gr
    key=fingerprint([ML/'nav/marble_dynamics/geometry_route.py',ML/'nav/marble_dynamics/ground_cache.py'],
        terrain=terrain.static_cache_key,mesh=mesh_key(mesh,dll),radius=.189750001,
        jump_step=(float(_gr.JUMP_STEP_U),float(_gr.JUMP_CLEAR_U),float(_gr.JUMP_EDGE_COST_U)))   # 2026-10-24: jump edges
    path=Path(folder)/('clearance_'+key+'.bin');started=time.perf_counter()
    with file_lock(path.with_suffix('.lock')):
        saved=read_cache(path,key);hit=saved is not None
        if hit:
            result=copy.copy(terrain);result._graph=result._graph_nojump=saved['graph']
            result.component=saved['component'];result.geometry_route_report=copy.deepcopy(saved['report'])
            result.ground_clearance=GroundClearance(mesh)
            result.ground_clearance.point_cache=saved['point_cache']
        else:
            result=walking_terrain(terrain,mesh)
            write_cache(path,key,dict(graph=result._graph,component=result.component,
                report=result.geometry_route_report,point_cache=result.ground_clearance.point_cache))
    if memo:result.ground_clearance=MemoGroundClearance(result.ground_clearance)
    query_path=Path(folder)/('segments_'+key+'.bin')
    if memo:
        with file_lock(query_path.with_suffix('.lock')):segments=read_cache(query_path,key)
        if segments is not None:result.ground_clearance.regions=segments
        result.ground_clearance.persistent_file=query_path;result.ground_clearance.persistent_key=key
    result.goal_field=GoalFieldMemo(result)
    result.geometry_route_report.update(persistent_graph_cache_hit=hit,
        current_setup_s=time.perf_counter()-started,original_construction_s=result.geometry_route_report['construction_s'])
    return result


def save_clearance(terrain):
    check=terrain.ground_clearance
    if not hasattr(check,'persistent_file'):return
    path=check.persistent_file;key=check.persistent_key
    with file_lock(path.with_suffix('.lock')):
        saved=read_cache(path,key)
        if saved is None:saved=copy.copy(check.regions)
        else:
            saved.update(check.regions)
            while len(saved)>check.entries:saved.popitem(last=False)
        write_cache(path,key,saved)


def cached_encoding(mesh, model, checkpoint, folder):
    import torch
    from nav.marble_dynamics.student_data import geometry
    key=fingerprint([mesh.path,Path(mesh.path).with_suffix('.json'),checkpoint,
        ML/'nav/marble_dynamics/student_data.py',ML/'nav/marble_dynamics/fast_nav.py'])
    path=Path(folder)/('encoding_'+key+'.bin')
    with file_lock(path.with_suffix('.lock')):
        saved=read_cache(path,key);hit=saved is not None
        if not hit:
            geo=geometry(mesh)
            with torch.inference_mode():
                encoded=model.encode_map(*(torch.from_numpy(geo[k]) for k in ('features','bbox_lo','bbox_hi')))
            saved=dict(geometry=geo,encoded=tuple(v.numpy().copy() for v in encoded));write_cache(path,key,saved)
    return saved['geometry'],tuple(torch.from_numpy(v) for v in saved['encoded']),hit


def marble_processes():
    """Count all native game executables, including games we do not own."""
    import ctypes
    from ctypes import wintypes
    if os.name!='nt':raise RuntimeError('Native game budgeting requires Windows')
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),
            ('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),
            ('parent',wintypes.DWORD),('priority',ctypes.c_long),('flags',wintypes.DWORD),
            ('exe',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    result=[];entry=Entry();entry.size=ctypes.sizeof(entry)
    try:
        present=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while present:
            if str(entry.exe).lower().startswith('marbleblast'):
                result.append(dict(pid=int(entry.pid),name=str(entry.exe)))
            present=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:kernel.CloseHandle(snapshot)
    return result


def guarded_launch(arguments, process_cap, **kwargs):
    import subprocess
    from nav.marble_dynamics.record import RUNTIME
    if not 1<=process_cap<=8:raise ValueError('Native game cap must be1–8')
    started=time.perf_counter()
    while time.perf_counter()-started<1800.:
        with file_lock(RUNTIME/'native_launch.lock'):
            if len(marble_processes())<process_cap:
                return subprocess.Popen(arguments,**kwargs)
        time.sleep(.5)
    raise TimeoutError('All native slots remain occupied; no other games terminated')


class CollectionProfile:
    """Time unchanged functions; report disjoint route/sweep/forecast buckets."""
    def __init__(self):
        self.sweep_s=0.;self.route_exclusive_s=0.;self.forecast_exclusive_s=0.
        self.route_depth=0;self.forecast_calls=0;self.restores=[];self.started=None

    def __enter__(self):
        from nav.routing import RoutePlanner
        from nav.collision.experimental_sweep import CompiledSweepMap
        from nav.marble_dynamics import bridge
        self.started=time.perf_counter()
        def install(owner,name,kind):
            original=getattr(owner,name);self.restores.append((owner,name,original))
            def measured(*args,**kwargs):
                start=time.perf_counter();sweep=self.sweep_s;route=self.route_exclusive_s
                outer=self.route_depth==0
                if kind=='route':self.route_depth+=1
                try:return original(*args,**kwargs)
                finally:
                    elapsed=time.perf_counter()-start
                    if kind=='sweep':self.sweep_s+=elapsed
                    elif kind=='route':
                        self.route_depth-=1
                        if outer:self.route_exclusive_s+=max(0.,elapsed-(self.sweep_s-sweep))
                    else:
                        self.forecast_calls+=1
                        self.forecast_exclusive_s+=max(0.,elapsed-(self.sweep_s-sweep)-(self.route_exclusive_s-route))
            setattr(owner,name,measured)
        install(CompiledSweepMap,'sweep_sphere','sweep')
        install(RoutePlanner,'navigate','route');install(RoutePlanner,'distance','route')
        install(bridge,'predict_replies','forecast')
        return self

    def __exit__(self,*exc):
        for owner,name,original in reversed(self.restores):setattr(owner,name,original)
        self.elapsed_s=time.perf_counter()-self.started

    def report(self):
        elapsed=getattr(self,'elapsed_s',time.perf_counter()-self.started)
        values=dict(sweep_s=self.sweep_s,route_exclusive_s=self.route_exclusive_s,
            forecast_exclusive_s=self.forecast_exclusive_s,elapsed_s=elapsed,
            forecast_calls=self.forecast_calls)
        values['other_s']=max(0.,elapsed-sum(values[k] for k in ('sweep_s','route_exclusive_s','forecast_exclusive_s')))
        values['wall_shares']={k:values[k]/max(elapsed,1e-9) for k in ('sweep_s','route_exclusive_s','forecast_exclusive_s','other_s')}
        values['contract']='Exclusive buckets: sweep includes exact broad/narrow phase; route excludes sweep; forecast excludes route and sweep. Cold map setup reported separately.'
        return values
