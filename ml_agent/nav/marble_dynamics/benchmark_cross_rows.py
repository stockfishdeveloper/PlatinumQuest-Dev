"""Isolated-process benchmark of exact F64 row cross products.

The temporary NumPy substitution is limited to this diagnostic process and
restored afterwards. It is not an application/library integration mechanism.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.geometry_route import walking_terrain
from nav.marble_dynamics.landing import choose_landing
from nav.marble_dynamics.record import sha
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


original_cross=np.cross


def cross_rows(a,b,**kwargs):
    if (kwargs or not isinstance(a,np.ndarray) or not isinstance(b,np.ndarray)
            or a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=3 or a.dtype!=np.float64 or b.dtype!=np.float64):
        return original_cross(a,b,**kwargs)
    out=np.empty_like(a)
    out[:,0]=a[:,1]*b[:,2]-a[:,2]*b[:,1]
    out[:,1]=a[:,2]*b[:,0]-a[:,0]*b[:,2]
    out[:,2]=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    return out


def main():
    source=Path('ml_agent/dynamics_reports/reliability_landing_validation_20261012/Concentric_Hunt/151113/physics_plan/terminal_search.json')
    x=json.loads(source.read_text());engine=GeneralPhysics(CollisionMap(Path('ml_agent/collision_maps/Concentric_Hunt.npz')))
    terrain=walking_terrain(TerrainGrid(TerrainMap.resolve('Concentric_Hunt')),engine.mesh)
    rows=[];fingerprint=None
    try:
        for repeat in range(2):
            for fast in ((False,True) if repeat==0 else (True,False)):
                np.cross=cross_rows if fast else original_cross
                route=RoutePlanner(terrain);started=time.perf_counter()
                chosen,pairs=choose_landing(engine,route,x['dynamics'],x['target'],2000,80,40,997,True)
                elapsed=time.perf_counter()-started
                digest=hashlib.sha256(json.dumps(dict(chosen=chosen[0] if chosen else None,pairs=pairs),sort_keys=True).encode()).hexdigest()
                if fingerprint is None:fingerprint=digest
                assert digest==fingerprint,('search changed',fast)
                row=dict(fast=fast,elapsed_s=elapsed,candidates=len(pairs),exact=True);rows.append(row);print(row,flush=True)
    finally:np.cross=original_cross
    atomic_json(Path('ml_agent/dynamics_reports/cross_rows_benchmark_20261012.json'),dict(rows=rows,source_sha256=sha(source),training=False,integrated=False,
        caveat='Only isolated diagnostic process; integrate explicit local helper if equivalence/performance justify it.'))


if __name__=='__main__':main()
