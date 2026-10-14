"""Isolate strict compiled sweeps in wall/ceiling walking-clearance queries."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from nav.collision.experimental_sweep import CompiledSweepMap,CachedCompiledSweepMap
from nav.collision.mesh import atomic_json
from nav.marble_dynamics import geometry_route
from nav.marble_dynamics.ground_cache import MemoGroundClearance
from nav.marble_dynamics.landing import choose_landing
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.record import ML,sha
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


OriginalObstacle=geometry_route.ObstacleGeometry


class CompiledObstacle(OriginalObstacle,CompiledSweepMap):pass
class CachedCompiledObstacle(OriginalObstacle,CachedCompiledSweepMap):pass


def adaptive_obstacle(mesh):
    if not hasattr(mesh,'kernel'):return OriginalObstacle(mesh)
    return (CachedCompiledObstacle if hasattr(mesh,'query_regions') else CompiledObstacle)(mesh)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def graph_hash(terrain):
    graph=terrain._graph_nojump
    return hashlib.sha256(graph.data.tobytes()+graph.indices.tobytes()+graph.indptr.tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(geometry_route.__file__),a.dll]
    atomic_json(a.output/'prepared.json',dict(source_hashes={str(path):sha(path) for path in sources},
        maps=['Concentric_Hunt','CubeIsle_Hunt','Duplex_Hunt'],training=False,live_integrated=False))
    graphs=[]
    for name in ('Concentric_Hunt','CubeIsle_Hunt','Duplex_Hunt'):
        terrain=TerrainGrid(TerrainMap.resolve(name));mesh=CachedCompiledSweepMap(ML/'collision_maps'/(name+'.npz'),a.dll)
        variants=[]
        for compiled in (False,True):
            geometry_route.ObstacleGeometry=adaptive_obstacle if compiled else OriginalObstacle
            started=time.perf_counter();geometry=geometry_route.walking_terrain(terrain,mesh)
            variants.append(dict(compiled=compiled,graph_sha256=graph_hash(geometry),elapsed_s=time.perf_counter()-started,
                retained_edges=geometry.geometry_route_report['retained_walk_edges']))
        exact=variants[0]['graph_sha256']==variants[1]['graph_sha256']
        graphs.append(dict(map=name,variants=variants,exact=exact));print(json.dumps(graphs[-1]),flush=True)
        atomic_json(a.output/'graphs.json',graphs)
    source=ML/'dynamics_reports/reliability_landing_validation_20261012/Concentric_Hunt/151113/physics_plan/terminal_search.json'
    terminal=json.loads(source.read_text());terrain=TerrainGrid(TerrainMap.resolve('Concentric_Hunt'));cases=[]
    try:
        for compiled in (False,True):
            geometry_route.ObstacleGeometry=adaptive_obstacle if compiled else OriginalObstacle
            mesh=CachedCompiledSweepMap(ML/'collision_maps/Concentric_Hunt.npz',a.dll);engine=GeneralPhysics(mesh)
            geometric=geometry_route.walking_terrain(terrain,mesh);geometric.ground_clearance=MemoGroundClearance(geometric.ground_clearance)
            started=time.perf_counter();chosen,pairs=choose_landing(engine,RoutePlanner(geometric),terminal['dynamics'],terminal['target'],1000,80,40,997,True)
            cases.append(dict(compiled_clearance=compiled,candidate_hashes=[digest([info,rollout]) for info,rollout in pairs],
                elapsed_s=time.perf_counter()-started,selected=chosen[0] if chosen else None))
    finally:geometry_route.ObstacleGeometry=OriginalObstacle
    exact=cases[0]['candidate_hashes']==cases[1]['candidate_hashes'] and cases[0]['selected']==cases[1]['selected']
    result=dict(graphs_exact=all(g['exact'] for g in graphs),landing_search_exact=exact,
        cases=cases,source_sha256=sha(source),training=False,live_integrated=False)
    atomic_json(a.output/'summary.json',result)
    print(json.dumps(dict(graphs_exact=result['graphs_exact'],landing_search_exact=exact,elapsed_s=[c['elapsed_s'] for c in cases])),flush=True)
    assert result['graphs_exact'] and exact


if __name__=='__main__':main()
