"""Conservative walking proposals checked against full-map walls and ceilings.

This experimental layer only supplies route aims/costs. Jump/air/powerup futures
are still evaluated by the complete mechanics model; a floor graph cannot
certify a maneuver. Original terrain arrays and serialized geometry are retained.
"""
import copy
import time

import numpy as np

from nav.collision.mesh import CollisionMap
from nav.routing import RoutePlanner
from nav.collision.experimental_sweep import CompiledSweepMap,CachedCompiledSweepMap


class ObstacleGeometry(CollisionMap):
    def __init__(self,mesh):
        self.__dict__=mesh.__dict__.copy()
        self.obstacles=self.normals[:,2]<=.4

    def query_aabb(self,lo,hi):
        ids=super().query_aabb(lo,hi)
        return ids[self.obstacles[ids]]


class CompiledObstacleGeometry(ObstacleGeometry,CompiledSweepMap):pass
class CachedCompiledObstacleGeometry(ObstacleGeometry,CachedCompiledSweepMap):pass


class GroundClearance:
    def __init__(self,mesh,radius=.189750001):
        kind=ObstacleGeometry
        if hasattr(mesh,'kernel'):
            kind=CachedCompiledObstacleGeometry if hasattr(mesh,'query_regions') else CompiledObstacleGeometry
        self.mesh=kind(mesh);self.radius=radius
        self.checked=0;self.blocked=0;self.point_cache={}

    def point_clear(self,point):
        key=tuple(point)
        if key not in self.point_cache:
            self.point_cache[key]=not len(self.mesh.sphere_contacts(point,self.radius-1e-6)['triangle_ids'])
        return self.point_cache[key]

    def segment(self,start,end):
        self.checked+=1
        # Starts at exact contact need a small geometric tolerance: the sweep
        # requires a clear start. This 1e-6u clearance never changes physics.
        radius=self.radius-1e-6
        if not self.point_clear(start) or not self.point_clear(end):
            self.blocked+=1;return False
        hit=self.mesh.sweep_sphere(start,end,radius)
        if hit is None:hit=self.mesh.sweep_sphere(end,start,radius)
        clear=hit is None
        self.blocked+=int(not clear)
        return clear

    def __call__(self,floor_points):
        centers=[np.asarray(p)+[0,0,self.radius+.001] for p in floor_points]
        return all(self.segment(a,b) for a,b in zip(centers,centers[1:]))


JUMP_STEP_U=0.       # (OFF: sim 10-24 1.2 u = episodes +2.1 percent slower, 0.8 u = -1.1 percent, noise; the TreasureBox trip route it was
                     # meant to shorten did not change) 2026-10-24 (operator: trips between gem groups): a step up to this high, pruned from the walking graph because a
                     # straight roll hits its face, is kept as a JUMP edge when a jump arc clears the geometry (rise at the lower
                     # cell to JUMP_CLEAR_U over the upper floor, cross above the lip, come down on the upper cell). Without it every
                     # pad, box and raised floor needed a ramp: planned routes ran 1.5x the straight line (Concentric 2.3x) and
                     # plans that crossed a step scored as if they had made no progress. 0 = off (the walking graph of before).
JUMP_CLEAR_U=.35     # ... the arc's height over the upper floor (the marble's jump apex is about 1.4 u)
JUMP_EDGE_COST_U=1.  # ... extra route length charged per jump edge (the hop's time)


def walking_terrain(terrain,mesh):
    """Copy graph metadata and prune crossing edges; never edit the source grid.

    Use walking adjacency as guidance rather than the old approximate jump
    edges, which omit vertical collision. Full dynamics proposes jumps directly.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    started=time.perf_counter();copy_grid=copy.copy(terrain)
    check=GroundClearance(mesh);graph=terrain._graph_nojump.tocoo()
    nodes=terrain._nodes
    points=np.column_stack((terrain.wxs[nodes[:,2]],terrain.wys[nodes[:,1]],
        terrain.walk_z[nodes[:,0],nodes[:,1],nodes[:,2]]))+[0,0,check.radius+.001]
    cache={};keep=[]
    for a,b in zip(graph.row,graph.col):
        key=(min(int(a),int(b)),max(int(a),int(b)))
        if key not in cache:cache[key]=check.segment(points[a],points[b])
        keep.append(cache[key])
    keep=np.asarray(keep,dtype=bool)
    data=np.asarray(graph.data,dtype=float).copy();jump_edges=0
    if JUMP_STEP_U:
        r=check.radius+.001
        for n,(a,b) in enumerate(zip(graph.row,graph.col)):
            if keep[n]:continue
            dz=float(points[b][2]-points[a][2])
            if not (.05<dz<=JUMP_STEP_U):continue
            top=float(points[b][2])+JUMP_CLEAR_U
            a_up=np.array([points[a][0],points[a][1],top]);b_up=np.array([points[b][0],points[b][1],top])
            if check.segment(points[a],a_up) and check.segment(a_up,b_up) and check.segment(b_up,points[b]):
                keep[n]=True;data[n]+=JUMP_EDGE_COST_U;jump_edges+=1
    filtered=coo_matrix((data[keep],(graph.row[keep],graph.col[keep])),shape=graph.shape).tocsr()
    copy_grid._graph=filtered;copy_grid._graph_nojump=filtered
    _,labels=connected_components(filtered,directed=True,connection='weak')
    copy_grid.component=np.full_like(terrain.component,-1)
    copy_grid.component[nodes[:,0],nodes[:,1],nodes[:,2]]=labels
    copy_grid.ground_clearance=check
    copy_grid.geometry_route_report=dict(source_geometry_sha256=mesh.info['mesh_sha256'],
        compiled_walking_sweep=hasattr(check.mesh,'kernel'),
        original_walk_edges=int(graph.nnz),retained_walk_edges=int(filtered.nnz),
        removed_walk_edges=int(graph.nnz-filtered.nnz),unique_pairs_checked=len(cache),
        original_approximate_jump_edges=int(terrain.jump_edges),jump_step_edges=int(jump_edges),jump_step_u=float(JUMP_STEP_U),
        construction_s=time.perf_counter()-started,
        contract='Conservative wall/ceiling-filtered walking guidance; complete dynamics still proposes jumps/air/powerups.')
    return copy_grid
