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
CLIMB_EDGES=True     # 2026-10-25 (operator: a real route to gems on raised places): the ROUTE graph (goal fields, waypoints) also gets a
                     # climb edge from a floor node to a neighbouring higher node the walk lacks, when a jump (+ Blast) arc clears the
                     # geometry (straight up from the lower cell to CLIMB_EDGE_CLEAR_U over the upper floor, across, down onto it).
                     # Without them a gem on a block, pad or deck had a goal field covering only its own patch (244 of 1760 spawn
                     # points on the 10 maps) and route distance fell back to the straight line + 5 u: driving under a deck read as
                     # progress, a far raised gem was steered at in a straight line into walls. Walking tests (crossings.reachable,
                     # the climb skill's walk time, the tour's walk distances) keep the walk-only graph (_graph_nojump).
CLIMB_EDGES_BETWEEN=True # ... only between walking regions the marble cannot walk between both ways (strongly connected components of the
                         # walking graph): a raised place reachable by ramps keeps its walking routes (with climb edges inside walked
                         # areas too, Sprawl's 10566 edges sent routes up 2-6 u climbs where walking was fine: sim +13 %, pair route1
                         # Sprawl 157 vs 185)
CLIMB_EDGE_MAX_U=6.5 # ... rises up to this (an open jump + Blast with a full meter, climbs.open_meter)
CLIMB_EDGE_CLEAR_U=.5
CLIMB_EDGE_BACK_U=.25    # the arc may also start this far back from the upper cell, on the floor there (a walk cell's centre can touch
                         # the step's face: Duplex's 0.5 u gem pads were all cut off with the centre start)
CLIMB_HOP_U=1.2      # up to this rise a plain jump: CLIMB_HOP_COST_U of route length; higher: CLIMB_COST_U + CLIMB_COST_PER_U per u of rise
CLIMB_HOP_COST_U=4.  # (JUMP_STEP_U's 1 u made routes hop over pads as shortcuts: sim 10-24, no gain)
CLIMB_COST_U=10.
CLIMB_COST_PER_U=4.


def route_map_off():
    """--no-route-map (the matched control of the 2026-10-25 route map): no climb edges, and route lookups take the cell's nearest
    level again (terrain.ROUTE_LEVEL_TOL_U = 0). Call before the walking terrain is built."""
    global CLIMB_EDGES
    import nav.terrain as _terrain
    CLIMB_EDGES=False;_terrain.ROUTE_LEVEL_TOL_U=0.


def climb_edges(terrain,check,points,walk,mesh=None,regions=None):
    """{(a, b): rise} directed climb edges from node a to an 8-neighbour node b higher by 0.05..CLIMB_EDGE_MAX_U that the walking
    graph `walk` lacks, with the jump (+ Blast) arc clear (see CLIMB_EDGES). points: the nodes' sphere centres; mesh: the full mesh
    (floor raycasts for the backed-off start); regions: per node its walking region (pairs inside one region are skipped)."""
    walk=walk.tocoo();have=set(zip(walk.row.tolist(),walk.col.tolist()))
    idx=terrain._node_idx;nodes=terrain._nodes;KW,H,W=terrain.walkable.shape
    r=check.radius+.001
    def arc(pa,pb):
        top=float(pb[2])+CLIMB_EDGE_CLEAR_U;b_up=np.array([pb[0],pb[1],top])
        starts=[pa]
        if mesh is not None and CLIMB_EDGE_BACK_U>0:
            d=np.asarray(pa[:2],dtype=float)-np.asarray(pb[:2],dtype=float);n=float(np.hypot(d[0],d[1]))
            if n>1e-9:
                q=np.array([pa[0]+d[0]/n*CLIMB_EDGE_BACK_U,pa[1]+d[1]/n*CLIMB_EDGE_BACK_U,pa[2]])
                hit=mesh.raycast(q+[0,0,1.],q-[0,0,2.],cull_backfaces=True)
                if hit is not None and float(hit['normal'][2])>.75:
                    q[2]=float(hit['point'][2])+r
                    starts.append(q)
        for s in starts:
            s_up=np.array([s[0],s[1],top])
            if check.segment(s,s_up) and check.segment(s_up,b_up) and check.segment(b_up,pb):
                return True
        return False
    out={}
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
        bj=nodes[:,1]+dy;bi=nodes[:,2]+dx
        sel=np.where((bj>=0)&(bj<H)&(bi>=0)&(bi<W))[0]
        for kb in range(KW):
            ib=idx[kb,bj[sel],bi[sel]];ok=ib>=0
            a=sel[ok];b=ib[ok]
            dz=points[b,2]-points[a,2]
            m=(dz>.05)&(dz<=CLIMB_EDGE_MAX_U)
            if regions is not None:m&=regions[a]!=regions[b]
            for aa,bb,d in zip(a[m].tolist(),b[m].tolist(),dz[m].tolist()):
                if (aa,bb) in have or (aa,bb) in out:continue
                if arc(points[aa],points[bb]):
                    out[(aa,bb)]=float(d)
    return out


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
    copy_grid.climb_edges={}
    if CLIMB_EDGES:
        # the route graph = the walking graph + climb edges; components and _graph_nojump stay walk-only
        regions=connected_components(filtered,directed=True,connection='strong')[1] if CLIMB_EDGES_BETWEEN else None
        ce=climb_edges(terrain,check,points,filtered,mesh,regions)
        if ce:
            ab=np.array(list(ce.keys()),dtype=np.int64);rise=np.array(list(ce.values()),dtype=float)
            na=nodes[ab[:,0]];nb=nodes[ab[:,1]]
            cost=(terrain.walk_res*np.hypot(na[:,1]-nb[:,1],na[:,2]-nb[:,2])
                  +np.where(rise<=CLIMB_HOP_U,CLIMB_HOP_COST_U,CLIMB_COST_U+CLIMB_COST_PER_U*rise))
            fc=filtered.tocoo()
            copy_grid._graph=coo_matrix((np.concatenate([fc.data,cost]),(np.concatenate([fc.row,ab[:,0]]),np.concatenate([fc.col,ab[:,1]]))),
                                        shape=graph.shape).tocsr()
        copy_grid.climb_edges=ce
    copy_grid.ground_clearance=check
    copy_grid.geometry_route_report=dict(source_geometry_sha256=mesh.info['mesh_sha256'],
        compiled_walking_sweep=hasattr(check.mesh,'kernel'),
        original_walk_edges=int(graph.nnz),retained_walk_edges=int(filtered.nnz),
        removed_walk_edges=int(graph.nnz-filtered.nnz),unique_pairs_checked=len(cache),
        original_approximate_jump_edges=int(terrain.jump_edges),jump_step_edges=int(jump_edges),jump_step_u=float(JUMP_STEP_U),
        climb_edges=len(copy_grid.climb_edges),
        construction_s=time.perf_counter()-started,
        contract='Conservative wall/ceiling-filtered walking guidance; complete dynamics still proposes jumps/air/powerups.')
    return copy_grid
