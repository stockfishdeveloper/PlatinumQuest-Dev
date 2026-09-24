"""Reconstruct the jump prior's components from a real_trace.csv (positions, velocities, goal) offline and
report how often each condition held. Usage: python logs/nav/gp_from_trace.py <trace.csv> [<trace2.csv> ...]"""
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from terrain_obs import TerrainMap
from nav.terrain import TerrainGrid
from nav.obs import ObsBuilder, VEC_GAP
from nav.model import (NavActorCritic, VEC_GOAL_RAY_CLEAR, VEC_GOAL_RAY_BEYOND, VEC_GOAL_DIST, VEC_SPEED, VEC_ON_FLOOR,
                       JUMP_PRIOR_DIST, JUMP_PRIOR_DROP, JUMP_PRIOR_SPEED)
t = TerrainGrid(TerrainMap.resolve('KingOfTheMarble_Hunt'))
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    ob = ObsBuilder(t)
    n = 0; at_edge = 0; ready = 0; cross = 0; both = 0; gp_on = 0; lip_seen = 0; jumps_on_gp = 0
    speeds_at_edge = []
    for r in rows:
        x, y, z = float(r['x']), float(r['y']), float(r['z']); vx, vy = float(r['vx']), float(r['vy'])
        raw = np.zeros(35, dtype=np.float32); raw[0:3] = (x, y, z); raw[3:6] = (vx, vy, 0.0)
        goal = (float(r['gx']), float(r['gy']), z)
        crop, vec, _ = ob.build(raw, goal)
        vec[VEC_ON_FLOOR] = float(r['on_floor'])
        edge_u = vec[VEC_GOAL_RAY_CLEAR] * vec[VEC_GOAL_DIST] * 50.0
        ae = (edge_u < JUMP_PRIOR_DIST) and (vec[VEC_GOAL_RAY_CLEAR] < 0.999) and (vec[VEC_GOAL_RAY_BEYOND] < JUMP_PRIOR_DROP)
        rd = (vec[VEC_ON_FLOOR] > 0.5) and (vec[VEC_SPEED] * 20.0 > JUMP_PRIOR_SPEED)
        cr = vec[VEC_GAP + 2] > 0.5
        gp = float(NavActorCritic.gap_prior(torch.from_numpy(crop)[None], torch.from_numpy(vec)[None])[0])
        n += 1; at_edge += ae; ready += rd; cross += cr; both += (ae and rd); gp_on += gp > 0.5
        if vec[VEC_GAP] < 0.999: lip_seen += 1
        if ae: speeds_at_edge.append(float(r['speed']))
        if gp > 0.5 and int(float(r['jump'])): jumps_on_gp += 1
    onf = sum(float(r['on_floor']) > 0.5 for r in rows) / n
    print(f'{os.path.basename(path)}: rows {n}, on_floor {100*onf:.1f}%, goal-ray at_edge {at_edge} ({100*at_edge/n:.2f}%), ready {100*ready/n:.1f}%, '
          f'at_edge&ready {both}, crossable-flag on {cross} ({100*cross/n:.2f}%), gap prior ON {gp_on} ({100*gp_on/n:.2f}%), jump cmd while gp on {jumps_on_gp}, '
          f'median speed at edge {np.median(speeds_at_edge) if speeds_at_edge else 0:.1f}')
