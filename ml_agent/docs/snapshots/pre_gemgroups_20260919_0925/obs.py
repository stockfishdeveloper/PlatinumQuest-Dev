"""Navigator observation: two height crops + a short vector. Map-independent, world frame.

Layout NAV_OBS_V1 (stored in every checkpoint; change => bump the version):
    crop  (6, 32, 32)  fine (0.5 u) then coarse (2 u): [rel height, present, 2nd level] each
    vec   (48,)
        0-3   waypoint: unit dx, dy; distance/50 clipped to 1; dz/10 clipped to +/-1
        4-9   self: vx/20, vy/20, vz/20, speed/20, on_floor (0/1), airborne decisions/8 clipped to 1
        10-47 edge rays (38): 16 headings [edge_dist, edge_dz] + 3 "gem" rays, of which ray 1 is the
              line to the waypoint (clear fraction, what lies beyond) and rays 2-3 are unused (0)
"""
import math
import numpy as np

from terrain_obs import EDGE_DIM
from nav.protocol import RAW_POS, RAW_VEL
from nav.terrain import CROP_SHAPE

NAV_OBS_VERSION = 'NAV_OBS_V1'
VEC_DIM = 10 + EDGE_DIM       # 48
WAYPOINT_DIST_SCALE = 50.0
VEL_SCALE = 20.0
ON_FLOOR_VZ = 0.3
ON_FLOOR_DZ = 0.6


class ObsBuilder:
    """Per-environment state (airborne counter) + the build() function."""

    def __init__(self, terrain):
        self.terrain = terrain
        self.airborne = 0

    def reset(self):
        self.airborne = 0

    def build(self, raw, goal):
        x, y, z = (float(v) for v in raw[RAW_POS])
        vx, vy, vz = (float(v) for v in raw[RAW_VEL])
        gx, gy, gz = goal
        crop = self.terrain.crop(x, y, z)
        dx, dy = gx - x, gy - y
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)
        floor = self.terrain.floor_z(x, y, z)
        on_floor = abs(vz) < ON_FLOOR_VZ and abs(z - floor) < ON_FLOOR_DZ
        self.airborne = 0 if on_floor else self.airborne + 1
        vec = np.zeros(VEC_DIM, dtype=np.float32)
        vec[0] = ux; vec[1] = uy
        vec[2] = min(d / WAYPOINT_DIST_SCALE, 1.0)
        vec[3] = max(-1.0, min(1.0, (gz - z) / 10.0))
        vec[4] = vx / VEL_SCALE; vec[5] = vy / VEL_SCALE; vec[6] = vz / VEL_SCALE
        vec[7] = math.sqrt(vx * vx + vy * vy + vz * vz) / VEL_SCALE
        vec[8] = 1.0 if on_floor else 0.0
        vec[9] = min(self.airborne / 8.0, 1.0)
        vec[10:] = self.terrain.edge_rays(x, y, z, [(dx, dy, gz - z, True)])
        return crop.astype(np.float32), vec, on_floor


assert CROP_SHAPE == (6, 32, 32)
