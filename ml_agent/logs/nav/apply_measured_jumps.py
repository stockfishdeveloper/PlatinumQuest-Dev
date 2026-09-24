"""HANDOFF 28.27: measured jump envelope (nav/physics.py) wired into the terrain graph, the observation
and the model. Asserts on the pre-change text; run once."""
import os, ast
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def patch(rel, reps):
    p = os.path.join(ROOT, rel); s = open(p, encoding='utf-8').read()
    for a, b in reps:
        assert a in s, (rel, a[:70])
        s = s.replace(a, b, 1)
    ast.parse(s); open(p, 'w', encoding='utf-8').write(s); print('patched', rel)

# ---------------------------------------------------------------- physics: median apex/flight
patch('nav/physics.py', [
    ("            apex = float(np.mean([r['apex'] for r in rows])); fl = float(np.mean([r['flight_s'] for r in rows]))",
     "            apex = float(np.median([r['apex'] for r in rows])); fl = float(np.median([r['flight_s'] for r in rows]))"),
])

# ---------------------------------------------------------------- terrain graph: 16-heading measured jump edges
patch('nav/terrain.py', [
    ("JUMP_GAP = 4.0        # gaps up to this wide (u) get a \"jump edge\" in the walk graph\n",
     "JUMP_GAP = 4.0        # (superseded 2026-09-23, HANDOFF 28.27) the old 8-direction rule admitted gaps up to\n"
     "                      # this wide. Jump edges now come from nav/physics.py: any of 16 headings, gap up to\n"
     "                      # physics.MAX_JUMP_GAP (measured range at CRUISE_SPEED minus the landing margin).\n"
     "JUMP_HEADINGS = 16    # headings searched from every edge cell (the observation's 16 ray headings)\n"),
    ("JUMP_COST = 1.5       # 3.0 -> 1.5 on 2026-09-22 (HANDOFF 28.13).",
     "JUMP_COST = 1.2       # 1.5 -> 1.2 on 2026-09-23 (HANDOFF 28.27): a jump edge is priced as its measured flight\n"
     "                      # (0.76 s, about the rolling time for the same distance) plus a landing loss. At 1.5 the\n"
     "                      # hypotenuse of a 7+7 u corner (10 u x 1.5 = 15) still lost to walking the legs (14).\n"
     "                      # (superseded) 3.0 -> 1.5 on 2026-09-22 (HANDOFF 28.13)."),
    # replace the old jump-edge loop
    ("        max_cells = int(round(JUMP_GAP / self.walk_res))\n"
     "        jr, jc, jw = [], [], []\n"
     "        for (j, i) in np.argwhere(self.edge):\n"
     "            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):\n"
     "                for k in range(2, max_cells + 1):\n"
     "                    jj, ii = j + dy * k, i + dx * k\n"
     "                    if not self.in_walk_grid(jj, ii):\n"
     "                        break\n"
     "                    if self.walkable[jj, ii]:\n"
     "                        if k >= 2 and not any(self.walkable[j + dy * m, i + dx * m] for m in range(1, k)):\n"
     "                            dz = self.walk_top[jj, ii] - self.walk_top[j, i]\n"
     "                            if -JUMP_DROP <= dz <= JUMP_RISE:\n"
     "                                d = self.walk_res * k * math.hypot(dx, dy)\n"
     "                                jr.append(idx[j, i]); jc.append(idx[jj, ii]); jw.append(d * JUMP_COST)\n"
     "                        break\n"
     "        self.jump_edges = len(jr)\n",
     "        # MEASURED jump edges (2026-09-23, HANDOFF 28.27; the 8-direction / 4 u rule is above in the\n"
     "        # comment history). From every edge cell, march the FINE height map in JUMP_HEADINGS headings:\n"
     "        # the void must begin within 1.5 u (it is this cell's own lip), and the first floor after it\n"
     "        # within physics.MAX_JUMP_GAP u with a height change in [-JUMP_DROP, +JUMP_RISE] is a landing.\n"
     "        # MAX_JUMP_GAP is the measured range at CRUISE_SPEED minus LANDING_MARGIN (~7 u on KOTM's\n"
     "        # marble), so a 7 x 7 hole is admitted straight across and every corner cut is admitted too.\n"
     "        jr, jc, jw = [], [], []\n"
     "        self.jump_edge_cells = []\n"
     "        for (j, i, jj, ii, gap) in self._measured_jump_edges():\n"
     "            jr.append(idx[j, i]); jc.append(idx[jj, ii]); jw.append(gap * JUMP_COST)\n"
     "            self.jump_edge_cells.append((j, i, jj, ii, gap))\n"
     "        self.jump_edges = len(jr)\n"),
    ("    def _build_graph(self):\n",
     "    def _measured_jump_edges(self):\n"
     "        \"\"\"[(j, i, jj, ii, gap_u), ...]: takeoff walk cell, landing walk cell, gap length.\"\"\"\n"
     "        from nav.physics import MAX_JUMP_GAP\n"
     "        present = np.isfinite(self.heights).any(0)\n"
     "        H, W = present.shape\n"
     "        heads = [(math.cos(2 * math.pi * k / JUMP_HEADINGS), math.sin(2 * math.pi * k / JUMP_HEADINGS)) for k in range(JUMP_HEADINGS)]\n"
     "        step = self.res\n"
     "        n_steps = int(math.ceil((MAX_JUMP_GAP + 1.5) / step))\n"
     "        out = []\n"
     "        for (j, i) in np.argwhere(self.edge):\n"
     "            x = float(self.wxs[i]); y = float(self.wys[j]); z0 = float(self.walk_top[j, i])\n"
     "            for cx, cy in heads:\n"
     "                in_void = False; d_void = 0.0\n"
     "                for s in range(1, n_steps + 1):\n"
     "                    d = s * step\n"
     "                    ii = int(round((x + cx * d - self.x0) / step)); jj = int(round((y + cy * d - self.y0) / step))\n"
     "                    if not (0 <= ii < W and 0 <= jj < H):\n"
     "                        break\n"
     "                    if not present[jj, ii]:\n"
     "                        if not in_void:\n"
     "                            in_void = True; d_void = d\n"
     "                        continue\n"
     "                    if not in_void:\n"
     "                        if d > 1.5:\n"
     "                            break                  # floor continues: this heading has no lip here\n"
     "                        continue\n"
     "                    gap = d - d_void + step\n"
     "                    if gap > MAX_JUMP_GAP:\n"
     "                        break\n"
     "                    zs = self.heights[:, jj, ii]; zs = zs[np.isfinite(zs)]\n"
     "                    if len(zs) and np.any((zs - z0 >= -JUMP_DROP) & (zs - z0 <= JUMP_RISE)):\n"
     "                        lj, li = self.cell_of(x + cx * (d + 0.5), y + cy * (d + 0.5))\n"
     "                        if self.in_walk_grid(lj, li) and self.walkable[lj, li] and (lj, li) != (j, i):\n"
     "                            out.append((j, i, lj, li, float(gap)))\n"
     "                    break\n"
     "        return out\n\n"
     "    def _build_graph(self):\n"),
])

# ---------------------------------------------------------------- terrain_obs: gap along a heading
patch('terrain_obs.py', [
    ("    def edge_rays(self, x, y, z, gem_rels):\n",
     "    def gap_along(self, x, y, z, ux, uy, max_u=RAY_RANGE):\n"
     "        \"\"\"(lip_u, gap_u, landing_dz): along unit heading (ux, uy) from (x, y): distance to the lip\n"
     "        (inf if the floor continues for max_u), the gap length from the lip to the first floor beyond\n"
     "        it (inf if none within max_u) and that floor's height relative to the current floor.\n"
     "        Added 2026-09-23 (HANDOFF 28.27) for the velocity-heading gap features.\"\"\"\n"
     "        z0 = self.floor_height(x, y, z)\n"
     "        n = int(round(max_u / RAY_STEP))\n"
     "        ds = (np.arange(1, n + 1) * RAY_STEP).astype(np.float32)\n"
     "        h = self.heights_at(x + ux * ds, y + uy * ds)                 # (K, n)\n"
     "        # follow the level nearest the current height; the lip is the first step with no level\n"
     "        # within FOLLOW_TOL of the one before it\n"
     "        z_cur = float(z0); lip = math.inf; lip_i = -1\n"
     "        for s in range(n):\n"
     "            col = h[:, s]; fin = np.isfinite(col)\n"
     "            if fin.any():\n"
     "                k = int(np.nanargmin(np.abs(np.where(fin, col, np.inf) - z_cur)))\n"
     "                if abs(float(col[k]) - z_cur) <= FOLLOW_TOL:\n"
     "                    z_cur = float(col[k]); continue\n"
     "            lip = float(ds[s]); lip_i = s\n"
     "            break\n"
     "        if lip_i < 0:\n"
     "            return math.inf, math.inf, 0.0\n"
     "        for s in range(lip_i, n):\n"
     "            col = h[:, s]; fin = np.isfinite(col)\n"
     "            if fin.any():\n"
     "                zs = col[fin]; k = int(np.argmin(np.abs(zs - z_cur)))\n"
     "                return lip, float(ds[s] - lip + RAY_STEP), float(zs[k] - z_cur)\n"
     "        return lip, math.inf, 0.0\n\n"
     "    def edge_rays(self, x, y, z, gem_rels):\n"),
])

# ---------------------------------------------------------------- obs: 3 gap features along the velocity
patch('nav/obs.py', [
    ("NAV_OBS_VERSION = 'NAV_OBS_V2'   # V2 (2026-09-19) appends the NEXT gem: see NEXT_DIM below\n",
     "NAV_OBS_VERSION = 'NAV_OBS_V3'   # V3 (2026-09-23, HANDOFF 28.27) appends GAP_DIM: the gap along the velocity\n"
     "                                 # V2 (2026-09-19) appends the NEXT gem: see NEXT_DIM below\n"),
    ("VEC_DIM = VEC_NEXT + NEXT_DIM # 53\n",
     "VEC_GAP = VEC_NEXT + NEXT_DIM # 53: where the gap block starts\n"
     "GAP_DIM = 3                   # along the velocity heading (or the waypoint bearing when slow):\n"
     "                              # lip distance / RAY_RANGE (1 = none), gap length / RAY_RANGE (1 = no\n"
     "                              # landing), crossable at the CURRENT speed per nav/physics (0/1)\n"
     "VEC_DIM = VEC_GAP + GAP_DIM   # 56\n"
     "GAP_MIN_SPEED = 1.0           # u/s: below this the gap features use the waypoint bearing\n"),
    ("from nav.terrain import CROP_SHAPE\n",
     "from nav.terrain import CROP_SHAPE, JUMP_DROP, JUMP_RISE\n"
     "from nav.physics import crossable\n"
     "from terrain_obs import RAY_RANGE\n"),
])
# the gap block fill: after the next-gem block is written. Find the end of build() by its return.
p = os.path.join(ROOT, 'nav', 'obs.py'); s = open(p, encoding='utf-8').read()
a = "        return crop, vec, on_floor\n"
assert s.count(a) == 1
b = ("        # GAP block (V3): what lies along the marble's own heading\n"
     "        sp = math.hypot(vx, vy)\n"
     "        hx, hy = (vx / sp, vy / sp) if sp > GAP_MIN_SPEED else (ux, uy)\n"
     "        if hx != 0.0 or hy != 0.0:\n"
     "            lip, gap, ldz = self.terrain.gap_along(x, y, z, hx, hy)\n"
     "            vec[VEC_GAP] = min(lip / RAY_RANGE, 1.0) if math.isfinite(lip) else 1.0\n"
     "            vec[VEC_GAP + 1] = min(gap / RAY_RANGE, 1.0) if math.isfinite(gap) else 1.0\n"
     "            ok = math.isfinite(gap) and (-JUMP_DROP <= ldz <= JUMP_RISE) and bool(crossable(gap, sp))\n"
     "            vec[VEC_GAP + 2] = 1.0 if ok else 0.0\n"
     "        return crop, vec, on_floor\n")
s = s.replace(a, b, 1); ast.parse(s); open(p, 'w', encoding='utf-8').write(s); print('patched nav/obs.py (gap block)')

# ---------------------------------------------------------------- model: the jump prior uses the measured crossable flag
patch('nav/model.py', [
    ("from nav.obs import VEC_DIM\n", "from nav.obs import VEC_DIM, VEC_GAP\n"),
    ("        present = crop[:, 1][bi, iy, ix] > 0.5\n"
     "        level = crop[:, 0][bi, iy, ix].abs() < JUMP_LANDING_DZ\n"
     "        landing = (present & level).any(dim=1)\n"
     "        return (at_edge & ready & landing).float()\n",
     "        present = crop[:, 1][bi, iy, ix] > 0.5\n"
     "        level = crop[:, 0][bi, iy, ix].abs() < JUMP_LANDING_DZ\n"
     "        landing_short = (present & level).any(dim=1)          # the old short-hop test (<= JUMP_LANDING_MAX u)\n"
     "        # 2026-09-23 (HANDOFF 28.27): OR the measured test: the gap along the marble's own heading is\n"
     "        # within jump_range(current speed) per nav/physics (obs.py fills vec[VEC_GAP + 2]). This is what\n"
     "        # lets the prior fire for a 7 u hole at 8 u/s and NOT at 5 u/s.\n"
     "        landing = landing_short | (vec[:, VEC_GAP + 2] > 0.5)\n"
     "        return (at_edge & ready & landing).float()\n"),
])
print('all patches applied')
