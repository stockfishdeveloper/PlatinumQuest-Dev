"""Generate the block-cluster training variants and the held-out map from the extracted cluster pool.

    python -m nav.maps.build_cluster_variants [--seed N] [--variants 8] [--per-map 14]

Fixed file names, overwritten on every run (the disk never holds more than these):
    data/multiplayer/hunt/custom/BlockClusters_v0_Hunt.mcs .. _v7_Hunt.mcs   training variants
    data/multiplayer/hunt/custom/BlockClustersHoldout_Hunt.mcs                 the test map
plus terrain_maps/terrain_<name>.npz/.png for each, built here by generate_terrain_map.

Pool: nav/maps/cluster_pool.json (from cluster_extract.py + the usable filter below). The HOLDOUT source map's
clusters never appear in a training variant; training variants draw from the other sources. Each placed cluster
is rotated by a random angle (any degrees) and mirrored at random. Every -autotrain game loads one variant.
"""
import os, sys, json, math, random, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(ML)
CUSTOM = os.path.join(REPO, 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt', 'custom')
POOL = os.path.join(HERE, 'cluster_pool.json')

HOLDOUT_SOURCE = 'Acropolis2_Hunt'
MAX_BOX_H = 1.5      # a box more than a plain jump above the base is a platform edge, not a gem box
MAX_BOX_AREA = 20.0
FLOOR_TOP = 0.38     # smallplatform.dif top at scale z 1
FLOOR_SCALE = 20     # 5 x 5 -> 100 x 100 u (18 = 90 u until 10-08 02:55: the dense 6-column grid reached the edge)
FLOOR_OFF = (-0.12, 0.12)   # smallplatform.dif's footprint is x -2.62..2.38, y -2.38..2.62 at scale 1: its centre is off by this
SPACING = 18.0       # default pitch; --pitch overrides (10-08: 16 u with 30 clusters a map = denser cluster time)
CUBE_OFF = (-0.06, 0.06, 0.12)   # Cube.dif centre at scale 1 (spans x -2.06..1.94, y -1.94..2.06, z -1.88..2.12)
CUBE_SIZE = 4.0
TILE_MAX = 2.0       # 10-08 22:20 (operator: the marble sticks/bumps on the 2 x 4 slab of Incidious c06 on v3): every box is
                     # emitted as UNIFORMLY scaled tiles of at most this side (a 2 x 4 becomes two 2 x 2 cubes) ...
TILE_INFLATE = 0.03  # ... each grown by this per side so touching hulls OVERLAP (no hairline seams or corner-point
                     # contacts between separate collision hulls); gems and the terrain map are unaffected (0.03 u)


def usable_clusters(clusters_json):
    cl = json.load(open(clusters_json))
    out = []
    for name, rows in cl.items():
        for ci, r in enumerate(rows):
            base = r['base']
            boxes = [b for b in r['boxes'] if (b['top'] - base) <= MAX_BOX_H and b['area'] <= MAX_BOX_AREA]
            if not boxes: continue
            ok = True
            for (x, y, z, db), f in zip(r['gems'], r['gem_floor']):
                if f is None: ok = False; break
                if abs(f - base) > 0.1 and not any(abs(f - b['top']) < 0.1 and b['x0'] - 0.3 <= x <= b['x1'] + 0.3
                                                   and b['y0'] - 0.3 <= y <= b['y1'] + 0.3 for b in boxes):
                    ok = False; break
            if not ok: continue
            xs = [g[0] for g in r['gems']] + [b['x0'] for b in boxes] + [b['x1'] for b in boxes]
            ys = [g[1] for g in r['gems']] + [b['y0'] for b in boxes] + [b['y1'] for b in boxes]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            if span > SPACING - 4: continue
            # store relative to the cluster centre and base floor
            gems = [(x - cx, y - cy, z - base, db if db.endswith('_MBU') else db + '_MBU') for x, y, z, db in r['gems']]
            bx = [(b['x0'] - cx, b['x1'] - cx, b['y0'] - cy, b['y1'] - cy, b['top'] - base) for b in boxes]
            out.append({'src': name, 'id': f'{name} c{ci:02d}', 'gems': gems, 'boxes': bx, 'span': span})
    return out


def transform(c, angle_deg, mirror):
    """Rotate a cluster by any angle (degrees, about z) and mirror in x if asked; gems and boxes move together.
    Boxes keep their own width/depth and carry the angle, so a rectangle stays a rectangle turned by the angle."""
    a = math.radians(angle_deg); ca, sa = math.cos(a), math.sin(a)
    def rp(x, y):
        if mirror: x = -x
        return ca * x - sa * y, sa * x + ca * y
    gems = []
    for x, y, dz, db in c['gems']:
        X, Y = rp(x, y); gems.append((X, Y, dz, db))
    boxes = []
    for x0, x1, y0, y1, h in c['boxes']:
        cx, cy = rp((x0 + x1) / 2, (y0 + y1) / 2)
        boxes.append((cx, cy, x1 - x0, y1 - y0, h, angle_deg))
    return {'src': c['src'], 'id': c['id'], 'gems': gems, 'boxes': boxes, 'angle': angle_deg, 'mirror': mirror}


def tiles(boxes):
    """Cover each box with SQUARE, uniformly scaled tiles (side = min(w, d, TILE_MAX)) spread evenly so they overlap, each
    inflated by TILE_INFLATE per side. Uniform scale per interior; overlapping hulls, no hairline seams or corner points."""
    out = []
    for cx, cy, w, d, h, ang in boxes:
        side = min(w, d, TILE_MAX)
        nu, nv = max(1, int(math.ceil(w / side - 1e-6))), max(1, int(math.ceil(d / side - 1e-6)))
        a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
        for i in range(nu):
            for j in range(nv):
                lu = -w / 2 + side / 2 + (i * (w - side) / (nu - 1) if nu > 1 else 0.0)   # centres from one end to the other
                lv = -d / 2 + side / 2 + (j * (d - side) / (nv - 1) if nv > 1 else 0.0)
                out.append((cx + ca * lu - sa * lv, cy + sa * lu + ca * lv, side + 2 * TILE_INFLATE, side + 2 * TILE_INFLATE, h, ang))
    return out


def mission_text(fname, title, placed, desc):
    fn = ''.join(ch for ch in fname if ch.isalnum())
    lines, gem_lines = [], []
    for i, (c, tx, ty) in enumerate(placed):
        lines.append(f'   // ---- {c["id"]} ({len(c["gems"])} gems, {len(c["boxes"])} boxes) at ({tx:.0f}, {ty:.0f}) ----')
        for cx, cy, w, d, h, ang in tiles(c['boxes']):
            sx, sy, sz = w / CUBE_SIZE, d / CUBE_SIZE, h / CUBE_SIZE
            # the cube's centre sits at R(ang) * (offset * scale) from its position: position = centre - that
            a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
            ox, oy, oz = CUBE_OFF[0] * sx, CUBE_OFF[1] * sy, CUBE_OFF[2] * sz
            rox, roy = ca * ox - sa * oy, sa * ox + ca * oy
            px, py, pz = cx + tx - rox, cy + ty - roy, FLOOR_TOP + h / 2 - oz
            # the engine turns an interior the opposite way to the textbook matrix (2026-10-07, teleport probe; see
            # generate_terrain_map._axis_angle_matrix), so the file carries -ang to get the +ang the gems were placed with
            lines.append('   new InteriorInstance() { position = "%.3f %.3f %.3f"; rotation = "0 0 1 %.2f"; scale = "%.4f %.4f %.4f"; '
                         'interiorFile = "~/data/interiors_mbp/Cube.dif"; showTerrainInside = "0"; };' % (px, py, pz, -ang, sx, sy, sz))
        gem_lines.append(f'      new SimGroup(Cluster{i}) {{   // {c["id"]}')
        for x, y, dz, db in c['gems']:
            gem_lines.append('         new Item() { position = "%.3f %.3f %.3f"; rotation = "1 0 0 0"; scale = "1 1 1"; dataBlock = "%s"; '
                             'collideable = "0"; static = "1"; rotate = "1"; };' % (x + tx, y + ty, FLOOR_TOP + dz, db))
        gem_lines.append('      };')
    half = FLOOR_SCALE * 5 / 2
    ngems = sum(len(c['gems']) for c, _, _ in placed)
    return f'''//--- INFO BEGIN ---
//Generated by ml_agent/nav/maps/build_cluster_variants.py: real gem clusters on a flat floor (skill 4 training).
function MP_PQ_{fn}_GetMissionInfo() {{
	return new ScriptObject() {{
		name = "{title}";
		type = "Custom";
		level = "9998";
		desc = "{desc}";
		artist = "AI Training";
		music = "Astrolabe.ogg";
		game = "Ultra";
		gameMode = "Hunt";
		blast = "0";
		Time = "180000";
		maxGemsPerSpawn = "8";
		minGemsPerSpawn = "4";
		minPointsPerSpawn = "4";
		radiusFromGem = "8";
		CustomRadarRule = $Radar::Flags::Gems | $Radar::Flags::EndPad;
		alarmStartTime = "15";
		easterEgg = "0";
		gems = "{ngems}";
		interior0 = $usermods @ "/data/interiors_mbg/addon/smallplatform.dif";
		interior1 = $usermods @ "/data/interiors_mbp/Cube.dif";
		interiors = "2";
		maxScore = "7";
		platinumScore0 = "3";
		platinumScore1 = "5";
		score0 = "2";
		score1 = "4";
		ultimateScore0 = "5";
		ultimateScore1 = "7";
	}};
}}
//--- INFO END ---
if ($loadingMissionInfo) return;
//--- CLIENT SCRIPTS BEGIN ---
//--- CLIENT SCRIPTS END ---
//--- SERVER SCRIPTS BEGIN ---
//--- SERVER SCRIPTS END ---
//--- MISSION BEGIN ---
function MP_PQ_{fn}_LoadMission() {{
	return new SimGroup(MissionGroup) {{

   new ScriptObject(MissionInfo) {{
      artist = "AI Training";
      blast = "0";
      desc = "{desc}";
      difficulty = "1";
      game = "Ultra";
      gameMode = "hunt";
      gemGroups = "0";
      level = "1";
      maxGemsPerSpawn = "8";
      minGemsPerSpawn = "4";
      minPointsPerSpawn = "4";
      music = "Astrolabe.ogg";
      name = "{title}";
      radiusFromGem = "8";
      Time = "180000";
      type = "Beginner";
   }};

   new Sky(Sky) {{
      position = "336 136 0";
      rotation = "1 0 0 0";
      scale = "1 1 1";
      materialList = "~/data/skies/Beginner/Beginner_Sky.dml";
      cloudHeightPer[0] = "0";
      cloudHeightPer[1] = "0";
      cloudHeightPer[2] = "0";
      cloudSpeed1 = "0.0001";
      cloudSpeed2 = "0.0002";
      cloudSpeed3 = "0.0003";
      visibleDistance = "500";
      fogDistance = "300";
      fogColor = "0.6 0.6 0.6 1";
      useSkyTextures = "1";
      renderBottomTexture = "1";
      noRenderBans = "1";
   }};

   new Sun() {{
      direction = "0.638261 0.459006 -0.61801";
      color = "1.4 1.2 0.4 1";
      ambient = "0.3 0.3 0.4 1";
   }};

   new InteriorInstance() {{
      position = "{-FLOOR_OFF[0] * FLOOR_SCALE:.2f} {-FLOOR_OFF[1] * FLOOR_SCALE:.2f} 0";
      rotation = "1 0 0 0";
      scale = "{FLOOR_SCALE} {FLOOR_SCALE} 1";
      interiorFile = "~/data/interiors_mbg/addon/smallplatform.dif";
      showTerrainInside = "0";
   }};

{chr(10).join(lines)}

   new Trigger(Bounds) {{
      position = "-200 200 -10";
      rotation = "1 0 0 0";
      scale = "400 400 50";
      dataBlock = "InBoundsTrigger";
      polyhedron = "0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000000 0.0000000 -1.0000000 0.0000000 0.0000000 0.0000000 1.0000000";
   }};

   new Trigger(SpawnPoint) {{
      position = "0 {-(half - 4):.1f} 1";
      rotation = "1 0 0 0";
      scale = "1 1 1";
      dataBlock = "SpawnTrigger";
      polyhedron = "0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000000 0.0000000 -1.0000000 0.0000000 0.0000000 0.0000000 1.0000000";
      add = "0 0 1";
      gravity = "0";
   }};

   new StaticShape(StartPoint) {{
      position = "0 {-(half - 4):.1f} {FLOOR_TOP + 0.2:.2f}";
      rotation = "1 0 0 0";
      scale = "0.01 0.01 0.01";
      dataBlock = "StartPad_MBU";
   }};

   new SimGroup(GemGroups) {{
{chr(10).join(gem_lines)}
   }};
	}};
}}
//--- MISSION END ---
'''


def place(clusters, rng, per_map):
    picks = rng.sample(clusters, min(per_map, len(clusters)))
    cols = int(math.ceil(math.sqrt(len(picks)))); rows = int(math.ceil(len(picks) / cols))
    slots = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    ox, oy = -(cols - 1) * SPACING / 2, -(rows - 1) * SPACING / 2
    placed = []
    for c, (col, row) in zip(picks, slots):
        t = transform(c, rng.uniform(0.0, 360.0), rng.random() < 0.5)   # any angle (operator 10-07: not just quarter turns)
        jx, jy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        placed.append((t, ox + col * SPACING + jx, oy + row * SPACING + jy))
    return placed


def write_mission(fname, text):
    path = os.path.join(CUSTOM, fname + '.mcs')
    tmp = path + '.new'
    open(tmp, 'w', newline='\n').write(text)
    os.replace(tmp, path)
    for stale in (path + '.dso',):
        if os.path.exists(stale): os.remove(stale)
    return path


def build_terrain(fname):
    r = subprocess.run([sys.executable, os.path.join(ML, 'generate_terrain_map.py'), fname], capture_output=True, text=True, cwd=ML)
    tail = [l for l in r.stdout.splitlines() if l.startswith('Grid') or l.startswith('Saved')][:2]
    return r.returncode, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--variants', type=int, default=8)
    ap.add_argument('--per-map', type=int, default=14)
    ap.add_argument('--clusters-json', default=POOL)
    ap.add_argument('--no-terrain', action='store_true')
    ap.add_argument('--pitch', type=float, default=None, help='cluster pitch in u (default SPACING)')
    ap.add_argument('--skip-holdout', action='store_true', help='leave BlockClustersHoldout_Hunt untouched (keeps the test map fixed)')
    a = ap.parse_args()
    global SPACING
    if a.pitch: SPACING = a.pitch
    seed = a.seed if a.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    pool = usable_clusters(a.clusters_json)
    train = [c for c in pool if c['src'] != HOLDOUT_SOURCE]
    hold = [c for c in pool if c['src'] == HOLDOUT_SOURCE]
    print(f'seed {seed}: pool {len(pool)} usable clusters, training {len(train)} from '
          f'{sorted(set(c["src"] for c in train))}, held out {len(hold)} from {HOLDOUT_SOURCE}')
    names = []
    for v in range(a.variants):
        fname = f'BlockClusters_v{v}_Hunt'
        placed = place(train, rng, a.per_map)
        write_mission(fname, mission_text(fname, f'Block Clusters v{v}', placed,
                                          f'Training variant {v} (seed {seed}): {len(placed)} real clusters, rotated and mirrored'))
        names.append((fname, placed))
    if not a.skip_holdout:
        fname = 'BlockClustersHoldout_Hunt'
        placed = place(hold, rng, a.per_map)
        write_mission(fname, mission_text(fname, 'Block Clusters Holdout', placed,
                                          f'HELD-OUT test map (seed {seed}): {len(placed)} clusters from {HOLDOUT_SOURCE}, never trained on'))
        names.append((fname, placed))
    for fname, placed in names:
        srcs = sorted(set(c['src'] for c, _, _ in placed))
        print(f'  {fname}: {len(placed)} clusters, {sum(len(c["gems"]) for c, _, _ in placed)} gems, sources {srcs}')
        if not a.no_terrain:
            rc, tail = build_terrain(fname)
            print('    terrain', 'ok' if rc == 0 else f'FAILED rc {rc}', tail[0] if tail else '')
    json.dump({'seed': seed, 'maps': {f: [c['id'] for c, _, _ in p] for f, p in names}},
              open(os.path.join(HERE, 'cluster_variants_last.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
