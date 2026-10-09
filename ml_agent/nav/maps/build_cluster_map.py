"""Place extracted real gem clusters (boxes + gems) on a flat training map: BlockClusters_Hunt.mcs."""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r'C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev'
OUT_MIS = os.path.join(REPO, 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt', 'custom', 'BlockClusters_Hunt.mcs')

MAX_BOX_H = 1.5      # boxes taller than a plain jump are platform edges, not the gem boxes
MAX_BOX_AREA = 20.0  # u^2
FLOOR_TOP = 0.38     # smallplatform.dif top at scale z 1
FLOOR_SCALE = 18     # smallplatform is 5 x 5 -> 90 x 90 u
SPACING = 18.0       # cluster pitch
# Cube.dif spans x -2.06..1.94, y -1.94..2.06, z -1.88..2.12 at scale 1 (centre offset -0.06, +0.06, +0.12)
CUBE_OFF = (-0.06, 0.06, 0.12)
CUBE_SIZE = 4.0

cl = json.load(open(os.path.join(HERE, 'clusters', 'clusters.json')))
picked = []
for name, rows in cl.items():
    for ci, r in enumerate(rows):
        base = r['base']
        boxes = [b for b in r['boxes'] if (b['top'] - base) <= MAX_BOX_H and b['area'] <= MAX_BOX_AREA]
        if not boxes: continue
        # every gem must sit on the base floor or on a kept box
        ok = True
        for (x, y, z, db), f in zip(r['gems'], r['gem_floor']):
            if f is None: ok = False; break
            if abs(f - base) > 0.1 and not any(abs(f - b['top']) < 0.1 and b['x0'] - 0.3 <= x <= b['x1'] + 0.3 and b['y0'] - 0.3 <= y <= b['y1'] + 0.3 for b in boxes):
                ok = False; break
        if not ok: continue
        xs = [g[0] for g in r['gems']] + [b['x0'] for b in boxes] + [b['x1'] for b in boxes]
        ys = [g[1] for g in r['gems']] + [b['y0'] for b in boxes] + [b['y1'] for b in boxes]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span > SPACING - 4: continue
        picked.append({'src': f'{name} c{ci:02d}', 'gems': r['gems'], 'boxes': boxes, 'base': base, 'cx': cx, 'cy': cy, 'span': span})
print(f'{len(picked)} clusters picked')
for p in picked: print(f"  {p['src']:24s} gems {len(p['gems'])} boxes {len(p['boxes'])} span {p['span']:.1f}")

n = len(picked); cols = int(math.ceil(math.sqrt(n))); rows_n = int(math.ceil(n / cols))
ox = -(cols - 1) * SPACING / 2; oy = -(rows_n - 1) * SPACING / 2
lines = []
gem_lines = []
for i, p in enumerate(picked):
    tx = ox + (i % cols) * SPACING; ty = oy + (i // cols) * SPACING
    lines.append(f'   // ---- {p["src"]} ({len(p["gems"])} gems, {len(p["boxes"])} boxes) ----')
    for b in p['boxes']:
        w, d, h = b['x1'] - b['x0'], b['y1'] - b['y0'], b['top'] - p['base']
        sx, sy, sz = w / CUBE_SIZE, d / CUBE_SIZE, h / CUBE_SIZE
        ccx = (b['x0'] + b['x1']) / 2 - p['cx'] + tx; ccy = (b['y0'] + b['y1']) / 2 - p['cy'] + ty; ccz = FLOOR_TOP + h / 2
        px, py, pz = ccx - CUBE_OFF[0] * sx, ccy - CUBE_OFF[1] * sy, ccz - CUBE_OFF[2] * sz
        lines.append('   new InteriorInstance() { position = "%.3f %.3f %.3f"; rotation = "1 0 0 0"; scale = "%.4f %.4f %.4f"; '
                     'interiorFile = "~/data/interiors_mbp/Cube.dif"; showTerrainInside = "0"; };' % (px, py, pz, sx, sy, sz))
    gem_lines.append(f'      new SimGroup(Cluster{i}) {{   // {p["src"]}')
    for (x, y, z, db) in p['gems']:
        gx, gy, gz = x - p['cx'] + tx, y - p['cy'] + ty, FLOOR_TOP + (z - p['base'])
        if not db.endswith('_MBU'): db = db + '_MBU'   # the Ultra game mode's gem datablocks (KOTM uses them)
        gem_lines.append('         new Item() { position = "%.3f %.3f %.3f"; rotation = "1 0 0 0"; scale = "1 1 1"; dataBlock = "%s"; collideable = "0"; static = "1"; rotate = "1"; };' % (gx, gy, gz, db))
    gem_lines.append('      };')

half = FLOOR_SCALE * 5 / 2
mis = f'''//--- INFO BEGIN ---
//Mission information for the level select. Generated from the MissionInfo object except with extra goodies.
function MP_PQ_BlockClustersHunt_GetMissionInfo() {{
	return new ScriptObject() {{
		name = "Block Clusters";
		type = "Custom";
		level = "9998";
		desc = "Real gem clusters from Sprawl, Duplex and ExampleMission on a flat floor (skill 4 training)";
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
		gems = "{sum(len(p['gems']) for p in picked)}";
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
//Don't continue loading if this just wants the mission info
if ($loadingMissionInfo) return;
//--- CLIENT SCRIPTS BEGIN ---
//--- CLIENT SCRIPTS END ---
//--- SERVER SCRIPTS BEGIN ---
//--- SERVER SCRIPTS END ---
//--- MISSION BEGIN ---
function MP_PQ_BlockClustersHunt_LoadMission() {{
	return new SimGroup(MissionGroup) {{

   new ScriptObject(MissionInfo) {{
      artist = "AI Training";
      blast = "0";
      desc = "Real gem clusters from Sprawl, Duplex and ExampleMission on a flat floor (skill 4 training)";
      difficulty = "1";
      game = "Ultra";
      gameMode = "hunt";
      gemGroups = "0";
      level = "1";
      maxGemsPerSpawn = "8";
      minGemsPerSpawn = "4";
      minPointsPerSpawn = "4";
      music = "Astrolabe.ogg";
      name = "Block Clusters";
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

   // Flat floor: smallplatform (5 x 5 x 0.5, top at z 0.38) scaled {FLOOR_SCALE} -> {FLOOR_SCALE * 5} x {FLOOR_SCALE * 5} u
   new InteriorInstance() {{
      position = "0 0 0";
      rotation = "1 0 0 0";
      scale = "{FLOOR_SCALE} {FLOOR_SCALE} 1";
      interiorFile = "~/data/interiors_mbg/addon/smallplatform.dif";
      showTerrainInside = "0";
   }};

   // Boxes the gems sit on: Cube.dif (4 u cube) scaled to each real box
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
open(OUT_MIS, 'w', newline='\n').write(mis)
print('wrote', OUT_MIS, 'clusters', n, 'floor', FLOOR_SCALE * 5, 'u')
json.dump(picked, open(os.path.join(HERE, 'clusters', 'picked.json'), 'w'), indent=1)
