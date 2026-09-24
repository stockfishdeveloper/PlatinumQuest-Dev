"""Live dashboard for the navigator trainer (waypoint task).

    python -m nav.dashboard_nav          then open http://localhost:8990

Reads every logs/nav/train_nav_*.log (all runs, resumed checkpoints included), parses
the NAV / SEG / MAP / ARRIVE / saved lines, and pushes the whole picture to the page
over server-sent events every few seconds. Same look and feel as dashboard.py, but the
numbers are the navigator's: seconds between pickups, arrival rate, falls per 100 u, speed,
per-map breakdown, and PPO health. Runs independently of the trainer (tails the logs).
"""
import os
import re
import numpy as np
import sys
import glob
import json
import time
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 8990
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(HERE, 'logs', 'nav')
CKPT_DIR = os.path.join(HERE, 'models', 'nav')
PUSH_EVERY_S = 3.0

# 2026-09-22: the NAV line is parsed as key=value pairs. The old fixed regex silently stopped matching
# when gems=/pickup=/sgem=/entd=/ec= were added to the line, and the dashboard showed nothing new.
KV_RE = re.compile(r'(\w+)=(-?[\d.,]+)')
NAV_HEAD_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] NAV upd=(\d+)(?: map=(\S+))?')
MAPS_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] MAPS (.*)$')
HUMAN_S_PER_GEM = 1.57          # seconds between pickups in the KOTM demo (682 gems, 18.2 min)

SAVED_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] saved (\S+)')
MAP_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] (?:MAP|mission) ([A-Za-z0-9_]+)')
RTF_RE = re.compile(r'rtf[= ]([\d.]+)')
DPS_RE = re.compile(r'dps=(\d+)')
RESUME_RE = re.compile(r'resumed .*: update (\d+)')
GAME_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] \[(\d+)\] GAME map=(\S+) points=([\d.]+) gems=(\d+) falls=(\d+)')


def parse_logs():
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'train_nav_*.log')))
    by_update = {}
    events = []            # (update_at_time, kind, text)
    last_update = 0
    counters = {'synthetic_falls': 0, 'respawn_misses': 0, 'teleport_ignored': 0, 'rounds': 0, 'restarts': 0}
    rtfs = []
    games = []             # real-gem training rounds: (update, inst, map, points, gems, falls, time)
    current_map = ''
    pending_maps = None    # the MAPS line (per-map stats) is logged just before its NAV line
    for f in files:
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines:
            m = MAPS_RE.search(line)
            if m:
                pending_maps = {}
                for part in m.group(2).split(' | '):
                    name, _, rest = part.partition(':')
                    kv = {k: float(v.replace(',', '')) for k, v in KV_RE.findall(rest)}
                    pending_maps[name.strip()] = {'n': kv.get('n'), 'arrive': kv.get('arrive'), 'falls100': kv.get('falls100'),
                                                  'speed': kv.get('speed'), 'gems': kv.get('gems'), 'pickup': kv.get('pickup'),
                                                  'sgem': kv.get('sgem')}
                continue
            m = NAV_HEAD_RE.search(line)
            if m:
                upd = int(m.group(2))
                kv = {k: v for k, v in KV_RE.findall(line)}
                fl = lambda k, d=None: float(kv[k].replace(',', '')) if k in kv else d
                rec = {
                    'time': m.group(1), 'update': upd, 'map': m.group(3) or current_map or '?', 'r': fl('r'),
                    'steps': int(fl('steps', 0)), 'segs': int(fl('segs', 0)), 'arrive': fl('arrive', 0.0),
                    'falls100': fl('falls100', 0.0), 'speed': fl('speed', 0.0), 'rew': fl('rew', 0.0),
                    'pl': fl('pl', 0.0), 'vl': fl('vl', 0.0), 'ent': fl('ent', 0.0), 'kl': fl('kl', 0.0),
                    'clip': fl('clip', 0.0), 'gn': fl('gn', 0.0), 'ep': int(fl('ep', 0)), 'dstd': fl('dstd', 0.0),
                    'flips': int(fl('flips', 0)), 'upd_s': fl('upd_s', 0.0), 'wall_s': fl('wall_s', 0.0),
                    'sgem': fl('sgem'), 'pickup': fl('pickup'), 'gems': fl('gems'),
                    'dps': fl('dps'),                                    # multi-instance trainer: decisions/s it measured itself
                    'maps': pending_maps,
                }
                pending_maps = None
                by_update[upd] = rec
                last_update = max(last_update, upd)
                continue
            m = GAME_RE.search(line)
            if m:
                games.append((last_update, int(m.group(2)), m.group(3), float(m.group(4)), int(m.group(5)), int(m.group(6)), m.group(1)))
                continue
            m = MAP_RE.search(line)
            if m:
                current_map = m.group(2)
                if ' MAP ' in line:
                    events.append((last_update, 'map', f'{m.group(1)} map -> {current_map}'))
                continue
            m = SAVED_RE.search(line)
            if m:
                events.append((last_update, 'ckpt', f'{m.group(1)} {os.path.basename(m.group(2))}'))
                continue
            m = RESUME_RE.search(line)
            if m:
                counters['restarts'] += 1
                resume_at = int(m.group(1))
                # A resume BELOW the last update is a revert to an older checkpoint (2026-09-22 did it
                # twice). Records of the abandoned lineage past that point would otherwise sit at the
                # end of every chart and hide the live run, so drop them; the new run refills them.
                if resume_at < last_update:
                    for k in [k for k in by_update if k > resume_at]:
                        del by_update[k]
                    games = [g for g in games if g[0] <= resume_at]
                    last_update = resume_at
                events.append((resume_at, 'resume', f'trainer (re)started at update {resume_at}'))
                continue
            if 'fall without OOB flag' in line:
                counters['synthetic_falls'] += 1
            elif 'RESPAWN did not take' in line or 'still off the map' in line:
                counters['respawn_misses'] += 1
            elif 'teleport to' in line and 'ignored' in line:
                counters['teleport_ignored'] += 1
            elif 'FRAME FLIP' in line:
                counters['frame_flips'] = counters.get('frame_flips', 0) + 1
            elif 'new round' in line:
                counters['rounds'] += 1
                m = RTF_RE.search(line)
                if m:
                    rtfs.append(float(m.group(1)))
    updates = [by_update[k] for k in sorted(by_update)]
    return updates, events, counters, rtfs, games


def eval_series():
    """Real-round evaluations from logs/nav/real_run_<map>_<ckpt>.json: per map, one entry per eval with the
    checkpoint's update and every round's points. Runs of fewer than 3 rounds (watch runs) are skipped."""
    out = {}
    for f in glob.glob(os.path.join(LOG_DIR, 'real_run_*_Hunt_*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('update') is None or len(d.get('rounds') or []) < 3:
            continue
        pts = [float(r['points']) for r in d['rounds']]
        out.setdefault(d['map'], []).append({'update': int(d['update']), 'points': pts, 'mean': sum(pts) / len(pts),
                                             'when': d.get('when', ''), 'ckpt': d.get('ckpt', '')})
    for m in out:
        out[m].sort(key=lambda e: (e['update'], e['when']))
    return out


HUMAN_POINTS = {'KingOfTheMarble_Hunt': 143.5, 'FlatGemTraining_Hunt': 104.0}


def build_state():
    updates, events, counters, rtfs, games = parse_logs()
    maps = []
    for u in updates:
        if u['map'] not in maps:
            maps.append(u['map'])
    # True per-map series from the MAPS lines (a map MIX across instances logs one pooled NAV line whose
    # map= is "A+B+C"; the MAPS line carries each map's own numbers).
    pm = {}
    for u in updates:
        for name, v in (u.get('maps') or {}).items():
            d = pm.setdefault(name, {k: [] for k in ('update', 'arrive', 'falls100', 'speed', 'sgem', 'pickup', 'gems')})
            d['update'].append(u['update'])
            for k in ('arrive', 'falls100', 'speed', 'sgem', 'pickup', 'gems'):
                d[k].append(v.get(k))
    per_map = {}
    if pm:
        for mp, d in pm.items():
            vals = lambda k: [x for x in d[k] if x is not None]
            per_map[mp] = {
                'updates': len(d['update']), 'last': d['update'][-1], 'arrive': d['arrive'][-1] or 0.0,
                'best_arrive': max(vals('arrive') or [0.0]), 'falls100': d['falls100'][-1] or 0.0,
                'best_falls100': min(vals('falls100') or [0.0]), 'speed': d['speed'][-1] or 0.0,
                'best_speed': max(vals('speed') or [0.0]), 'sgem': d['sgem'][-1], 'best_sgem': min(vals('sgem') or [0.0]) or None,
                'r': updates[-1]['r'],
            }
    else:
        for mp in maps:
            us = [u for u in updates if u['map'] == mp]
            per_map[mp] = {
                'updates': len(us), 'last': us[-1]['update'], 'arrive': us[-1]['arrive'], 'best_arrive': max(u['arrive'] for u in us),
                'falls100': us[-1]['falls100'], 'best_falls100': min(u['falls100'] for u in us), 'speed': us[-1]['speed'],
                'best_speed': max(u['speed'] for u in us), 'r': us[-1]['r'], 'sgem': None, 'best_sgem': None,
            }
    latest = updates[-1] if updates else None
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'nav_*.pth')), key=os.path.getmtime)
    latest_ckpt = os.path.basename(ckpts[-1]) if ckpts else '--'
    latest_ckpt_time = datetime.fromtimestamp(os.path.getmtime(ckpts[-1])).strftime('%H:%M:%S') if ckpts else '--'
    stdout = os.path.join(LOG_DIR, 'stdout.txt')
    alive = os.path.exists(stdout) and (time.time() - os.path.getmtime(stdout)) < 120
    recent = updates[-30:]
    dec_per_s = (np.mean([u['dps'] for u in recent if u.get('dps')]) if any(u.get('dps') for u in recent)
                 else sum(2048 for _ in recent) / max(sum(u['wall_s'] for u in recent), 1.0)) if recent else 0.0
    return {
        'now': datetime.now().strftime('%H:%M:%S'), 'trainer_alive': alive,
        'latest': latest, 'maps': maps, 'per_map': per_map, 'events': events[-40:], 'counters': counters,
        'rtf': rtfs[-1] if rtfs else None, 'dec_per_s': dec_per_s,
        'ckpt': latest_ckpt, 'ckpt_time': latest_ckpt_time, 'n_updates': len(updates),
        'series': {k: [u.get(k) for u in updates] for k in ('update', 'map', 'r', 'arrive', 'falls100', 'speed', 'rew', 'pl', 'vl',
                                                             'ent', 'kl', 'clip', 'gn', 'dstd', 'wall_s', 'steps', 'ep', 'sgem', 'pickup')},
        'pm': pm, 'human_sgem': HUMAN_S_PER_GEM, 'evals': eval_series(), 'human_points': HUMAN_POINTS,
        'games': [{'update': g[0], 'inst': g[1], 'map': g[2], 'points': g[3], 'gems': g[4], 'falls': g[5], 'time': g[6]} for g in games],
    }


_HEAT_CACHE = {}


def build_heatmap(map_name, res=1.0, min_samples=3):
    """Mean marble speed per `res` u cell from the latest real-round trace of `map_name`
    (logs/nav/real_trace_<map>.csv, written by nav/real_run.py), plus the floor mask from the
    terrain map so the page can draw the map under the colours. Cached by trace mtime."""
    import csv
    if not re.fullmatch(r'[A-Za-z0-9_]+', map_name or ''):
        return {'error': 'bad map name'}
    trace = os.path.join(LOG_DIR, f'real_trace_{map_name}.csv')
    terrain = os.path.join(HERE, 'terrain_maps', f'terrain_{map_name}.npz')
    if not os.path.exists(trace):
        return {'error': f'no real-round trace for {map_name} yet (run real_run.ps1 -Map {map_name})'}
    key = (map_name, os.path.getmtime(trace))
    if key in _HEAT_CACHE:
        return _HEAT_CACHE[key]
    rows = list(csv.DictReader(open(trace)))
    x = np.array([float(r['x']) for r in rows]); y = np.array([float(r['y']) for r in rows])
    sp = np.array([float(r['speed']) for r in rows]); fl = np.array([float(r['on_floor']) for r in rows]) > 0
    jp = np.array([float(r.get('jump', 0) or 0) for r in rows]) > 0.5
    if os.path.exists(terrain):
        T = np.load(terrain); txs, tys = T['xs'], T['ys']; fine = np.isfinite(T['heights'][0])
        x0, x1, y0, y1 = float(np.floor(txs.min())), float(np.ceil(txs.max())), float(np.floor(tys.min())), float(np.ceil(tys.max()))
    else:
        fine = None; x0, x1, y0, y1 = np.floor(x.min()), np.ceil(x.max()), np.floor(y.min()), np.ceil(y.max())
    xe = np.arange(x0, x1 + res, res); ye = np.arange(y0, y1 + res, res)
    cnt, _, _ = np.histogram2d(x[fl], y[fl], bins=[xe, ye]); ssum, _, _ = np.histogram2d(x[fl], y[fl], bins=[xe, ye], weights=sp[fl])
    mean = np.where(cnt >= min_samples, ssum / np.maximum(cnt, 1), np.nan)
    jsum, _, _ = np.histogram2d(x[fl & jp], y[fl & jp], bins=[xe, ye])          # takeoffs: jump commanded on the floor
    jrate = np.where(cnt >= min_samples, 100.0 * jsum / np.maximum(cnt, 1), np.nan)   # % of floor decisions in the cell
    floor = np.zeros(mean.shape, bool)
    if fine is not None:
        fx = np.clip(np.digitize(txs, xe) - 1, 0, mean.shape[0] - 1); fy = np.clip(np.digitize(tys, ye) - 1, 0, mean.shape[1] - 1)
        jj, ii = np.where(fine); floor[fx[ii], fy[jj]] = True
    z = [[(None if not np.isfinite(mean[i, j]) else round(float(mean[i, j]), 2)) for i in range(mean.shape[0])] for j in range(mean.shape[1])]
    fz = [[(1 if floor[i, j] else None) for i in range(mean.shape[0])] for j in range(mean.shape[1])]
    jz = [[(None if not np.isfinite(jrate[i, j]) else round(float(jrate[i, j]), 2)) for i in range(mean.shape[0])] for j in range(mean.shape[1])]
    rounds = len(set(r['round'] for r in rows))
    out = {'map': map_name, 'x': [float(v) for v in (xe[:-1] + res / 2)], 'y': [float(v) for v in (ye[:-1] + res / 2)], 'z': z, 'floor': fz, 'jz': jz, 'takeoffs': int((fl & jp).sum()),
           'decisions': int(fl.sum()), 'rounds': rounds, 'trace_time': datetime.fromtimestamp(os.path.getmtime(trace)).strftime('%Y-%m-%d %H:%M'),
           'p10': float(np.nanpercentile(mean, 10)), 'p90': float(np.nanpercentile(mean, 90))}
    _HEAT_CACHE.clear(); _HEAT_CACHE[key] = out
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/':
            body = HTML.encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith('/heatmap'):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query); name = (q.get('map') or ['KingOfTheMarble_Hunt'])[0]
            try:
                body = json.dumps(build_heatmap(name)).encode('utf-8')
            except Exception as e:      # never take the page down over one bad trace
                body = json.dumps({'error': f'{type(e).__name__}: {e}'}).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == '/state':
            body = json.dumps(build_state()).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == '/stream':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache'); self.send_header('Connection', 'keep-alive'); self.end_headers()
            try:
                while True:
                    self.wfile.write(f'data: {json.dumps(build_state())}\n\n'.encode('utf-8')); self.wfile.flush()
                    time.sleep(PUSH_EVERY_S)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
        else:
            self.send_response(404); self.end_headers()


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>PlatinumQuest Navigator Training</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root { --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#e6edf3; --muted:#7d8590; --green:#3fb950; --yellow:#d29922; --red:#f85149; --gold:#f0c040; --blue:#58a6ff; --purple:#bc8cff; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:'Consolas','SF Mono','Fira Code',monospace; }
#header { position:sticky; top:0; z-index:100; background:var(--card); border-bottom:1px solid var(--border); padding:10px 16px; }
#header-top { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
#header h1 { font-size:1rem; font-weight:600; white-space:nowrap; }
.badge { font-size:0.7rem; padding:2px 8px; border-radius:10px; font-weight:600; }
.badge-live { background:#238636; color:white; } .badge-reconnecting { background:var(--yellow); color:black; }
.badge-game { background:var(--border); color:var(--muted); } .badge-game.connected { background:#238636; color:white; }
#meta-row { display:flex; gap:20px; flex-wrap:wrap; font-size:0.75rem; color:var(--muted); margin-top:6px; } #meta-row b { color:var(--text); }
#gauges { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; padding:10px 12px; }
@media (max-width:900px){ #gauges { grid-template-columns:repeat(3,1fr);} } @media (max-width:500px){ #gauges { grid-template-columns:repeat(2,1fr);} }
.gauge { background:var(--card); border:1px solid var(--border); border-radius:6px; padding:10px 8px; text-align:center; }
.gauge .label { font-size:0.65rem; color:var(--muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px; }
.gauge .value { font-size:1.3rem; font-weight:700; }
#charts { display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 12px 12px; } @media (max-width:900px){ #charts { grid-template-columns:1fr;} }
.chart-card { background:var(--card); border:1px solid var(--border); border-radius:6px; padding:8px; overflow:hidden; }
.chart-card.full-width { grid-column:1 / -1; }
.chart-title { font-size:0.7rem; color:var(--muted); margin-bottom:2px; text-transform:uppercase; letter-spacing:0.5px; }
#tables { display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 12px 24px; } @media (max-width:900px){ #tables { grid-template-columns:1fr;} }
table { width:100%; border-collapse:collapse; font-size:0.75rem; } th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:600; text-transform:uppercase; font-size:0.65rem; letter-spacing:0.5px; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.ev { color:var(--muted); } .ev.radius { color:var(--gold); } .ev.map { color:var(--blue); } .ev.ckpt { color:var(--green); } .ev.resume { color:var(--purple); }
.hint { font-size:0.65rem; color:var(--muted); padding:0 12px 10px; }
</style></head>
<body>
<div id="header">
  <div id="header-top">
    <h1>PlatinumQuest Navigator Training</h1>
    <span id="live-badge" class="badge badge-reconnecting">CONNECTING...</span>
    <span id="trainer-badge" class="badge badge-game">Trainer: --</span>
  </div>
  <div id="meta-row">
    <span>Update: <b id="m-update">--</b></span>
    <span>Decisions: <b id="m-steps">--</b></span>
    <span>Segments: <b id="m-segs">--</b></span>
    <span>Map: <b id="m-map">--</b></span>
    <span>Decisions/s: <b id="m-dps">--</b></span>
    <span>Game rtf: <b id="m-rtf">--</b>x</span>
    <span>Checkpoint: <b id="m-ckpt">--</b></span>
    <span>Show map: <select id="map-sel" style="background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:4px;font-family:inherit;font-size:0.75rem;padding:1px 4px"><option value="ALL">all maps (overlay)</option></select></span>
    <span>Window: <select id="win-sel" style="background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:4px;font-family:inherit;font-size:0.75rem;padding:1px 4px"><option value="200">last 200 updates</option><option value="500" selected>last 500</option><option value="1000">last 1000</option><option value="2000">last 2000</option><option value="0">all</option></select></span>
  </div>
</div>
<div id="gauges">
  <div class="gauge"><div class="label" id="g-sgem-label">Seconds between pickups</div><div class="value" id="g-sgem" style="color:#f0c040">--</div><div class="label" id="g-sgem-sub">human 1.57 s</div></div>
  <div class="gauge"><div class="label" id="g-best-label">Best points per game</div><div class="value" id="g-best" style="color:#3fb950">--</div><div class="label" id="g-best-sub">human 143.5 (KOTM)</div></div>
  <div class="gauge"><div class="label">Falls / 100 u</div><div class="value" id="g-falls" style="color:#f85149">--</div><div class="label">target &lt; 0.5</div></div>
  <div class="gauge"><div class="label">Speed u/s</div><div class="value" id="g-speed">--</div><div class="label">target &ge; 7</div></div>
  <div class="gauge"><div class="label">Entropy</div><div class="value" id="g-ent">--</div><div class="label" id="g-dstd">dir std --</div></div>
</div>
<div id="charts">
  <div class="chart-card"><div class="chart-title">SECONDS BETWEEN PICKUPS, selected map (pickup-to-pickup inside a group; lower is better; dashed = human 1.57 s on KOTM)</div><div id="c-sgem" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title" id="pts-title">POINTS PER TRAINING GAME, selected map (one bar per finished round in order, real score from the game; dashed = human; sampled policy, no stuck-breaker)</div><div id="c-points" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title" id="heat-title">Marble speed heat map, selected map (latest real rounds; green = fastest, red = slowest)</div><div id="c-heat" style="height:420px"></div></div>
  <div class="chart-card"><div class="chart-title" id="heatj-title">Jump heat map, selected map (takeoffs as % of floor decisions per cell; green = jumps a lot, red = never)</div><div id="c-heatj" style="height:420px"></div></div>
  <div class="chart-card"><div class="chart-title">Falls per 100 u, selected map</div><div id="c-falls" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Speed (u/s, on-floor travel), selected map</div><div id="c-speed" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Reward per segment (300-segment rolling, pooled over all maps)</div><div id="c-rew" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Entropy + direction std</div><div id="c-ent" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">KL per update + clip fraction</div><div id="c-kl" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Policy loss vs value loss</div><div id="c-loss" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Gradient norm</div><div id="c-gn" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Wall seconds per update (throughput; spikes = stalls / map switches)</div><div id="c-wall" style="height:230px"></div></div>
</div>
<div id="tables">
  <div class="chart-card"><div class="chart-title">Per map (latest slot vs best)</div><table id="t-maps"><thead><tr><th>map</th><th>updates</th><th>s/pickup</th><th>best</th><th>falls/100u</th><th>best</th><th>speed</th><th>best</th></tr></thead><tbody></tbody></table></div>
  <div class="chart-card"><div class="chart-title">Events (map switches, checkpoints, restarts)</div><table id="t-events"><tbody></tbody></table></div>
</div>
<div class="hint">Reads logs/nav/train_nav_*.log every 3 s. Falls per 100 u counts falls per distance travelled on the floor.</div>
<script>
const darkLayout = (extra) => Object.assign({
  paper_bgcolor:'#161b22', plot_bgcolor:'#0d1117', font:{color:'#e6edf3', size:10, family:'Consolas, monospace'},
  margin:{l:50, r:12, t:8, b:32}, showlegend:false,
  xaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false, title:'update'}, yaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false},
}, extra || {});
const cfg = {responsive:true, displayModeBar:false};
const palette = ['#3fb950','#58a6ff','#f0c040','#bc8cff','#f85149','#d29922','#39d3c3','#ff7b72'];
const legend = {showlegend:true, legend:{x:0, y:1.15, orientation:'h', font:{size:9}}};
['c-sgem','c-points','c-heat','c-heatj','c-falls','c-speed','c-rew','c-ent','c-kl','c-loss','c-gn','c-wall'].forEach(id => Plotly.newPlot(id, [], darkLayout(legend), cfg));

let ST = null;   // latest state, so per-map traces can read the MAPS-line series (st.pm)
// Map selector: one map at a time (default KingOfTheMarble, the scored map) or the overlay of all.
let SEL = (function(){ try { return localStorage.getItem('nav_map_sel') || 'KingOfTheMarble_Hunt'; } catch (e) { return 'KingOfTheMarble_Hunt'; } })();
let WIN = (function(){ try { return parseInt(localStorage.getItem('nav_win_sel') || '500'); } catch (e) { return 500; } })();
let LO = -Infinity;   // first update shown; recomputed on every state push
const winEl = document.getElementById('win-sel'); winEl.value = String(WIN);
winEl.addEventListener('change', () => { WIN = parseInt(winEl.value); try { localStorage.setItem('nav_win_sel', String(WIN)); } catch (e) {} if (ST) update(ST); });
function windowed(series) {   // copy of the pooled series restricted to update >= LO
  const keep = series.update.map(u => u >= LO); const out = {};
  for (const k of Object.keys(series)) out[k] = series[k].filter((_, i) => keep[i]);
  return out;
}
const selEl = document.getElementById('map-sel');
selEl.addEventListener('change', () => { SEL = selEl.value; try { localStorage.setItem('nav_map_sel', SEL); } catch (e) {} if (ST) update(ST); loadHeatmap(); });
let HEAT_LOADED = '';   // "<map>@<trace_time>" currently drawn, so we only redraw when the trace changes
async function loadHeatmap() {
  const name = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
  try {
    const h = await (await fetch('/heatmap?map=' + encodeURIComponent(name))).json();
    const title = document.getElementById('heat-title');
    if (h.error) { title.textContent = 'Marble speed heat map: ' + h.error; Plotly.react('c-heat', [], darkLayout(), cfg); HEAT_LOADED = ''; return; }
    const key = h.map + '@' + h.trace_time; if (key === HEAT_LOADED) return; HEAT_LOADED = key;
    title.textContent = 'Marble speed heat map, ' + h.map.replace('_Hunt','') + ' (' + h.rounds + ' real rounds, ' + h.decisions.toLocaleString() + ' floor decisions, trace ' + h.trace_time + '; green = fastest, red = slowest, 1 u cells)';
    const floor = {x:h.x, y:h.y, z:h.floor, type:'heatmap', colorscale:[[0,'#2a2f36'],[1,'#2a2f36']], showscale:false, hoverinfo:'skip', zmin:0, zmax:1};
    const speed = {x:h.x, y:h.y, z:h.z, type:'heatmap', zmin:2, zmax:11, colorscale:[[0,'#a50026'],[0.25,'#f46d43'],[0.5,'#fee08b'],[0.75,'#a6d96a'],[1,'#1a9850']],
                   colorbar:{title:{text:'u/s', font:{color:'#e6edf3'}}, tickfont:{color:'#e6edf3'}, thickness:10}, hovertemplate:'x %{x}, y %{y}: %{z} u/s<extra></extra>'};
    Plotly.react('c-heat', [floor, speed], darkLayout({margin:{l:40, r:10, t:8, b:32}, xaxis:{title:'x', gridcolor:'#21262d', color:'#7d8590', scaleanchor:'y', constrain:'domain'}, yaxis:{title:'y', gridcolor:'#21262d', color:'#7d8590'}}), cfg);
    if (h.jz) {
      document.getElementById('heatj-title').textContent = 'Jump heat map, ' + h.map.replace('_Hunt','') + ' (' + h.takeoffs + ' takeoffs in ' + h.rounds + ' real rounds; takeoffs as % of floor decisions per 1 u cell; green = jumps a lot, red = never)';
      const jump = {x:h.x, y:h.y, z:h.jz, type:'heatmap', zmin:0, zmax:5, colorscale:[[0,'#a50026'],[0.3,'#f46d43'],[0.6,'#fee08b'],[1,'#1a9850']],
                    colorbar:{title:{text:'% takeoff', font:{color:'#e6edf3'}}, tickfont:{color:'#e6edf3'}, thickness:10}, hovertemplate:'x %{x}, y %{y}: %{z}% takeoffs<extra></extra>'};
      Plotly.react('c-heatj', [floor, jump], darkLayout({margin:{l:40, r:10, t:8, b:32}, xaxis:{title:'x', gridcolor:'#21262d', color:'#7d8590', scaleanchor:'y', constrain:'domain'}, yaxis:{title:'y', gridcolor:'#21262d', color:'#7d8590'}}), cfg);
    }
  } catch (e) { console.error(e); }
}
loadHeatmap(); setInterval(loadHeatmap, 60000);
function syncSelector(st) {
  const names = st.pm ? Object.keys(st.pm) : [];
  const have = [...selEl.options].map(o => o.value);
  names.forEach(n => { if (!have.includes(n)) { const o = document.createElement('option'); o.value = n; o.textContent = n.replace('_Hunt',''); selEl.appendChild(o); } });
  if (![...selEl.options].some(o => o.value === SEL)) SEL = names.includes('KingOfTheMarble_Hunt') ? 'KingOfTheMarble_Hunt' : 'ALL';
  selEl.value = SEL;
}
function perMapTraces(s, key, mode) {
  const traces = [];
  if (ST && ST.pm && Object.keys(ST.pm).length) {
    const maps = Object.keys(ST.pm);
    if (!ST.pm[maps[0]][key]) {
      // pooled-only series (e.g. reward): one trace over the whole run
      const x = [], y = [];
      s.update.forEach((u, k) => { if (s[key][k] !== null && s[key][k] !== undefined) { x.push(u); y.push(s[key][k]); } });
      return [{x, y, type:'scatter', mode:'lines', line:{width:1.5, color:palette[0]}, name:'all maps (pooled)'}];
    }
    maps.forEach((mp, i) => {
      if (SEL !== 'ALL' && mp !== SEL) return;
      const d = ST.pm[mp]; if (!d[key]) return;
      const x = [], y = [];
      d.update.forEach((u, k) => { if (u >= LO && d[key][k] !== null && d[key][k] !== undefined) { x.push(u); y.push(d[key][k]); } });
      traces.push({x, y, type:'scatter', mode: mode || 'lines', line:{width:1.5, color:palette[i % palette.length]}, name: mp.replace('_Hunt','')});
    });
    return traces;
  }
  const maps = [...new Set(s.map)];
  maps.forEach((mp, i) => {
    const x = [], y = [];
    s.update.forEach((u, k) => { if (s.map[k] === mp && s[key][k] !== null) { x.push(u); y.push(s[key][k]); } });
    traces.push({x, y, type:'scatter', mode: mode || 'lines+markers', marker:{size:3}, line:{width:1.5, color:palette[i % palette.length]}, name: mp.replace('_Hunt','')});
  });
  return traces;
}
function fmt(v, d) { return (v === null || v === undefined) ? '--' : Number(v).toFixed(d === undefined ? 2 : d); }

function update(st) {
  ST = st;
  syncSelector(st);
  const allU = st.series.update; LO = (WIN > 0 && allU.length) ? allU[allU.length - 1] - WIN : -Infinity;
  const s = windowed(st.series), L = st.latest;
  const pmSel = SEL !== 'ALL' ? st.per_map[SEL] : null;   // the selected map's own numbers for the gauges
  const selName = SEL === 'ALL' ? 'all maps' : SEL.replace('_Hunt','');
  document.getElementById('g-sgem-label').textContent = 'Seconds between pickups (' + selName + ')';
  document.getElementById('g-sgem').textContent = (pmSel && pmSel.sgem) ? fmt(pmSel.sgem) + ' s' : (L && L.sgem ? fmt(L.sgem) + ' s' : '--');
  document.getElementById('g-sgem-sub').textContent = 'human 1.57 s (KOTM)' + ((pmSel && pmSel.best_sgem) ? ' | best ' + fmt(pmSel.best_sgem) : '');
  {
    const bmap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
    const all = (st.games || []).filter(g => g.map === bmap && g.gems <= 130);
    const best = all.length ? all.reduce((a, b) => (b.points > a.points ? b : a)) : null;
    document.getElementById('g-best-label').textContent = 'Best points per game (' + bmap.replace('_Hunt', '') + ')';
    document.getElementById('g-best').textContent = best ? best.points.toFixed(0) : '--';
    document.getElementById('g-best-sub').textContent = 'human 143.5 (KOTM)' + (best ? ' | upd ' + best.update + ' ' + best.time : '') + ' | ' + all.length + ' games';
  }
  {
    const tr = perMapTraces(s, 'sgem');
    if (s.update.length) tr.push({x:[s.update[0], s.update[s.update.length-1]], y:[st.human_sgem, st.human_sgem], type:'scatter', mode:'lines', line:{dash:'dash', color:'#e6edf3', width:1}, name:'human (KOTM demo)'});
    Plotly.react('c-sgem', tr, darkLayout(Object.assign({yaxis:{title:'s / pickup', gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);
  }
  {
    const emap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
    // one round cannot hold more than ~130 gems: larger entries are two rounds merged by a worker
    // that crossed a round end mid-respawn (fixed in vec_worker 17:00; older workers may still emit them)
    const gs = (st.games || []).filter(g => g.map === emap && g.update >= LO && g.gems <= 130);
    const x = gs.map((g, i) => i + 1), y = gs.map(g => g.points);
    let bi = -1; y.forEach((v, i) => { if (bi < 0 || v > y[bi]) bi = i; });
    const cols = y.map((v, i) => (i === bi ? '#3fb950' : '#f0883e'));
    const tr = [{x, y, type:'bar', marker:{color:cols}, hovertemplate:'%{y}<extra></extra>', name:'training game'}];
    if (gs.length >= 5) {
      const k = 10, mx = [], my = [];
      for (let i = k - 1; i < gs.length; i++) { let sum = 0; for (let j = i - k + 1; j <= i; j++) sum += gs[j].points; mx.push(i + 1); my.push(sum / k); }
      tr.push({x:mx, y:my, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:2}, name:'10-game mean'});
    }
    const hp = (st.human_points || {})[emap];
    if (hp && x.length) tr.push({x:[Math.min(...x), Math.max(...x)], y:[hp, hp], type:'scatter', mode:'lines', line:{dash:'dash', color:'#e6edf3', width:1}, name:'human'});
    Plotly.react('c-points', tr, darkLayout(Object.assign({yaxis:{title:'points', gridcolor:'#21262d', color:'#7d8590'}, xaxis:{title:'training game (in order)', gridcolor:'#21262d', color:'#7d8590'}, bargap:0.15}, legend)), cfg);
  }
  document.getElementById('trainer-badge').textContent = 'Trainer: ' + (st.trainer_alive ? 'running' : 'stopped');
  document.getElementById('trainer-badge').className = 'badge badge-game' + (st.trainer_alive ? ' connected' : '');
  if (L) {
    document.getElementById('m-update').textContent = L.update; document.getElementById('m-steps').textContent = L.steps.toLocaleString();
    document.getElementById('m-segs').textContent = L.segs; document.getElementById('m-map').textContent = L.map.replace('_Hunt','');
    const G = pmSel || L;   // selected map's arrive / falls / speed, pooled when 'all maps'
    document.getElementById('g-falls').textContent = fmt(G.falls100);
    document.getElementById('g-speed').textContent = fmt(G.speed, 1); 
    document.getElementById('g-ent').textContent = fmt(L.ent); document.getElementById('g-dstd').textContent = 'dir std ' + fmt(L.dstd);

  }
  document.getElementById('m-dps').textContent = fmt(st.dec_per_s, 1); document.getElementById('m-rtf').textContent = st.rtf === null ? '--' : fmt(st.rtf);
  document.getElementById('m-ckpt').textContent = st.ckpt + ' (' + st.ckpt_time + ')';
  Plotly.react('c-falls', perMapTraces(s, 'falls100'), darkLayout(legend), cfg);
  Plotly.react('c-speed', perMapTraces(s, 'speed'), darkLayout(legend), cfg);
  Plotly.react('c-rew', perMapTraces(s, 'rew'), darkLayout(legend), cfg);
  Plotly.react('c-ent', [{x:s.update, y:s.ent, type:'scatter', mode:'lines', line:{color:'#bc8cff', width:1.5}, name:'entropy'},
                         {x:s.update, y:s.dstd, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:1, dash:'dot'}, name:'dir std', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#58a6ff', gridcolor:'#21262d'}}, legend)), cfg);
  Plotly.react('c-kl', [{x:s.update, y:s.kl, type:'scatter', mode:'lines', line:{color:'#f0c040', width:1.5}, name:'KL'},
                        {x:s.update, y:s.clip, type:'scatter', mode:'lines', line:{color:'#7d8590', width:1, dash:'dot'}, name:'clip frac', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#7d8590', gridcolor:'#21262d', range:[0,1]}}, legend)), cfg);
  Plotly.react('c-loss', [{x:s.update, y:s.pl, type:'scatter', mode:'lines', line:{color:'#3fb950', width:1.5}, name:'policy loss'},
                          {x:s.update, y:s.vl, type:'scatter', mode:'lines', line:{color:'#f85149', width:1}, name:'value loss', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#f85149', gridcolor:'#21262d', type:'log'}}, legend)), cfg);
  Plotly.react('c-gn', [{x:s.update, y:s.gn, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:1.5}, name:'grad norm'}], darkLayout(), cfg);
  Plotly.react('c-wall', [{x:s.update, y:s.wall_s, type:'bar', marker:{color:'#7d8590'}, name:'wall s'}], darkLayout(), cfg);
  const tb = document.querySelector('#t-maps tbody'); tb.innerHTML = '';
  for (const [mp, v] of Object.entries(st.per_map)) {
    tb.innerHTML += `<tr><td>${mp.replace('_Hunt','')}</td><td class="num">${v.updates}</td><td class="num" style="color:#f0c040">${v.sgem ? v.sgem.toFixed(2) : '--'}</td><td class="num">${v.best_sgem ? v.best_sgem.toFixed(2) : '--'}</td><td class="num">${v.falls100.toFixed(2)}</td><td class="num" style="color:#f0c040">${v.best_falls100.toFixed(2)}</td><td class="num">${v.speed.toFixed(1)}</td><td class="num" style="color:#f0c040">${v.best_speed.toFixed(1)}</td></tr>`;
  }
  const te = document.querySelector('#t-events tbody'); te.innerHTML = '';
  st.events.slice().reverse().forEach(e => { te.innerHTML += `<tr><td class="num">upd ${e[0]}</td><td class="ev ${e[1]}">${e[2]}</td></tr>`; });
}
const source = new EventSource('/stream');
source.onmessage = (e) => { try { update(JSON.parse(e.data)); } catch (err) { console.error(err); } };
source.onopen = () => { const b = document.getElementById('live-badge'); b.textContent = 'LIVE'; b.className = 'badge badge-live'; };
source.onerror = () => { const b = document.getElementById('live-badge'); b.textContent = 'RECONNECTING...'; b.className = 'badge badge-reconnecting'; };
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    srv.daemon_threads = True
    print(f'navigator dashboard: http://localhost:{PORT}  (logs: {LOG_DIR})', flush=True)
    st = build_state()
    print(f'parsed {st["n_updates"]} updates across {len(st["maps"])} maps; latest checkpoint {st["ckpt"]}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
