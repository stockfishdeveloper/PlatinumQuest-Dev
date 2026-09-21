"""Live dashboard for the navigator trainer (waypoint task).

    python -m nav.dashboard_nav          then open http://localhost:8990

Reads every logs/nav/train_nav_*.log (all runs, resumed checkpoints included), parses
the NAV / SEG / MAP / ARRIVE / saved lines, and pushes the whole picture to the page
over server-sent events every few seconds. Same look and feel as dashboard.py, but the
numbers are the navigator's: arrival rate, falls per 100 u, speed, arrival radius,
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

NAV_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] NAV upd=(\d+)(?: map=(\S+))?(?: r=([\d.]+))? steps=([\d,]+) segs=(\d+) arrive=(\d+)% '
                    r'falls100=([\d.]+) speed=([\d.]+) rew=(-?[\d.]+) pl=(-?[\d.]+) vl=([\d.]+) ent=(-?[\d.]+) kl=(-?[\d.]+) '
                    r'clip=([\d.]+) gn=([\d.]+) ep=(\d+) dstd=([\d.]+) (?:flips=(\d+) )?upd_s=([\d.]+) wall_s=(\d+)')
RADIUS_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] ARRIVE radius tightened to ([\d.]+) u')
SAVED_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] saved (\S+)')
MAP_RE = re.compile(r'\[(\d\d:\d\d:\d\d)\] (?:MAP|mission) ([A-Za-z0-9_]+)')
RTF_RE = re.compile(r'rtf[= ]([\d.]+)')
DPS_RE = re.compile(r'dps=(\d+)')
RESUME_RE = re.compile(r'resumed .*: update (\d+)')


def parse_logs():
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'train_nav_*.log')))
    by_update = {}
    events = []            # (update_at_time, kind, text)
    last_update = 0
    counters = {'synthetic_falls': 0, 'respawn_misses': 0, 'teleport_ignored': 0, 'rounds': 0, 'restarts': 0}
    rtfs = []
    current_map = ''
    for f in files:
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines:
            m = NAV_RE.search(line)
            if m:
                upd = int(m.group(2))
                rec = {
                    'time': m.group(1), 'update': upd, 'map': m.group(3) or current_map or '?', 'r': float(m.group(4)) if m.group(4) else None,
                    'steps': int(m.group(5).replace(',', '')), 'segs': int(m.group(6)), 'arrive': float(m.group(7)),
                    'falls100': float(m.group(8)), 'speed': float(m.group(9)), 'rew': float(m.group(10)),
                    'pl': float(m.group(11)), 'vl': float(m.group(12)), 'ent': float(m.group(13)), 'kl': float(m.group(14)),
                    'clip': float(m.group(15)), 'gn': float(m.group(16)), 'ep': int(m.group(17)), 'dstd': float(m.group(18)),
                    'flips': int(m.group(19)) if m.group(19) else 0, 'upd_s': float(m.group(20)), 'wall_s': float(m.group(21)),
                }
                md = DPS_RE.search(line)
                rec['dps'] = float(md.group(1)) if md else None      # multi-instance trainer: decisions/s it measured itself
                by_update[upd] = rec
                last_update = max(last_update, upd)
                continue
            m = MAP_RE.search(line)
            if m:
                current_map = m.group(2)
                if ' MAP ' in line:
                    events.append((last_update, 'map', f'{m.group(1)} map -> {current_map}'))
                continue
            m = RADIUS_RE.search(line)
            if m:
                events.append((last_update, 'radius', f'{m.group(1)} arrival radius -> {m.group(2)} u'))
                continue
            m = SAVED_RE.search(line)
            if m:
                events.append((last_update, 'ckpt', f'{m.group(1)} {os.path.basename(m.group(2))}'))
                continue
            m = RESUME_RE.search(line)
            if m:
                counters['restarts'] += 1
                events.append((int(m.group(1)), 'resume', f'trainer (re)started at update {m.group(1)}'))
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
    return updates, events, counters, rtfs


def build_state():
    updates, events, counters, rtfs = parse_logs()
    maps = []
    for u in updates:
        if u['map'] not in maps:
            maps.append(u['map'])
    per_map = {}
    for mp in maps:
        us = [u for u in updates if u['map'] == mp]
        per_map[mp] = {
            'updates': len(us), 'last': us[-1]['update'], 'arrive': us[-1]['arrive'], 'best_arrive': max(u['arrive'] for u in us),
            'falls100': us[-1]['falls100'], 'best_falls100': min(u['falls100'] for u in us), 'speed': us[-1]['speed'],
            'best_speed': max(u['speed'] for u in us), 'r': us[-1]['r'],
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
        'series': {k: [u[k] for u in updates] for k in ('update', 'map', 'r', 'arrive', 'falls100', 'speed', 'rew', 'pl', 'vl',
                                                         'ent', 'kl', 'clip', 'gn', 'dstd', 'wall_s', 'steps', 'ep')},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/':
            body = HTML.encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
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
    <span>Arrive radius: <b id="m-r" style="color:#f0c040">--</b> u (final 0.65)</span>
    <span>Decisions/s: <b id="m-dps">--</b></span>
    <span>Game rtf: <b id="m-rtf">--</b>x</span>
    <span>Checkpoint: <b id="m-ckpt">--</b></span>
  </div>
</div>
<div id="gauges">
  <div class="gauge"><div class="label">Arrival % (this map, last 300 segs)</div><div class="value" id="g-arrive" style="color:#3fb950">--</div><div class="label">target &gt; 90 %</div></div>
  <div class="gauge"><div class="label">Falls / 100 u</div><div class="value" id="g-falls" style="color:#f85149">--</div><div class="label">target &lt; 0.5</div></div>
  <div class="gauge"><div class="label">Speed u/s</div><div class="value" id="g-speed">--</div><div class="label">target &ge; 7</div></div>
  <div class="gauge"><div class="label">Reward / segment</div><div class="value" id="g-rew">--</div></div>
  <div class="gauge"><div class="label">Entropy</div><div class="value" id="g-ent">--</div><div class="label" id="g-dstd">dir std --</div></div>
  <div class="gauge"><div class="label">KL / clip frac</div><div class="value" id="g-kl">--</div><div class="label">healthy 0.005 - 0.03</div></div>
  <div class="gauge"><div class="label">Grad norm</div><div class="value" id="g-gn">--</div></div>
  <div class="gauge"><div class="label">Wall s / update</div><div class="value" id="g-wall">--</div><div class="label">~45 s at 3x, one instance</div></div>
  <div class="gauge"><div class="label">Best arrival % (any map)</div><div class="value" id="g-best" style="color:#f0c040">--</div><div class="label" id="g-best-map"></div></div>
  <div class="gauge"><div class="label">Env warts (this run)</div><div class="value" id="g-warts" style="font-size:0.9rem">--</div><div class="label">synthetic falls / respawn misses / teleports ignored</div></div>
</div>
<div id="charts">
  <div class="chart-card"><div class="chart-title">Arrival % per update, one line per map (the milestone number)</div><div id="c-arrive" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Falls per 100 u per map</div><div id="c-falls" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Speed (u/s, on-floor travel) per map</div><div id="c-speed" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Arrival radius (u) &mdash; ratchets down toward the real gem hitbox 0.65</div><div id="c-radius" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Reward per segment (300-segment rolling)</div><div id="c-rew" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Entropy + direction std</div><div id="c-ent" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">KL per update + clip fraction</div><div id="c-kl" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Policy loss vs value loss</div><div id="c-loss" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Gradient norm</div><div id="c-gn" style="height:230px"></div></div>
  <div class="chart-card"><div class="chart-title">Wall seconds per update (throughput; spikes = stalls / map switches)</div><div id="c-wall" style="height:230px"></div></div>
</div>
<div id="tables">
  <div class="chart-card"><div class="chart-title">Per map (latest slot vs best)</div><table id="t-maps"><thead><tr><th>map</th><th>updates</th><th>arrive %</th><th>best</th><th>falls/100u</th><th>best</th><th>speed</th><th>best</th></tr></thead><tbody></tbody></table></div>
  <div class="chart-card"><div class="chart-title">Events (map switches, radius changes, checkpoints, restarts)</div><table id="t-events"><tbody></tbody></table></div>
</div>
<div class="hint">Reads logs/nav/train_nav_*.log every 3 s. A segment = one waypoint; arrive / fall / timeout are its outcomes. Falls per 100 u counts falls per distance travelled on the floor.</div>
<script>
const darkLayout = (extra) => Object.assign({
  paper_bgcolor:'#161b22', plot_bgcolor:'#0d1117', font:{color:'#e6edf3', size:10, family:'Consolas, monospace'},
  margin:{l:50, r:12, t:8, b:32}, showlegend:false,
  xaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false, title:'update'}, yaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false},
}, extra || {});
const cfg = {responsive:true, displayModeBar:false};
const palette = ['#3fb950','#58a6ff','#f0c040','#bc8cff','#f85149','#d29922','#39d3c3','#ff7b72'];
const legend = {showlegend:true, legend:{x:0, y:1.15, orientation:'h', font:{size:9}}};
['c-arrive','c-falls','c-speed','c-radius','c-rew','c-ent','c-kl','c-loss','c-gn','c-wall'].forEach(id => Plotly.newPlot(id, [], darkLayout(legend), cfg));

function perMapTraces(s, key, mode) {
  const maps = [...new Set(s.map)]; const traces = [];
  maps.forEach((mp, i) => {
    const x = [], y = [];
    s.update.forEach((u, k) => { if (s.map[k] === mp && s[key][k] !== null) { x.push(u); y.push(s[key][k]); } });
    traces.push({x, y, type:'scatter', mode: mode || 'lines+markers', marker:{size:3}, line:{width:1.5, color:palette[i % palette.length]}, name: mp.replace('_Hunt','')});
  });
  return traces;
}
function fmt(v, d) { return (v === null || v === undefined) ? '--' : Number(v).toFixed(d === undefined ? 2 : d); }

function update(st) {
  const s = st.series, L = st.latest;
  document.getElementById('trainer-badge').textContent = 'Trainer: ' + (st.trainer_alive ? 'running' : 'stopped');
  document.getElementById('trainer-badge').className = 'badge badge-game' + (st.trainer_alive ? ' connected' : '');
  if (L) {
    document.getElementById('m-update').textContent = L.update; document.getElementById('m-steps').textContent = L.steps.toLocaleString();
    document.getElementById('m-segs').textContent = L.segs; document.getElementById('m-map').textContent = L.map.replace('_Hunt','');
    document.getElementById('m-r').textContent = L.r === null ? '1.50' : fmt(L.r);
    document.getElementById('g-arrive').textContent = fmt(L.arrive, 0) + '%'; document.getElementById('g-falls').textContent = fmt(L.falls100);
    document.getElementById('g-speed').textContent = fmt(L.speed, 1); document.getElementById('g-rew').textContent = fmt(L.rew, 1);
    document.getElementById('g-ent').textContent = fmt(L.ent); document.getElementById('g-dstd').textContent = 'dir std ' + fmt(L.dstd);
    document.getElementById('g-kl').textContent = fmt(L.kl, 3) + ' / ' + fmt(L.clip); document.getElementById('g-gn').textContent = fmt(L.gn);
    document.getElementById('g-wall').textContent = fmt(L.wall_s, 0) + ' s';
  }
  document.getElementById('m-dps').textContent = fmt(st.dec_per_s, 1); document.getElementById('m-rtf').textContent = st.rtf === null ? '--' : fmt(st.rtf);
  document.getElementById('m-ckpt').textContent = st.ckpt + ' (' + st.ckpt_time + ')';
  const c = st.counters; document.getElementById('g-warts').textContent = c.synthetic_falls + ' / ' + c.respawn_misses + ' / ' + c.teleport_ignored;
  let best = -1, bestMap = '';
  for (const [mp, v] of Object.entries(st.per_map)) { if (v.best_arrive > best) { best = v.best_arrive; bestMap = mp; } }
  document.getElementById('g-best').textContent = best < 0 ? '--' : best.toFixed(0) + '%'; document.getElementById('g-best-map').textContent = bestMap.replace('_Hunt','');
  Plotly.react('c-arrive', perMapTraces(s, 'arrive'), darkLayout(Object.assign({yaxis:{range:[0,100], gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);
  Plotly.react('c-falls', perMapTraces(s, 'falls100'), darkLayout(legend), cfg);
  Plotly.react('c-speed', perMapTraces(s, 'speed'), darkLayout(legend), cfg);
  Plotly.react('c-radius', [{x:s.update, y:s.r.map(v => v === null ? 1.5 : v), type:'scatter', mode:'lines', line:{shape:'hv', color:'#f0c040', width:2}, name:'radius'},
                            {x:[s.update[0], s.update[s.update.length-1]], y:[0.65,0.65], type:'scatter', mode:'lines', line:{dash:'dash', color:'#7d8590', width:1}, name:'gem hitbox'}],
               darkLayout(Object.assign({yaxis:{range:[0,1.7], gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);
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
    tb.innerHTML += `<tr><td>${mp.replace('_Hunt','')}</td><td class="num">${v.updates}</td><td class="num">${v.arrive.toFixed(0)}%</td><td class="num" style="color:#f0c040">${v.best_arrive.toFixed(0)}%</td><td class="num">${v.falls100.toFixed(2)}</td><td class="num" style="color:#f0c040">${v.best_falls100.toFixed(2)}</td><td class="num">${v.speed.toFixed(1)}</td><td class="num" style="color:#f0c040">${v.best_speed.toFixed(1)}</td></tr>`;
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
