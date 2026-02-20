"""
Real-time PPO Training Dashboard
Runs as a background daemon thread, serves a web dashboard on port 8889.
Zero external dependencies — uses stdlib http.server + SSE (Server-Sent Events).
"""

import json
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

DASHBOARD_PORT = 8889
MAX_HISTORY = 10000  # Cap history arrays to bound memory (~1MB)


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress default access logs — would spam training console

    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/stream':
            self._serve_sse()
        elif self.path == '/history':
            self._serve_history()
        else:
            self.send_error(404)

    def _serve_html(self):
        content = DASHBOARD_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_history(self):
        dashboard = self.server.dashboard
        history = dashboard.get_history()
        data = json.dumps(history).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Send retry interval (browser will reconnect after 3s if connection drops)
        self.wfile.write(b'retry: 3000\n\n')
        self.wfile.flush()

        last_update = -1
        keepalive_counter = 0

        try:
            while True:
                dashboard = self.server.dashboard
                snap = dashboard.get_snapshot()

                if snap and snap['update'] != last_update:
                    data = json.dumps(snap)
                    self.wfile.write(f'data: {data}\n\n'.encode('utf-8'))
                    self.wfile.flush()
                    last_update = snap['update']
                    keepalive_counter = 0

                time.sleep(1)
                keepalive_counter += 1

                # Send keepalive comment every 15s to prevent WiFi NAT timeout
                if keepalive_counter >= 15:
                    self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()
                    keepalive_counter = 0

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # Client disconnected — EventSource will auto-reconnect


class DashboardServer:
    """Manages dashboard state and runs HTTP server in a background thread."""

    def __init__(self, ppo_server, host='0.0.0.0', port=DASHBOARD_PORT):
        self.ppo_server = ppo_server
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._snapshot = None
        self._history = {
            'updates': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'grad_norm': [],
            'avg_reward': [],
            'gems_per_hr': [],
            'rollout_gem_pts': [],
            'rollout_oob': [],
            'total_steps': [],
            'episodes': [],
            'pos_reward_pct': [],
            'dry_rollouts': [],
            'timestamps': [],
            'total_no_gem_steps': [],
            'action_pcts_history': [],  # list of 9-element lists
        }
        self._server = None
        self._thread = None

    def start(self):
        """Start HTTP server in daemon thread."""
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), DashboardHandler)
            self._server.dashboard = self  # Attach reference for handler access
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name='dashboard'
            )
            self._thread.start()
            self.ppo_server.log(f"Dashboard: http://0.0.0.0:{self.port} (accessible on WiFi)")
        except OSError as e:
            self.ppo_server.log(f"Dashboard: Failed to start on port {self.port}: {e} (training continues without dashboard)")

    def push_snapshot(self, stats, avg_reward):
        """Called from training thread after each PPO update. Must be fast."""
        s = self.ppo_server
        elapsed_hrs = (time.time() - s.run_start_time) / 3600
        gems_per_hr = s.total_gem_pts / max(elapsed_hrs, 1 / 3600)
        pos_pct = (s.rollout_positive / max(s.rollout_steps, 1)) * 100

        counts = s.rollout_action_counts
        counts = counts.tolist() if hasattr(counts, 'tolist') else list(counts)
        action_pcts = [
            round(c / max(s.rollout_steps, 1) * 100, 1)
            for c in counts
        ]

        snap = {
            'update': s.total_updates,
            'timestamp': time.time(),
            'elapsed_hrs': round(elapsed_hrs, 4),
            'total_steps': s.total_steps,
            'total_episodes': s.total_episodes,
            'total_gem_pts': s.total_gem_pts,
            'total_oob': s.total_oob,
            'total_no_gem_steps': s.total_no_gem_steps,
            'no_gem_events': s.no_gem_events,
            'dry_rollouts': s.dry_rollouts,
            'policy_loss': round(stats['policy_loss'], 6),
            'value_loss': round(stats['value_loss'], 6),
            'entropy': round(stats['entropy'], 4),
            'grad_norm': round(stats['grad_norm'], 4),
            'avg_reward_100ep': round(float(avg_reward), 2),
            'best_avg_reward': round(float(s.best_avg_reward), 2),
            'gems_per_hr': round(gems_per_hr, 2),
            'pos_reward_pct': round(pos_pct, 1),
            'rollout_gem_pts': s.rollout_gem_pts,
            'rollout_oob': s.rollout_oob,
            'rollout_steps': s.rollout_steps,
            'action_pcts': action_pcts,
            'action_names': ['Idle', 'Fwd', 'Back', 'Left', 'Right', 'FL', 'FR', 'BL', 'BR'],
            'recent_rewards': [round(float(r), 1) for r in list(s.episode_rewards)[-20:]],
            'recent_episode_gems': list(s.recent_episode_gems),
            'rollout_size': s.rollout_size,
            'batch_size': s.batch_size,
            'n_epochs': s.n_epochs,
            'gamma': s.gamma,
            'lam': s.lam,
            'reward_scale': s.reward_scale,
            'entropy_collapse': stats['entropy'] < 0.5,
            'entropy_low': stats['entropy'] < 1.0,
            'dry_warning': s.dry_rollouts >= 5,
            'game_connected': True,
        }

        with self._lock:
            self._snapshot = snap
            h = self._history
            h['updates'].append(snap['update'])
            h['policy_loss'].append(snap['policy_loss'])
            h['value_loss'].append(snap['value_loss'])
            h['entropy'].append(snap['entropy'])
            h['grad_norm'].append(snap['grad_norm'])
            h['avg_reward'].append(snap['avg_reward_100ep'])
            h['gems_per_hr'].append(snap['gems_per_hr'])
            h['rollout_gem_pts'].append(snap['rollout_gem_pts'])
            h['rollout_oob'].append(snap['rollout_oob'])
            h['total_steps'].append(snap['total_steps'])
            h['episodes'].append(snap['total_episodes'])
            h['pos_reward_pct'].append(snap['pos_reward_pct'])
            h['dry_rollouts'].append(snap['dry_rollouts'])
            h['timestamps'].append(snap['timestamp'])
            h['total_no_gem_steps'].append(snap['total_no_gem_steps'])
            h['action_pcts_history'].append(action_pcts)

            # Cap history to prevent unbounded memory growth
            if len(h['updates']) > MAX_HISTORY:
                for key in h:
                    h[key] = h[key][-MAX_HISTORY:]

    def get_snapshot(self):
        with self._lock:
            return self._snapshot

    def get_history(self):
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}


# =============================================================================
# Embedded Dashboard HTML
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPO Training Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #0d1117; --card: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #7d8590; --green: #3fb950;
  --yellow: #d29922; --red: #f85149; --gold: #f0c040;
  --blue: #58a6ff; --purple: #bc8cff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Consolas', 'SF Mono', 'Fira Code', monospace; }

/* Header */
#header {
  position: sticky; top: 0; z-index: 100;
  background: var(--card); border-bottom: 1px solid var(--border);
  padding: 10px 16px;
}
#header-top { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
#header h1 { font-size: 1rem; font-weight: 600; white-space: nowrap; }
.badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge-live { background: #238636; color: white; }
.badge-reconnecting { background: var(--yellow); color: black; }
.badge-game { background: var(--border); color: var(--muted); }
.badge-game.connected { background: #238636; color: white; }
#meta-row { display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.75rem; color: var(--muted); margin-top: 6px; }
#meta-row b { color: var(--text); }

/* Alert banners */
.alert { display: none; padding: 8px 16px; font-weight: 600; font-size: 0.8rem; text-align: center; }
.alert.visible { display: block; }
.alert-collapse { background: var(--red); color: white; animation: pulse 1s infinite alternate; }
.alert-dry { background: #7c5e00; color: var(--gold); }
@keyframes pulse { from { opacity: 1; } to { opacity: 0.7; } }

/* Gauges */
#gauges { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; padding: 10px 12px; }
@media (max-width: 900px) { #gauges { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 500px) { #gauges { grid-template-columns: repeat(2, 1fr); } }
.gauge {
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 8px; text-align: center;
}
.gauge .label { font-size: 0.65rem; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.gauge .value { font-size: 1.3rem; font-weight: 700; }

/* Chart grid */
#charts { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 0 12px 12px; }
@media (max-width: 900px) { #charts { grid-template-columns: 1fr; } }
.chart-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px; overflow: hidden;
}
.chart-card.full-width { grid-column: 1 / -1; }
.chart-title { font-size: 0.7rem; color: var(--muted); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

/* Config panel */
#config-section { padding: 0 12px 16px; }
#config-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
#config {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 12px; font-size: 0.75rem;
}
@media (max-width: 768px) { #config { grid-template-columns: repeat(2, 1fr); } }
#config b { color: var(--text); }
#config span { color: var(--muted); }
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <div id="header-top">
    <h1>PlatinumQuest PPO Training</h1>
    <span id="live-badge" class="badge badge-reconnecting">CONNECTING...</span>
    <span id="game-badge" class="badge badge-game">Game: --</span>
  </div>
  <div id="meta-row">
    <span>Update: <b id="m-update">--</b></span>
    <span>Steps: <b id="m-steps">--</b></span>
    <span>Episodes: <b id="m-eps">--</b></span>
    <span>Time: <b id="m-time">--</b></span>
    <span>Gems: <b id="m-gems">--</b>pts</span>
    <span>Gems/hr: <b id="m-gems-hr">--</b></span>
    <span>OOB: <b id="m-oob">--</b></span>
    <span>Best Avg: <b id="m-best">--</b></span>
  </div>
</div>

<!-- Alerts -->
<div id="alert-collapse" class="alert alert-collapse">ENTROPY COLLAPSE (&lt; 0.5) - Policy has collapsed onto a single action!</div>
<div id="alert-dry" class="alert alert-dry">DRY STREAK: <span id="dry-count">0</span> consecutive rollouts with zero gems collected</div>

<!-- Gauges -->
<div id="gauges">
  <div class="gauge"><div class="label">Avg Reward (100ep)</div><div class="value" id="g-avgrwd">--</div></div>
  <div class="gauge"><div class="label">Entropy</div><div class="value" id="g-entropy">--</div></div>
  <div class="gauge"><div class="label">Policy Loss</div><div class="value" id="g-ploss">--</div></div>
  <div class="gauge"><div class="label">Value Loss</div><div class="value" id="g-vloss">--</div></div>
  <div class="gauge"><div class="label">Grad Norm</div><div class="value" id="g-gradnorm">--</div></div>
  <div class="gauge"><div class="label">Dry Rollouts</div><div class="value" id="g-dry">--</div></div>
</div>

<!-- Charts -->
<div id="charts">
  <div class="chart-card"><div class="chart-title">Avg Reward (100-episode rolling)</div><div id="c-avgrwd" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">PPO Losses (Policy + Value)</div><div id="c-losses" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Entropy (exploration health)</div><div id="c-entropy" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Gradient Norm</div><div id="c-gradnorm" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Gems Per Hour</div><div id="c-gemshr" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Per-Rollout Gem Points</div><div id="c-rolloutgems" style="height:220px"></div></div>
  <div class="chart-card full-width"><div class="chart-title">Action Distribution (stacked %)</div><div id="c-actions" style="height:240px"></div></div>
  <div class="chart-card"><div class="chart-title">Recent Episode Rewards (last 20)</div><div id="c-eprewards" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Recent Episode Gem Points (last 20)</div><div id="c-epgems" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">OOB Events Per Rollout</div><div id="c-oob" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">No-Gem Step % (cumulative)</div><div id="c-nogem" style="height:220px"></div></div>
  <div class="chart-card full-width"><div class="chart-title">Training Throughput (steps/sec)</div><div id="c-throughput" style="height:200px"></div></div>
</div>

<!-- Config -->
<div id="config-section">
  <div id="config-label">Hyperparameters</div>
  <div id="config">
    <span>Rollout Size: <b id="cfg-rollout">--</b></span>
    <span>Batch Size: <b id="cfg-batch">--</b></span>
    <span>Epochs/Update: <b id="cfg-epochs">--</b></span>
    <span>Gamma: <b id="cfg-gamma">--</b></span>
    <span>Lambda: <b id="cfg-lam">--</b></span>
    <span>Reward Scale: <b id="cfg-rwdscale">--</b></span>
    <span>Actions: <b>9 (Categorical)</b></span>
    <span>Obs Dim: <b>61</b></span>
  </div>
</div>

<script>
// ============================================================
// Plotly layout helper
// ============================================================
const darkLayout = (extra) => Object.assign({
  paper_bgcolor: '#161b22',
  plot_bgcolor: '#0d1117',
  font: { color: '#e6edf3', size: 10, family: 'Consolas, monospace' },
  margin: { l: 50, r: 12, t: 8, b: 32 },
  showlegend: false,
  xaxis: { gridcolor: '#21262d', color: '#7d8590', zeroline: false },
  yaxis: { gridcolor: '#21262d', color: '#7d8590', zeroline: false },
}, extra || {});

const plotConfig = { responsive: true, displayModeBar: false };

// ============================================================
// Initialize all charts (empty)
// ============================================================

// 1. Avg Reward
Plotly.newPlot('c-avgrwd', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#3fb950', width: 2 }, name: 'Avg Reward' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f0c040', width: 1, dash: 'dash' }, name: 'Best' }
], darkLayout({ showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } } }), plotConfig);

// 2. Losses (dual Y-axis)
Plotly.newPlot('c-losses', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#58a6ff', width: 1.5 }, name: 'Policy Loss' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f85149', width: 1.5 }, name: 'Value Loss', yaxis: 'y2' }
], darkLayout({
  showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } },
  yaxis2: { overlaying: 'y', side: 'right', gridcolor: '#21262d', color: '#7d8590', type: 'log', zeroline: false }
}), plotConfig);

// 3. Entropy with thresholds
Plotly.newPlot('c-entropy', [
  { x: [], y: [], type: 'scatter', mode: 'lines', fill: 'tozeroy',
    line: { color: '#3fb950', width: 2 }, fillcolor: 'rgba(63,185,80,0.1)', name: 'Entropy' }
], darkLayout({
  yaxis: { range: [0, 2.4], gridcolor: '#21262d', color: '#7d8590', zeroline: false },
  shapes: [
    { type: 'line', y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: 'paper', line: { color: '#f85149', width: 1, dash: 'dash' } },
    { type: 'line', y0: 1.0, y1: 1.0, x0: 0, x1: 1, xref: 'paper', line: { color: '#d29922', width: 1, dash: 'dot' } },
    { type: 'line', y0: 2.197, y1: 2.197, x0: 0, x1: 1, xref: 'paper', line: { color: '#30363d', width: 1, dash: 'dot' } }
  ]
}), plotConfig);

// 4. Grad Norm
Plotly.newPlot('c-gradnorm', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#d29922', width: 1.5 } }
], darkLayout({
  shapes: [
    { type: 'line', y0: 1.0, y1: 1.0, x0: 0, x1: 1, xref: 'paper', line: { color: '#f85149', width: 1, dash: 'dash' } }
  ]
}), plotConfig);

// 5. Gems/hr
Plotly.newPlot('c-gemshr', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f0c040', width: 2 } }
], darkLayout(), plotConfig);

// 6. Rollout gems (bar)
Plotly.newPlot('c-rolloutgems', [
  { x: [], y: [], type: 'bar', marker: { color: '#f0c040' } }
], darkLayout({ bargap: 0.3 }), plotConfig);

// 7. Action distribution (stacked bar)
const ACTION_NAMES = ['Idle','Fwd','Back','Left','Right','FL','FR','BL','BR'];
const ACTION_COLORS = ['#6e7681','#3fb950','#f85149','#58a6ff','#d29922','#bc8cff','#39d353','#ff7b72','#79c0ff'];
Plotly.newPlot('c-actions',
  ACTION_NAMES.map((name, i) => ({
    x: [], y: [], type: 'bar', name: name,
    marker: { color: ACTION_COLORS[i] }
  })),
  darkLayout({
    barmode: 'stack', showlegend: true, bargap: 0.1,
    legend: { orientation: 'h', y: -0.12, font: { size: 9 } },
    yaxis: { range: [0, 100], gridcolor: '#21262d', color: '#7d8590', zeroline: false,
             title: { text: '%', font: { size: 9 } } }
  }), plotConfig
);

// 8. Episode rewards (last 20, color-coded bars)
Plotly.newPlot('c-eprewards', [
  { x: [], y: [], type: 'bar', marker: { color: [] } }
], darkLayout({ bargap: 0.2 }), plotConfig);

// 9. Episode gems (last 20)
Plotly.newPlot('c-epgems', [
  { x: [], y: [], type: 'bar', marker: { color: '#f0c040' } }
], darkLayout({ bargap: 0.2 }), plotConfig);

// 10. OOB per rollout
Plotly.newPlot('c-oob', [
  { x: [], y: [], type: 'bar', marker: { color: '#f85149' } }
], darkLayout({ bargap: 0.3 }), plotConfig);

// 11. No-gem %
Plotly.newPlot('c-nogem', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#d29922', width: 1.5 } }
], darkLayout(), plotConfig);

// 12. Throughput
Plotly.newPlot('c-throughput', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#79c0ff', width: 1.5 } }
], darkLayout(), plotConfig);


// ============================================================
// State
// ============================================================
let prevTimestamp = null;
let bestReward = -Infinity;

// ============================================================
// History hydration on page load
// ============================================================
async function loadHistory() {
  try {
    const resp = await fetch('/history');
    const h = await resp.json();
    if (!h.updates || h.updates.length === 0) return;

    const xs = h.updates;
    const n = xs.length;

    // Avg reward + best line
    bestReward = Math.max(...h.avg_reward);
    const bestLine = h.avg_reward.map(() => bestReward);
    Plotly.extendTraces('c-avgrwd', { x: [xs, xs], y: [h.avg_reward, bestLine] }, [0, 1]);

    // Losses
    Plotly.extendTraces('c-losses', { x: [xs, xs], y: [h.policy_loss, h.value_loss] }, [0, 1]);

    // Entropy
    Plotly.extendTraces('c-entropy', { x: [xs], y: [h.entropy] }, [0]);

    // Grad norm
    Plotly.extendTraces('c-gradnorm', { x: [xs], y: [h.grad_norm] }, [0]);

    // Gems/hr
    Plotly.extendTraces('c-gemshr', { x: [xs], y: [h.gems_per_hr] }, [0]);

    // Rollout gems
    Plotly.extendTraces('c-rolloutgems', { x: [xs], y: [h.rollout_gem_pts] }, [0]);

    // Action distribution
    if (h.action_pcts_history && h.action_pcts_history.length > 0) {
      const lastN = h.action_pcts_history.slice(-100);
      const actXs = xs.slice(-100);
      for (let i = 0; i < 9; i++) {
        Plotly.extendTraces('c-actions', { x: [actXs], y: [lastN.map(a => a[i])] }, [i]);
      }
    }

    // OOB
    Plotly.extendTraces('c-oob', { x: [xs], y: [h.rollout_oob] }, [0]);

    // No-gem %
    const nogemPct = h.total_no_gem_steps.map((ng, i) =>
      h.total_steps[i] > 0 ? (ng / h.total_steps[i] * 100) : 0
    );
    Plotly.extendTraces('c-nogem', { x: [xs], y: [nogemPct] }, [0]);

    // Throughput
    if (h.timestamps.length > 1) {
      const tpXs = xs.slice(1);
      const tpY = [];
      for (let i = 1; i < h.timestamps.length; i++) {
        const dt = h.timestamps[i] - h.timestamps[i - 1];
        tpY.push(dt > 0.001 ? 512 / dt : 0);
      }
      Plotly.extendTraces('c-throughput', { x: [tpXs], y: [tpY] }, [0]);
    }

    // Set prevTimestamp for live throughput calc
    if (h.timestamps.length > 0) {
      prevTimestamp = h.timestamps[h.timestamps.length - 1];
    }

  } catch (e) {
    console.log('History load failed (training just started?):', e);
  }
}

// ============================================================
// Live SSE update handler
// ============================================================
function updateDashboard(snap) {
  // === Header ===
  document.getElementById('m-update').textContent = snap.update;
  document.getElementById('m-steps').textContent = snap.total_steps.toLocaleString();
  document.getElementById('m-eps').textContent = snap.total_episodes;
  document.getElementById('m-time').textContent = snap.elapsed_hrs.toFixed(2) + 'h';
  document.getElementById('m-gems').textContent = snap.total_gem_pts;
  document.getElementById('m-gems-hr').textContent = snap.gems_per_hr.toFixed(1);
  document.getElementById('m-oob').textContent = snap.total_oob;
  document.getElementById('m-best').textContent = snap.best_avg_reward.toFixed(1);

  // === Badges ===
  const liveBadge = document.getElementById('live-badge');
  liveBadge.textContent = 'LIVE';
  liveBadge.className = 'badge badge-live';

  const gameBadge = document.getElementById('game-badge');
  gameBadge.textContent = snap.game_connected ? 'Game: CONNECTED' : 'Game: WAITING';
  gameBadge.className = 'badge badge-game' + (snap.game_connected ? ' connected' : '');

  // === Alerts ===
  document.getElementById('alert-collapse').classList.toggle('visible', snap.entropy_collapse);
  const dryAlert = document.getElementById('alert-dry');
  dryAlert.classList.toggle('visible', snap.dry_warning);
  if (snap.dry_warning) document.getElementById('dry-count').textContent = snap.dry_rollouts;

  // === Gauges ===
  document.getElementById('g-avgrwd').textContent = snap.avg_reward_100ep.toFixed(1);
  document.getElementById('g-avgrwd').style.color = snap.avg_reward_100ep > 0 ? '#3fb950' : snap.avg_reward_100ep < -20 ? '#f85149' : '#e6edf3';

  const entEl = document.getElementById('g-entropy');
  entEl.textContent = snap.entropy.toFixed(3);
  entEl.style.color = snap.entropy_collapse ? '#f85149' : snap.entropy_low ? '#d29922' : '#3fb950';

  document.getElementById('g-ploss').textContent = snap.policy_loss.toFixed(4);
  document.getElementById('g-vloss').textContent = snap.value_loss < 0.001 ? snap.value_loss.toExponential(2) : snap.value_loss.toFixed(4);
  document.getElementById('g-gradnorm').textContent = snap.grad_norm.toFixed(3);

  const dryEl = document.getElementById('g-dry');
  dryEl.textContent = snap.dry_rollouts;
  dryEl.style.color = snap.dry_warning ? '#f85149' : snap.dry_rollouts > 0 ? '#d29922' : '#3fb950';

  // === Config (once) ===
  document.getElementById('cfg-rollout').textContent = snap.rollout_size;
  document.getElementById('cfg-batch').textContent = snap.batch_size;
  document.getElementById('cfg-epochs').textContent = snap.n_epochs;
  document.getElementById('cfg-gamma').textContent = snap.gamma;
  document.getElementById('cfg-lam').textContent = snap.lam;
  document.getElementById('cfg-rwdscale').textContent = snap.reward_scale;

  // === Time-series charts (extend traces - O(1)) ===
  const x = snap.update;

  // Best reward tracking
  if (snap.best_avg_reward > bestReward) bestReward = snap.best_avg_reward;
  Plotly.extendTraces('c-avgrwd', { x: [[x], [x]], y: [[snap.avg_reward_100ep], [bestReward]] }, [0, 1]);
  Plotly.extendTraces('c-losses', { x: [[x], [x]], y: [[snap.policy_loss], [snap.value_loss]] }, [0, 1]);
  Plotly.extendTraces('c-entropy', { x: [[x]], y: [[snap.entropy]] }, [0]);
  Plotly.extendTraces('c-gradnorm', { x: [[x]], y: [[snap.grad_norm]] }, [0]);
  Plotly.extendTraces('c-gemshr', { x: [[x]], y: [[snap.gems_per_hr]] }, [0]);
  Plotly.extendTraces('c-rolloutgems', { x: [[x]], y: [[snap.rollout_gem_pts]] }, [0]);
  Plotly.extendTraces('c-oob', { x: [[x]], y: [[snap.rollout_oob]] }, [0]);

  // No-gem %
  const nogemPct = snap.total_steps > 0 ? (snap.total_no_gem_steps / snap.total_steps * 100) : 0;
  Plotly.extendTraces('c-nogem', { x: [[x]], y: [[nogemPct]] }, [0]);

  // Throughput
  if (prevTimestamp !== null) {
    const dt = snap.timestamp - prevTimestamp;
    const sps = dt > 0.001 ? snap.rollout_size / dt : 0;
    Plotly.extendTraces('c-throughput', { x: [[x]], y: [[sps]] }, [0]);
  }
  prevTimestamp = snap.timestamp;

  // Action distribution (extend stacked bar)
  for (let i = 0; i < 9; i++) {
    Plotly.extendTraces('c-actions', { x: [[x]], y: [[snap.action_pcts[i]]] }, [i]);
  }

  // === Snapshot charts (full redraw — small fixed-size arrays) ===

  // Episode rewards (last 20, color-coded)
  const rwds = snap.recent_rewards;
  if (rwds.length > 0) {
    const epIdxs = rwds.map((_, i) => i + 1);
    const colors = rwds.map(r => r > 100 ? '#3fb950' : r < -20 ? '#f85149' : '#7d8590');
    Plotly.react('c-eprewards',
      [{ x: epIdxs, y: rwds, type: 'bar', marker: { color: colors } }],
      darkLayout({ bargap: 0.2 }), plotConfig
    );
  }

  // Episode gem points (last 20)
  const gems = snap.recent_episode_gems;
  if (gems.length > 0) {
    const gemIdxs = gems.map((_, i) => i + 1);
    Plotly.react('c-epgems',
      [{ x: gemIdxs, y: gems, type: 'bar', marker: { color: '#f0c040' } }],
      darkLayout({ bargap: 0.2 }), plotConfig
    );
  }
}

// ============================================================
// Boot
// ============================================================
loadHistory().then(() => {
  const source = new EventSource('/stream');

  source.onmessage = (e) => {
    try { updateDashboard(JSON.parse(e.data)); }
    catch (err) { console.error('Dashboard update error:', err); }
  };

  source.onopen = () => {
    const b = document.getElementById('live-badge');
    b.textContent = 'LIVE';
    b.className = 'badge badge-live';
  };

  source.onerror = () => {
    const b = document.getElementById('live-badge');
    b.textContent = 'RECONNECTING...';
    b.className = 'badge badge-reconnecting';
  };
});
</script>
</body>
</html>"""
