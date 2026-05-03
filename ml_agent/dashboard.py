"""
Real-time PPO Training Dashboard
Runs as a background daemon thread, serves a web dashboard on port 8889.
Zero external dependencies — uses stdlib http.server + SSE (Server-Sent Events).
"""

import json
import math
import os
import sys
import time
import threading
import numpy as np
import torch
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

DASHBOARD_PORT = 8889
_anthropic_client = None
_anthropic_client_lock = threading.Lock()


def _get_anthropic_client():
    """Lazy-init the Anthropic client. Returns None if unavailable."""
    global _anthropic_client
    if not _ANTHROPIC_AVAILABLE:
        return None
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return None
    with _anthropic_client_lock:
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic()
        return _anthropic_client


# =============================================================================
# Jump probability heatmap probe
# =============================================================================
# Map geometry must stay in sync with FlatWithJump_Hunt.mcs
HEATMAP_PLATFORMS = [(10, 10), (-10, 10), (-10, -10), (10, -10)]
HEATMAP_GROUND_GEMS = [(17, 0), (-17, 0), (0, 17), (0, -17)]
HEATMAP_ALL_GEMS = (
    [(x, y, 1.6) for x, y in HEATMAP_PLATFORMS]
    + [(x, y, 0.6) for x, y in HEATMAP_GROUND_GEMS]
)
HEATMAP_GRID_MIN = -25.0
HEATMAP_GRID_MAX = 25.0
HEATMAP_GRID_STEP = 1.0


def _heatmap_normalize_obs(obs):
    """Mirror of PPOServer.normalize_obs."""
    obs[0:3] /= 100.0
    obs[3:6] /= 20.0
    for i in range(5):
        b = 6 + i*5
        if obs[b+4] < -500:
            obs[b:b+3] = 0.0
            obs[b+3] = 0.0
            obs[b+4] = 1.0
        else:
            if obs[b+4] > 0.01:
                obs[b:b+3] /= obs[b+4]
            else:
                obs[b:b+3] = 0.0
            obs[b+3] /= 5.0
            obs[b+4] /= 100.0
    obs[31] /= 300000.0
    obs[32] /= 300000.0
    obs[33] /= 100.0
    obs[34] /= 50.0
    return np.clip(obs, -2.0, 2.0)


def _heatmap_make_obs(marble_xy, marble_vel, gem_xyz):
    obs = np.zeros(59, dtype=np.float32)
    mx, my = marble_xy
    mz = 0.3
    vx, vy = marble_vel
    obs[0] = mx; obs[1] = my; obs[2] = mz
    obs[3] = vx; obs[4] = vy; obs[5] = 0
    gx, gy, gz = gem_xyz
    obs[6] = gx - mx; obs[7] = gy - my; obs[8] = gz - mz
    obs[9] = 1
    obs[10] = math.sqrt((gx-mx)**2 + (gy-my)**2 + (gz-mz)**2)
    for i in range(1, 5):
        obs[6 + i*5 + 4] = -999
    obs[31] = 150000; obs[32] = 150000; obs[33] = 30; obs[34] = 3
    # IMPORTANT: frame history (obs[35:59]) must be in NORMALIZED scale because
    # _heatmap_normalize_obs only divides obs[0:35]. During real training, frame
    # history is built FROM normalize_obs's already-normalized output, so the
    # historical pos/vel values are already pre-scaled. We mirror that here:
    # divide pos by 100 and vel by 20 before storing.
    for i in range(4):
        obs[35 + i*6 + 0] = mx / 100.0
        obs[35 + i*6 + 1] = my / 100.0
        obs[35 + i*6 + 2] = mz / 100.0
        obs[35 + i*6 + 3] = vx / 20.0
        obs[35 + i*6 + 4] = vy / 20.0
    return _heatmap_normalize_obs(obs)


def compute_heatmap(actor):
    """Compute a 51x51 jump-probability grid in one batched forward pass.

    Returns (grid_2d_floats, stats_dict). Snapshots the actor weights into
    a fresh model so the probe doesn't race with concurrent training updates.
    """
    # Snapshot weights (cheap memory copy) and load into a fresh model
    sd = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    from train_ppo import Actor
    obs_dim = sd['features.0.weight'].shape[1]
    model = Actor(obs_dim=obs_dim)
    model.load_state_dict(sd)
    model.eval()

    xs = np.arange(HEATMAP_GRID_MIN, HEATMAP_GRID_MAX + HEATMAP_GRID_STEP, HEATMAP_GRID_STEP)
    ys = np.arange(HEATMAP_GRID_MIN, HEATMAP_GRID_MAX + HEATMAP_GRID_STEP, HEATMAP_GRID_STEP)
    n_cells = len(xs) * len(ys)
    n_gems = len(HEATMAP_ALL_GEMS)

    # Build all observations at once
    all_obs = np.zeros((n_cells * n_gems, obs_dim), dtype=np.float32)
    idx = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for gem in HEATMAP_ALL_GEMS:
                gx, gy, _gz = gem
                dx = gx - x
                dy = gy - y
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0.01:
                    vx = (dx / d) * 6.0
                    vy = (dy / d) * 6.0
                else:
                    vx, vy = 0.0, 0.0
                all_obs[idx] = _heatmap_make_obs((float(x), float(y)), (vx, vy), gem)
                idx += 1

    # One batched forward pass — fast on small models
    t = torch.from_numpy(all_obs)
    with torch.no_grad():
        _, _, jump_logits, _ = model(t)
        jump_probs = torch.sigmoid(jump_logits).cpu().numpy().squeeze(-1)

    # Average across the 8 gem scenarios per cell, reshape to grid
    jump_probs = jump_probs.reshape(n_cells, n_gems).mean(axis=1)
    grid = jump_probs.reshape(len(ys), len(xs))

    # Platform-cell mask (cells within 1 unit of any platform XY)
    plat_mask = np.zeros_like(grid, dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for px, py in HEATMAP_PLATFORMS:
                if abs(x - px) <= 1.0 and abs(y - py) <= 1.0:
                    plat_mask[j, i] = True
                    break
    plat_avg = float(grid[plat_mask].mean()) if plat_mask.any() else 0.0
    nonplat_avg = float(grid[~plat_mask].mean())

    stats = {
        'min': float(grid.min()),
        'max': float(grid.max()),
        'mean': float(grid.mean()),
        'std': float(grid.std()),
        'platform_avg': plat_avg,
        'nonplatform_avg': nonplat_avg,
        'gap': plat_avg - nonplat_avg,
    }
    return grid, stats


class QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses connection-abort tracebacks."""
    def handle_error(self, request, client_address):
        exc_type = sys.exc_info()[0]
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Browser disconnected — not an error
        else:
            super().handle_error(request, client_address)


MAX_HISTORY = 10000  # Cap history arrays to bound memory (~1MB)


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress default access logs — would spam training console

    def do_GET(self):
        try:
            if self.path == '/':
                self._serve_html()
            elif self.path == '/stream':
                self._serve_sse()
            elif self.path == '/history':
                self._serve_history()
            elif self.path == '/heatmap':
                self._serve_heatmap()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # Client disconnected

    def do_POST(self):
        try:
            if self.path == '/api/chat':
                self._serve_chat()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def _serve_html(self):
        content = DASHBOARD_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
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

    def _serve_heatmap(self):
        dashboard = self.server.dashboard
        payload = dashboard.get_heatmap_payload()
        data = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
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

    def _sse_send(self, payload):
        """Write a single SSE event with a JSON payload."""
        self.wfile.write(f'data: {json.dumps(payload)}\n\n'.encode('utf-8'))
        self.wfile.flush()

    def _serve_chat(self):
        # Read request body
        try:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length) if length > 0 else b''
            body = json.loads(raw.decode('utf-8')) if raw else {}
            conversation = body.get('messages', [])
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, 'Invalid JSON')
            return

        if not isinstance(conversation, list) or not conversation:
            self.send_error(400, 'messages must be a non-empty list')
            return

        client = _get_anthropic_client()

        # Open SSE response
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if client is None:
            if not _ANTHROPIC_AVAILABLE:
                err = 'Anthropic SDK not installed. Run: pip install anthropic'
            else:
                err = 'ANTHROPIC_API_KEY environment variable is not set.'
            self._sse_send({'type': 'error', 'text': err})
            self._sse_send({'type': 'done'})
            return

        # Gather live training context
        dashboard = self.server.dashboard
        snap = dashboard.get_snapshot() or {}
        history = dashboard.get_history()

        # Build the per-turn context block appended to the user message.
        context_json = json.dumps({
            'current_snapshot': snap,
            'history': history,
        }, default=str)

        base_prompt = (
            "You are embedded in a live PPO reinforcement-learning training dashboard "
            "for a Marble Blast Platinum agent. On every user turn you receive a JSON "
            "blob with the current snapshot and full history arrays under a "
            "<training_state> tag. Use it to answer questions about training health, "
            "diagnose issues, and suggest concrete next steps. Be concise and specific — "
            "reference actual numbers from the data. Do not fabricate metrics. If you need "
            "a value that isn't present, say so."
        )

        # Load project context from CHAT_CONTEXT.md if present, so the assistant
        # knows what stage of training we're in, the reward structure, past bugs
        # already fixed, and what each dashboard metric means. Reread each turn
        # so user edits take effect without restarting training.
        context_md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CHAT_CONTEXT.md')
        project_context = ''
        try:
            with open(context_md_path, 'r', encoding='utf-8') as f:
                project_context = f.read().strip()
        except (OSError, UnicodeDecodeError):
            project_context = ''

        if project_context:
            system_prompt = (
                base_prompt
                + "\n\n<project_context>\n"
                + project_context
                + "\n</project_context>"
            )
        else:
            system_prompt = base_prompt

        # Inject the live state into the latest user message without polluting prior turns
        messages = [dict(m) for m in conversation]
        if messages and messages[-1].get('role') == 'user':
            last_content = messages[-1].get('content', '')
            if not isinstance(last_content, str):
                last_content = str(last_content)
            messages[-1]['content'] = (
                f"<training_state>\n{context_json}\n</training_state>\n\n{last_content}"
            )

        try:
            with client.messages.stream(
                model='claude-opus-4-7',
                max_tokens=8000,
                thinking={'type': 'adaptive'},
                system=system_prompt,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == 'content_block_delta' and getattr(event.delta, 'type', None) == 'text_delta':
                        self._sse_send({'type': 'text', 'text': event.delta.text})
                final = stream.get_final_message()
            usage = getattr(final, 'usage', None)
            if usage:
                self._sse_send({
                    'type': 'usage',
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                })
            self._sse_send({'type': 'done'})
        except anthropic.APIStatusError as e:
            self._sse_send({'type': 'error', 'text': f'Claude API error ({e.status_code}): {e.message}'})
            self._sse_send({'type': 'done'})
        except Exception as e:
            self._sse_send({'type': 'error', 'text': f'{type(e).__name__}: {e}'})
            self._sse_send({'type': 'done'})


class DashboardServer:
    """Manages dashboard state and runs HTTP server in a background thread."""

    def __init__(self, ppo_server, host='0.0.0.0', port=DASHBOARD_PORT):
        self.ppo_server = ppo_server
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._snapshot = None
        self._best_gems_hr = 0.0
        self._kl_stop_window = []   # Rolling 100-update window for KL-stop %
        self._history = {
            'updates': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'grad_norm': [],
            'critic_grad_norm': [],
            'kl': [],
            'kl_stop_pct': [],
            'avg_reward': [],
            'gems_per_hr': [],
            'best_gems_hr': [],
            'rollout_gem_pts': [],
            'rollout_oob': [],
            'total_steps': [],
            'episodes': [],
            'pos_reward_pct': [],
            'dry_rollouts': [],
            'timestamps': [],
            'total_no_gem_steps': [],
            'avg_gap_penalty': [],
            'near_misses': [],
            'dwell_steps': [],
            'jump_rate': [],
            'brake_rate': [],
            'avg_steps_per_gem': [],
            'mean_angle': [],
            'policy_std': [],
            'throttle_mean': [],
            'throttle_min': [],
            'throttle_max': [],
            'throttle_std': [],
        }
        self._server = None
        self._thread = None

        # Heatmap state — updated in a background thread on game end so the
        # main training loop never blocks on it. The probe is one batched
        # forward pass through a snapshot of the actor weights, takes ~1s.
        self._heatmap_lock = threading.Lock()
        self._heatmap_grid = None         # 2D list of floats [0,1], shape (rows, cols)
        self._heatmap_stats = None        # dict with min/max/mean/std/platform_avg/etc.
        self._heatmap_version = 0         # increments each successful refresh
        self._heatmap_label = ''          # checkpoint label or "live"
        self._heatmap_thread = None
        self._heatmap_thread_lock = threading.Lock()  # prevents two probes at once

    def start(self):
        """Start HTTP server in daemon thread."""
        try:
            self._server = QuietHTTPServer((self.host, self.port), DashboardHandler)
            self._server.dashboard = self  # Attach reference for handler access
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name='dashboard'
            )
            self._thread.start()
            self.ppo_server.log(f"Dashboard: http://0.0.0.0:{self.port} (accessible on WiFi)")
            # Kick off an initial heatmap probe so the dashboard has something
            # to render before the first game finishes.
            self.trigger_heatmap_update()
        except OSError as e:
            self.ppo_server.log(f"Dashboard: Failed to start on port {self.port}: {e} (training continues without dashboard)")

    def push_snapshot(self, stats, avg_reward):
        """Called from training thread after each PPO update. Must be fast."""
        s = self.ppo_server
        elapsed_hrs = (time.time() - s.run_start_time) / 3600
        gems_per_hr = s.total_gem_pts / max(elapsed_hrs, 1 / 3600)
        if gems_per_hr > self._best_gems_hr and s.total_updates > 10:
            self._best_gems_hr = gems_per_hr
        pos_pct = (s.rollout_positive / max(s.rollout_steps, 1)) * 100

        # Continuous action stats
        import math
        recent = list(s.recent_actions)[-min(200, len(s.recent_actions)):]
        if recent:
            angles_deg = [((a * 180 / math.pi) % 360) for a in recent]
            mean_deg = sum(angles_deg) / len(angles_deg)
        else:
            mean_deg = 0.0
        log_std = torch.clamp(s.actor.log_std, s.actor.LOG_STD_MIN, s.actor.LOG_STD_MAX)
        policy_std_deg = log_std.exp().item() * 180 / math.pi

        # Throttle stats
        recent_thr = list(s.recent_throttles)[-min(200, len(s.recent_throttles)):]
        if recent_thr:
            throttle_mean = sum(recent_thr) / len(recent_thr)
            throttle_min = min(recent_thr)
            throttle_max = max(recent_thr)
        else:
            throttle_mean = throttle_min = throttle_max = 0.0
        thr_log_std = torch.clamp(s.actor.throttle_log_std,
                                  s.actor.THROTTLE_LOG_STD_MIN,
                                  s.actor.THROTTLE_LOG_STD_MAX)
        throttle_std = thr_log_std.exp().item()

        avg_gap_penalty = float(np.mean(s.recent_avg_gap_penalty)) if s.recent_avg_gap_penalty else 0
        avg_near_misses = float(np.mean(s.recent_near_misses)) if s.recent_near_misses else 0
        avg_dwell_steps = float(np.mean(s.recent_dwell_steps)) if s.recent_dwell_steps else 0
        avg_jump_rate = float(np.mean(s.recent_jump_rate)) if s.recent_jump_rate else 0
        avg_brake_rate = float(np.mean(s.recent_brake_rate)) if s.recent_brake_rate else 0
        avg_steps_per_gem = float(np.mean(s.recent_avg_steps_per_gem)) if s.recent_avg_steps_per_gem else 0

        # Rolling KL-stop percentage (last 100 updates)
        self._kl_stop_window.append(1 if stats.get('kl_early_stopped') else 0)
        if len(self._kl_stop_window) > 100:
            self._kl_stop_window = self._kl_stop_window[-100:]
        kl_stop_pct = round(100 * sum(self._kl_stop_window) / len(self._kl_stop_window), 1)

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
            'critic_grad_norm': round(stats.get('critic_grad_norm', 0.0), 4),
            'kl': round(stats.get('max_kl', 0.0), 4),
            'kl_stop_pct': kl_stop_pct,
            'avg_reward_100ep': round(float(avg_reward), 2) if not (avg_reward != avg_reward) else 0.0,
            'best_avg_reward': round(float(s.best_avg_reward), 2) if s.best_avg_reward > -1e9 else 0.0,
            'gems_per_hr': round(gems_per_hr, 2),
            'best_gems_hr': round(self._best_gems_hr, 2),
            'pos_reward_pct': round(pos_pct, 1),
            'rollout_gem_pts': s.rollout_gem_pts,
            'rollout_oob': s.rollout_oob,
            'rollout_steps': s.rollout_steps,
            'mean_angle': round(mean_deg, 1),
            'policy_std': round(policy_std_deg, 1),
            'recent_rewards': [round(float(r), 1) for r in list(s.episode_rewards)[-20:]],
            'recent_episode_gems': list(s.recent_episode_gems),
            'recent_game_gems': list(s.recent_game_gems),
            'best_game_gems': s.best_game_gems,
            'avg_gap_penalty': round(avg_gap_penalty, 2),
            'near_misses': round(avg_near_misses, 1),
            'dwell_steps': round(avg_dwell_steps, 0),
            'jump_rate': round(avg_jump_rate, 2),
            'brake_rate': round(avg_brake_rate, 2),
            'avg_steps_per_gem': round(avg_steps_per_gem, 1),
            'rollout_size': s.rollout_size,
            'batch_size': s.batch_size,
            'n_epochs': s.n_epochs,
            'gamma': s.gamma,
            'lam': s.lam,
            'reward_scale': s.reward_scale,
            'throttle_mean': round(throttle_mean, 4),
            'throttle_min': round(throttle_min, 4),
            'throttle_max': round(throttle_max, 4),
            'throttle_std': round(throttle_std, 4),
            'entropy_collapse': stats['entropy'] < -0.5,
            'entropy_low': stats['entropy'] < 0.3,
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
            h['critic_grad_norm'].append(snap['critic_grad_norm'])
            h['kl'].append(snap['kl'])
            h['kl_stop_pct'].append(snap['kl_stop_pct'])
            h['avg_reward'].append(snap['avg_reward_100ep'])
            h['gems_per_hr'].append(snap['gems_per_hr'])
            h['best_gems_hr'].append(snap['best_gems_hr'])
            h['rollout_gem_pts'].append(snap['rollout_gem_pts'])
            h['rollout_oob'].append(snap['rollout_oob'])
            h['total_steps'].append(snap['total_steps'])
            h['episodes'].append(snap['total_episodes'])
            h['pos_reward_pct'].append(snap['pos_reward_pct'])
            h['dry_rollouts'].append(snap['dry_rollouts'])
            h['timestamps'].append(snap['timestamp'])
            h['total_no_gem_steps'].append(snap['total_no_gem_steps'])
            h['avg_gap_penalty'].append(snap['avg_gap_penalty'])
            h['near_misses'].append(snap['near_misses'])
            h['dwell_steps'].append(snap['dwell_steps'])
            h['jump_rate'].append(snap['jump_rate'])
            h['brake_rate'].append(snap['brake_rate'])
            h['avg_steps_per_gem'].append(snap['avg_steps_per_gem'])
            h['mean_angle'].append(snap['mean_angle'])
            h['policy_std'].append(snap['policy_std'])
            h['throttle_mean'].append(snap['throttle_mean'])
            h['throttle_min'].append(snap['throttle_min'])
            h['throttle_max'].append(snap['throttle_max'])
            h['throttle_std'].append(snap['throttle_std'])

            # Cap history to prevent unbounded memory growth
            if len(h['updates']) > MAX_HISTORY:
                for key in h:
                    h[key] = h[key][-MAX_HISTORY:]

    def get_snapshot(self):
        with self._lock:
            snap = None if self._snapshot is None else dict(self._snapshot)
            if snap is not None:
                # Tag the snapshot with the current heatmap version so the browser
                # knows when to refetch. Cheap (one int) and avoids polling.
                with self._heatmap_lock:
                    snap['heatmap_version'] = self._heatmap_version
            return snap

    def get_history(self):
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}

    def get_heatmap_payload(self):
        """Return latest heatmap data + map geometry for the browser to render."""
        with self._heatmap_lock:
            return {
                'version': self._heatmap_version,
                'label': self._heatmap_label,
                'grid': self._heatmap_grid,        # None until first probe completes
                'stats': self._heatmap_stats,
                'platforms': HEATMAP_PLATFORMS,
                'ground_gems': HEATMAP_GROUND_GEMS,
                'grid_min': HEATMAP_GRID_MIN,
                'grid_max': HEATMAP_GRID_MAX,
                'grid_step': HEATMAP_GRID_STEP,
            }

    def trigger_heatmap_update(self):
        """Non-blocking: kick off a background heatmap probe.
        Skips if a probe is already in flight (avoids piling up threads)."""
        with self._heatmap_thread_lock:
            if self._heatmap_thread is not None and self._heatmap_thread.is_alive():
                return  # already computing
            self._heatmap_thread = threading.Thread(
                target=self._heatmap_worker,
                daemon=True,
                name='heatmap_probe',
            )
            self._heatmap_thread.start()

    def _heatmap_worker(self):
        """Run the probe on a snapshot of the actor weights. Runs in a daemon
        thread, never blocks the training loop."""
        try:
            actor = self.ppo_server.actor
            grid, stats = compute_heatmap(actor)
            label = f'update_{self.ppo_server.total_updates}'
            with self._heatmap_lock:
                # Convert to plain Python floats for JSON serialization
                self._heatmap_grid = [[float(v) for v in row] for row in grid]
                self._heatmap_stats = stats
                self._heatmap_label = label
                self._heatmap_version += 1
        except Exception as e:
            try:
                self.ppo_server.log(f'Heatmap probe error: {e}')
            except Exception:
                pass


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
#gauges { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; padding: 10px 12px; }
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

/* Heatmap section */
#heatmap-section { padding: 0 12px 24px; }
#heatmap-label {
  font-size: 0.65rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
}
#heatmap-label .hint { text-transform: none; letter-spacing: 0; color: var(--muted); font-size: 0.7rem; }
#heatmap-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: 12px; display: flex; flex-direction: column; align-items: center;
}
#heatmap-stats {
  font-size: 0.78rem; color: var(--muted); margin-bottom: 8px; text-align: center;
}
#heatmap-stats span { display: inline-block; margin: 0 10px; }
#heatmap-stats b { color: var(--text); }
#heatmap-canvas-wrap {
  position: relative;
  border: 1px solid var(--border);
  background: #000;
  max-width: 100%;
  /* Cap visual size on big screens, but shrink on small ones via the children */
  width: 612px;
  aspect-ratio: 1 / 1;
}
#heatmap-canvas {
  display: block;
  width: 100%;
  height: 100%;
  /* Internal pixel grid stays at the canvas resolution; CSS scales the display */
  image-rendering: pixelated;
}
#heatmap-overlay {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
#heatmap-legend {
  display: flex; align-items: center; gap: 6px; margin-top: 8px;
  font-size: 0.7rem; color: var(--muted); flex-wrap: wrap; justify-content: center;
}
#heatmap-legend .gradient-bar {
  display: inline-block; width: 200px; height: 14px;
  background: linear-gradient(to right, rgb(0,0,0), rgb(64,64,64), rgb(128,128,128), rgb(192,192,192), rgb(255,255,255));
  border: 1px solid var(--border); vertical-align: middle;
}
#heatmap-legend .lm { display: inline-flex; align-items: center; gap: 4px; margin-left: 14px; }
#heatmap-empty {
  width: 612px; height: 612px; display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 0.85rem;
}

/* Chat panel (bottom of page) */
#chat-section { padding: 0 12px 24px; }
#chat-label {
  font-size: 0.65rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
}
#chat-label .hint { text-transform: none; letter-spacing: 0; color: var(--muted); font-size: 0.7rem; }
#chat-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  display: flex; flex-direction: column; height: 420px;
}
#chat-log {
  flex: 1; overflow-y: auto; padding: 12px 14px;
  font-size: 0.82rem; line-height: 1.5;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}
#chat-log::-webkit-scrollbar { width: 8px; }
#chat-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
.chat-turn { margin-bottom: 14px; }
.chat-turn:last-child { margin-bottom: 0; }
.chat-role {
  font-size: 0.65rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600;
}
.chat-turn.user .chat-role { color: var(--blue); }
.chat-turn.assistant .chat-role { color: var(--purple); }
.chat-turn.error .chat-role { color: var(--red); }
.chat-body { white-space: pre-wrap; word-wrap: break-word; }
.chat-turn.assistant .chat-body { color: var(--text); }
.chat-turn.user .chat-body { color: var(--text); }
.chat-turn.error .chat-body { color: var(--red); }
.chat-cursor {
  display: inline-block; width: 7px; height: 0.95em;
  background: var(--green); margin-left: 2px; vertical-align: text-bottom;
  animation: cursorBlink 1s steps(2) infinite;
}
@keyframes cursorBlink { 50% { opacity: 0; } }

#chat-input-row {
  display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--border);
}
#chat-input {
  flex: 1; background: var(--bg); border: 1px solid var(--border);
  color: var(--text); border-radius: 4px; padding: 8px 10px;
  font-family: inherit; font-size: 0.85rem; resize: none; min-height: 38px; max-height: 140px;
  outline: none;
}
#chat-input:focus { border-color: var(--blue); }
#chat-input:disabled { opacity: 0.5; }
#chat-send, #chat-clear {
  background: var(--border); border: 1px solid var(--border); color: var(--text);
  border-radius: 4px; padding: 0 14px; font-family: inherit; font-size: 0.8rem;
  font-weight: 600; cursor: pointer; white-space: nowrap;
}
#chat-send { background: #1f6feb; border-color: #1f6feb; color: white; }
#chat-send:hover:not(:disabled) { background: #388bfd; }
#chat-send:disabled { opacity: 0.5; cursor: not-allowed; }
#chat-clear:hover { background: var(--card); }
.chat-empty { color: var(--muted); font-style: italic; text-align: center; padding: 40px 20px; }
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
    <span>Best Gems/hr: <b id="m-best-gems-hr" style="color:#f0c040">--</b></span>
    <span>Best Gems/Game: <b id="m-best-game-gems" style="color:#3fb950">--</b></span>
    <span>OOB: <b id="m-oob">--</b></span>
    <span>Best Avg: <b id="m-best">--</b></span>
  </div>
</div>

<!-- Alerts -->
<div id="alert-collapse" class="alert alert-collapse">ENTROPY COLLAPSE (&lt; -0.5) - Policy std has collapsed to near-zero!</div>
<div id="alert-dry" class="alert alert-dry">DRY STREAK: <span id="dry-count">0</span> consecutive rollouts with zero gems collected</div>

<!-- Gauges -->
<div id="gauges">
  <div class="gauge"><div class="label">Avg Reward (100ep)</div><div class="value" id="g-avgrwd">--</div></div>
  <div class="gauge"><div class="label">Gems/hr</div><div class="value" id="g-gemshr" style="color:#f0c040">--</div></div>
  <div class="gauge"><div class="label">Avg Gems/Game</div><div class="value" id="g-gems-game" style="color:#3fb950">--</div></div>
  <div class="gauge"><div class="label">Entropy</div><div class="value" id="g-entropy">--</div></div>
  <div class="gauge"><div class="label">KL-Stop %</div><div class="value" id="g-klstop">--</div></div>
  <div class="gauge"><div class="label">Grad Norm</div><div class="value" id="g-gradnorm">--</div></div>
  <div class="gauge"><div class="label">Last Game Gems</div><div class="value" id="g-lastgems" style="color:#f0c040">--</div></div>
  <div class="gauge"><div class="label">Dry Rollouts</div><div class="value" id="g-dry">--</div></div>
</div>

<!-- Charts -->
<div id="charts">
  <div class="chart-card"><div class="chart-title">Avg Reward (100-episode rolling)</div><div id="c-avgrwd" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Gems Per Game (last 100 games) + rolling avg</div><div id="c-epgems" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Gems Per Hour</div><div id="c-gemshr" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">KL Divergence + KL-Stop % (last 100 updates)</div><div id="c-kl" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Gradient Norm (actor + critic)</div><div id="c-gradnorm" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Actor Loss vs Value Loss</div><div id="c-losses" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Entropy (exploration health)</div><div id="c-entropy" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Policy Std Dev (degrees)</div><div id="c-policystd" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">OOB Events Per Rollout</div><div id="c-oob" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Avg Gap Penalty Per Gem (lower = faster pickups)</div><div id="c-gappen" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Jump Rate % Per Game</div><div id="c-jumprate" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Brake Rate % Per Game (% of steps with brake=1)</div><div id="c-brakerate" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Avg Steps Between Gem Pickups</div><div id="c-stepspergem" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Throttle (mean + min/max range)</div><div id="c-throttle" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Laziness (Reward / Gems per Hour)</div><div id="c-laziness" style="height:220px"></div></div>
  <div class="chart-card"><div class="chart-title">Training Throughput (steps/sec)</div><div id="c-throughput" style="height:220px"></div></div>
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
    <span>Actions: <b>Continuous (angle)</b></span>
    <span>Obs Dim: <b>61</b></span>
  </div>
</div>

<!-- Jump probability heatmap -->
<div id="heatmap-section">
  <div id="heatmap-label">
    <span>Jump Probability Heatmap</span>
    <span class="hint" id="heatmap-status">(probing&hellip;)</span>
  </div>
  <div id="heatmap-card">
    <div id="heatmap-stats"></div>
    <div id="heatmap-canvas-wrap">
      <canvas id="heatmap-canvas" width="612" height="612"></canvas>
      <canvas id="heatmap-overlay" width="612" height="612"></canvas>
    </div>
    <div id="heatmap-legend">
      <span>0%</span>
      <span class="gradient-bar"></span>
      <span>100%</span>
      <span class="lm"><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" stroke="#58a6ff" stroke-width="2" fill="none"/></svg> platform</span>
      <span class="lm"><svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0883e"/></svg> elevated gem</span>
      <span class="lm"><svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0c040"/></svg> ground gem</span>
    </div>
  </div>
</div>

<!-- Chat panel (Claude) -->
<div id="chat-section">
  <div id="chat-label">
    <span>Ask Claude</span>
    <span class="hint">(Opus 4.7 &mdash; has live access to all dashboard data)</span>
  </div>
  <div id="chat-card">
    <div id="chat-log">
      <div class="chat-empty">Ask anything about the current training run &mdash; metrics, trends, diagnostics, next steps.</div>
    </div>
    <div id="chat-input-row">
      <textarea id="chat-input" placeholder="e.g. Is entropy stable? What should I try next?" rows="1"></textarea>
      <button id="chat-send">Send</button>
      <button id="chat-clear" title="Clear conversation">Clear</button>
    </div>
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

// 2. Losses — actor (left Y) vs value (right Y, log scale)
Plotly.newPlot('c-losses', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#58a6ff', width: 1.5 }, name: 'Actor Loss' },
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
  yaxis: { autorange: true, gridcolor: '#21262d', color: '#7d8590', zeroline: false },
  shapes: [
    { type: 'line', y0: -0.5, y1: -0.5, x0: 0, x1: 1, xref: 'paper', line: { color: '#f85149', width: 1, dash: 'dash' } },
    { type: 'line', y0: 0.3, y1: 0.3, x0: 0, x1: 1, xref: 'paper', line: { color: '#d29922', width: 1, dash: 'dot' } },
    { type: 'line', y0: 2.0, y1: 2.0, x0: 0, x1: 1, xref: 'paper', line: { color: '#30363d', width: 1, dash: 'dot' } }
  ]
}), plotConfig);

// Policy Std Dev (degrees)
Plotly.newPlot('c-policystd', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f0883e', width: 2 }, name: 'PolicyStd' }
], darkLayout({
  yaxis: { autorange: true, gridcolor: '#21262d', color: '#7d8590', zeroline: false }
}), plotConfig);

// 4. Gems/hr + best line
Plotly.newPlot('c-gemshr', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f0c040', width: 2 }, name: 'Gems/hr' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#3fb950', width: 1, dash: 'dash' }, name: 'Best' }
], darkLayout({ showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } } }), plotConfig);

// 5. Gems per game (bars) + rolling average line — full redraw each update
Plotly.newPlot('c-epgems', [
  { x: [], y: [], type: 'bar', marker: { color: '#f0c040' }, name: 'Game Gems' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f85149', width: 2 }, name: 'Avg' }
], darkLayout({ bargap: 0.15, showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } } }), plotConfig);

// 6. KL divergence (left Y) + KL-stop % rolling window (right Y)
Plotly.newPlot('c-kl', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#bc8cff', width: 1.5 }, name: 'KL' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f0c040', width: 2 }, name: 'KL-Stop %', yaxis: 'y2' }
], darkLayout({
  showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } },
  yaxis2: { overlaying: 'y', side: 'right', gridcolor: '#21262d', color: '#7d8590',
            range: [0, 100], zeroline: false, ticksuffix: '%' },
  shapes: [
    { type: 'line', y0: 2.25, y1: 2.25, x0: 0, x1: 1, xref: 'paper',
      line: { color: '#f85149', width: 1, dash: 'dash' } }  // KL-stop threshold
  ]
}), plotConfig);

// 7. Gradient norm (actor + critic)
Plotly.newPlot('c-gradnorm', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#ff7b72', width: 1.5 }, name: 'Actor' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#79c0ff', width: 1.5 }, name: 'Critic' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#30363d', width: 1, dash: 'dot' }, name: 'Clip (1.0)' }
], darkLayout({
  showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } },
  yaxis: { type: 'log', autorange: true, gridcolor: '#21262d', color: '#7d8590', zeroline: false }
}), plotConfig);

// 8. OOB per rollout
Plotly.newPlot('c-oob', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f85149', width: 1.5 } }
], darkLayout(), plotConfig);

// 9. Avg Gap Penalty Per Gem
Plotly.newPlot('c-gappen', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#bc8cff', width: 2 } }
], darkLayout(), plotConfig);

// 10. Jump Rate % Per Game
Plotly.newPlot('c-jumprate', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#f85149', width: 2 } }
], darkLayout(), plotConfig);

// 10b. Brake Rate % Per Game
Plotly.newPlot('c-brakerate', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#79c0ff', width: 2 } }
], darkLayout(), plotConfig);

// 11. Avg Steps Between Gem Pickups
Plotly.newPlot('c-stepspergem', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#d29922', width: 2 } }
], darkLayout(), plotConfig);

// 12. Throttle (mean + min/max fill)
// Trace order: 0=Min (bottom), 1=Max (fills down to Min), 2=Mean (on top)
Plotly.newPlot('c-throttle', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: 'rgba(88,166,255,0.3)', width: 0 },
    name: 'Min', showlegend: false },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: 'rgba(88,166,255,0.3)', width: 0 },
    fill: 'tonexty', fillcolor: 'rgba(88,166,255,0.15)', showlegend: false, name: 'Max' },
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#58a6ff', width: 2 }, name: 'Mean' }
], darkLayout({
  showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } },
  yaxis: { range: [0, 1.05], gridcolor: '#21262d', color: '#7d8590', zeroline: false }
}), plotConfig);

// 13. Laziness
Plotly.newPlot('c-laziness', [
  { x: [], y: [], type: 'scatter', mode: 'lines', line: { color: '#d29922', width: 2 } }
], darkLayout({
  yaxis: { autorange: true, gridcolor: '#21262d', color: '#7d8590', zeroline: false }
}), plotConfig);

// 13. Throughput
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

    // Policy Std
    if (h.policy_std) {
      Plotly.extendTraces('c-policystd', { x: [xs], y: [h.policy_std] }, [0]);
    }

    // Gems/hr + best line
    const bestGemsLine = h.best_gems_hr || h.gems_per_hr.map(() => 0);
    Plotly.extendTraces('c-gemshr', { x: [xs, xs], y: [h.gems_per_hr, bestGemsLine] }, [0, 1]);

    // KL divergence + KL-stop %
    if (h.kl) {
      const klStop = h.kl_stop_pct || h.kl.map(() => 0);
      Plotly.extendTraces('c-kl', { x: [xs, xs], y: [h.kl, klStop] }, [0, 1]);
    }

    // Gradient norm (actor + critic + clip line)
    if (h.grad_norm) {
      const criticGN = h.critic_grad_norm || h.grad_norm.map(() => 0);
      const clipLine = h.grad_norm.map(() => 1.0);
      Plotly.extendTraces('c-gradnorm', { x: [xs, xs, xs], y: [h.grad_norm, criticGN, clipLine] }, [0, 1, 2]);
    }

    // OOB
    Plotly.extendTraces('c-oob', { x: [xs], y: [h.rollout_oob] }, [0]);

    // Avg Gap Penalty Per Gem
    if (h.avg_gap_penalty) {
      Plotly.extendTraces('c-gappen', { x: [xs], y: [h.avg_gap_penalty] }, [0]);
    }

    // Jump Rate
    if (h.jump_rate) {
      Plotly.extendTraces('c-jumprate', { x: [xs], y: [h.jump_rate] }, [0]);
    }

    // Brake Rate
    if (h.brake_rate) {
      Plotly.extendTraces('c-brakerate', { x: [xs], y: [h.brake_rate] }, [0]);
    }

    // Avg Steps Per Gem
    if (h.avg_steps_per_gem) {
      Plotly.extendTraces('c-stepspergem', { x: [xs], y: [h.avg_steps_per_gem] }, [0]);
    }

    // Throttle (trace 0=Min, 1=Max, 2=Mean)
    if (h.throttle_mean) {
      Plotly.extendTraces('c-throttle', {
        x: [xs, xs, xs],
        y: [h.throttle_min, h.throttle_max, h.throttle_mean]
      }, [0, 1, 2]);
    }

    // Laziness
    if (h.avg_reward && h.gems_per_hr) {
      const lazY = h.avg_reward.map((r, i) => h.gems_per_hr[i] > 0 ? r / h.gems_per_hr[i] : 0);
      Plotly.extendTraces('c-laziness', { x: [xs], y: [lazY] }, [0]);
    }

    // Throughput
    if (h.timestamps.length > 1) {
      const tpXs = xs.slice(1);
      const tpY = [];
      for (let i = 1; i < h.timestamps.length; i++) {
        const dt = h.timestamps[i] - h.timestamps[i - 1];
        tpY.push(dt > 0.001 ? (h.config?.rollout_size || 2048) / dt : 0);
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
  // Heatmap refresh — only if a new version is available
  if (typeof window.heatmapMaybeRefresh === 'function' && snap.heatmap_version !== undefined) {
    window.heatmapMaybeRefresh(snap.heatmap_version);
  }

  // === Header ===
  document.getElementById('m-update').textContent = snap.update;
  document.getElementById('m-steps').textContent = snap.total_steps.toLocaleString();
  document.getElementById('m-eps').textContent = snap.total_episodes;
  document.getElementById('m-time').textContent = snap.elapsed_hrs.toFixed(2) + 'h';
  document.getElementById('m-gems').textContent = snap.total_gem_pts;
  document.getElementById('m-gems-hr').textContent = snap.gems_per_hr.toFixed(1);
  document.getElementById('m-best-gems-hr').textContent = snap.best_gems_hr.toFixed(1);
  document.getElementById('m-best-game-gems').textContent = snap.best_game_gems || '--';
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

  document.getElementById('g-gemshr').textContent = snap.gems_per_hr.toFixed(1);

  const gameGemsArr = snap.recent_game_gems || [];
  const lastGameGems = gameGemsArr.length > 0 ? gameGemsArr[gameGemsArr.length - 1] : 0;
  document.getElementById('g-lastgems').textContent = lastGameGems;

  // Avg gems/game (last 20 full games)
  const gemsGameEl = document.getElementById('g-gems-game');
  if (gameGemsArr.length > 0) {
    const recent20 = gameGemsArr.slice(-20);
    const avgGems = recent20.reduce((a, b) => a + b, 0) / recent20.length;
    gemsGameEl.textContent = avgGems.toFixed(1);
    gemsGameEl.style.color = avgGems >= 100 ? '#f0c040' : avgGems >= 85 ? '#3fb950' : '#e6edf3';
  } else {
    gemsGameEl.textContent = '--';
  }

  // KL-stop %
  const klStopEl = document.getElementById('g-klstop');
  const klPct = snap.kl_stop_pct || 0;
  klStopEl.textContent = klPct.toFixed(0) + '%';
  klStopEl.style.color = klPct >= 80 ? '#f85149' : klPct >= 40 ? '#d29922' : '#3fb950';

  // Grad norm
  const gnEl = document.getElementById('g-gradnorm');
  const gn = snap.grad_norm || 0;
  gnEl.textContent = gn.toFixed(2);
  gnEl.style.color = gn > 10 ? '#f85149' : gn > 2 ? '#d29922' : '#3fb950';

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
  Plotly.extendTraces('c-policystd', { x: [[x]], y: [[snap.policy_std]] }, [0]);
  Plotly.extendTraces('c-gemshr', { x: [[x], [x]], y: [[snap.gems_per_hr], [snap.best_gems_hr]] }, [0, 1]);
  Plotly.extendTraces('c-kl', { x: [[x], [x]], y: [[snap.kl || 0], [snap.kl_stop_pct || 0]] }, [0, 1]);
  Plotly.extendTraces('c-gradnorm', { x: [[x], [x], [x]], y: [[snap.grad_norm || 0], [snap.critic_grad_norm || 0], [1.0]] }, [0, 1, 2]);
  Plotly.extendTraces('c-oob', { x: [[x]], y: [[snap.rollout_oob]] }, [0]);

  // Avg Gap Penalty Per Gem
  Plotly.extendTraces('c-gappen', { x: [[x]], y: [[snap.avg_gap_penalty]] }, [0]);

  // Jump Rate
  Plotly.extendTraces('c-jumprate', { x: [[x]], y: [[snap.jump_rate || 0]] }, [0]);

  // Brake Rate
  Plotly.extendTraces('c-brakerate', { x: [[x]], y: [[snap.brake_rate || 0]] }, [0]);

  // Avg Steps Per Gem
  Plotly.extendTraces('c-stepspergem', { x: [[x]], y: [[snap.avg_steps_per_gem || 0]] }, [0]);

  // Throttle (trace 0=Min, 1=Max, 2=Mean)
  Plotly.extendTraces('c-throttle', { x: [[x], [x], [x]], y: [[snap.throttle_min || 1], [snap.throttle_max || 1], [snap.throttle_mean || 1]] }, [0, 1, 2]);

  // Laziness
  const laziness = snap.gems_per_hr > 0 ? snap.avg_reward_100ep / snap.gems_per_hr : 0;
  Plotly.extendTraces('c-laziness', { x: [[x]], y: [[laziness]] }, [0]);

  // Throughput
  if (prevTimestamp !== null) {
    const dt = snap.timestamp - prevTimestamp;
    const sps = dt > 0.001 ? snap.rollout_size / dt : 0;
    Plotly.extendTraces('c-throughput', { x: [[x]], y: [[sps]] }, [0]);
  }
  prevTimestamp = snap.timestamp;

  // === Snapshot charts (full redraw — small fixed-size arrays) ===

  // Gems per game (last 100 full games) — bars + rolling avg line
  const gameGems = snap.recent_game_gems || [];
  if (gameGems.length > 0) {
    const gemIdxs = gameGems.map((_, i) => i + 1);
    const maxGems = Math.max(...gameGems);
    const colors = gameGems.map(g => {
      if (g === maxGems && maxGems > 0) return '#3fb950';
      if (g > 0) return '#f0c040';
      return '#30363d';
    });
    // Rolling 10-game average line
    const window = 10;
    const rollingAvg = gameGems.map((_, i) => {
      const slice = gameGems.slice(Math.max(0, i - window + 1), i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
    Plotly.react('c-epgems', [
      { x: gemIdxs, y: gameGems, type: 'bar', marker: { color: colors }, name: 'Game Gems' },
      { x: gemIdxs, y: rollingAvg, type: 'scatter', mode: 'lines',
        line: { color: '#f85149', width: 2 }, name: `Avg (${window})` }
    ], darkLayout({ bargap: 0.15, showlegend: true, legend: { x: 0, y: 1, font: { size: 9 } } }), plotConfig);
  }
}

// ============================================================
// Jump probability heatmap — refetched when SSE reports new version
// ============================================================
(function setupHeatmap() {
  const canvas = document.getElementById('heatmap-canvas');
  const overlay = document.getElementById('heatmap-overlay');
  const ctx = canvas.getContext('2d');
  const octx = overlay.getContext('2d');
  const statsEl = document.getElementById('heatmap-stats');
  const statusEl = document.getElementById('heatmap-status');
  const W = canvas.width, H = canvas.height;
  let lastVersion = -1;
  let inFlight = false;

  function worldToPx(x, y, gridMin, gridMax, gridStep, cellPx) {
    const col = (x - gridMin) / gridStep;
    const row = (gridMax - y) / gridStep;
    return [col * cellPx + cellPx / 2, row * cellPx + cellPx / 2];
  }

  function render(payload) {
    if (!payload || !payload.grid) {
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#7d8590';
      ctx.font = '13px Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('No heatmap data yet — waiting for first probe…', W/2, H/2);
      return;
    }

    const grid = payload.grid;            // [rows][cols], grid[0]=y_min row
    const rows = grid.length, cols = grid[0].length;
    const cellPx = Math.floor(Math.min(W / cols, H / rows));
    const gridW = cols * cellPx, gridH = rows * cellPx;

    // Set the canvas's INTERNAL drawing resolution. CSS controls the displayed
    // size separately (width:100% on the canvas, capped by the wrapper's
    // max-width:100%) so the heatmap shrinks to fit narrow / mobile viewports
    // without distorting the underlying pixel grid.
    canvas.width = overlay.width = gridW;
    canvas.height = overlay.height = gridH;

    // Render the heatmap to the bottom canvas (greyscale)
    const img = ctx.createImageData(gridW, gridH);
    // y axis is flipped: grid[0] is y_min (bottom), but we want it at the bottom of the image
    for (let r = 0; r < rows; r++) {
      // grid row r corresponds to y = gridMin + r * gridStep
      // We want low Y at the bottom of canvas, so render row r at canvas-row (rows - 1 - r)
      const drawRow = rows - 1 - r;
      for (let c = 0; c < cols; c++) {
        const p = grid[r][c];
        const v = Math.max(0, Math.min(255, Math.round(p * 255)));
        // Fill the whole cellPx x cellPx block in the image
        for (let dy = 0; dy < cellPx; dy++) {
          for (let dx = 0; dx < cellPx; dx++) {
            const px = c * cellPx + dx;
            const py = drawRow * cellPx + dy;
            const idx = (py * gridW + px) * 4;
            img.data[idx]     = v;
            img.data[idx + 1] = v;
            img.data[idx + 2] = v;
            img.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Overlay: platform boxes + gem diamonds
    octx.clearRect(0, 0, gridW, gridH);
    const gridMin = payload.grid_min;
    const gridMax = payload.grid_max;
    const gridStep = payload.grid_step;

    // Platform boundaries (~2x2 world units → 2*cellPx pixels)
    const platSizePx = 2.0 * cellPx;
    octx.lineWidth = 2;
    octx.strokeStyle = '#58a6ff';
    for (const [px, py] of payload.platforms) {
      const [cx, cy] = worldToPx(px, py, gridMin, gridMax, gridStep, cellPx);
      octx.strokeRect(cx - platSizePx/2, cy - platSizePx/2, platSizePx, platSizePx);
    }

    function diamond(cx, cy, r, fill) {
      octx.beginPath();
      octx.moveTo(cx, cy - r);
      octx.lineTo(cx + r, cy);
      octx.lineTo(cx, cy + r);
      octx.lineTo(cx - r, cy);
      octx.closePath();
      octx.fillStyle = fill;
      octx.fill();
      octx.strokeStyle = '#0d1117';
      octx.lineWidth = 0.7;
      octx.stroke();
    }

    const r = 5;
    for (const [px, py] of payload.platforms) {
      const [cx, cy] = worldToPx(px, py, gridMin, gridMax, gridStep, cellPx);
      diamond(cx, cy, r, '#f0883e');   // elevated gems (orange)
    }
    for (const [gx, gy] of payload.ground_gems) {
      const [cx, cy] = worldToPx(gx, gy, gridMin, gridMax, gridStep, cellPx);
      diamond(cx, cy, r, '#f0c040');   // ground gems (gold)
    }

    // Stats line
    const s = payload.stats || {};
    const fmt = (x) => (x === undefined ? '--' : (x * 100).toFixed(1) + '%');
    const fmtPp = (x) => (x === undefined ? '--' : (x * 100).toFixed(2) + 'pp');
    statsEl.innerHTML =
      `<span><b>Min:</b> ${fmt(s.min)}</span>` +
      `<span><b>Max:</b> ${fmt(s.max)}</span>` +
      `<span><b>Mean:</b> ${fmt(s.mean)}</span>` +
      `<span><b>Std:</b> ${fmtPp(s.std)}</span>` +
      `<span><b>Platform avg:</b> ${fmt(s.platform_avg)}</span>` +
      `<span><b>Non-platform avg:</b> ${fmt(s.nonplatform_avg)}</span>` +
      `<span><b>Gap:</b> ${fmtPp(s.gap)}</span>`;

    if (statusEl) {
      statusEl.textContent = '(' + (payload.label || 'live') + ')';
    }
  }

  async function fetchHeatmap() {
    if (inFlight) return;
    inFlight = true;
    try {
      const resp = await fetch('/heatmap');
      if (!resp.ok) return;
      const payload = await resp.json();
      render(payload);
    } catch (err) {
      // Silent — will retry on next version bump
    } finally {
      inFlight = false;
    }
  }

  // Initial load (in case data is already there before any SSE message arrives)
  fetchHeatmap();

  // Public hook: dashboard SSE handler calls this when a new heatmap version arrives
  window.heatmapMaybeRefresh = function (newVersion) {
    if (typeof newVersion !== 'number') return;
    if (newVersion !== lastVersion) {
      lastVersion = newVersion;
      fetchHeatmap();
    }
  };
})();

// ============================================================
// Chat (Claude) — isolated from the dashboard SSE stream
// ============================================================
(function setupChat() {
  const logEl = document.getElementById('chat-log');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const clearBtn = document.getElementById('chat-clear');
  const conversation = [];   // [{role, content}, ...] — full multi-turn history
  let busy = false;

  function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
  }
  inputEl.addEventListener('input', autoResize);

  function scrollToBottom() {
    logEl.scrollTop = logEl.scrollHeight;
  }

  function clearEmptyPlaceholder() {
    const empty = logEl.querySelector('.chat-empty');
    if (empty) empty.remove();
  }

  function addTurn(role, text) {
    clearEmptyPlaceholder();
    const turn = document.createElement('div');
    turn.className = 'chat-turn ' + role;
    const r = document.createElement('div');
    r.className = 'chat-role';
    r.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Claude' : 'Error';
    const b = document.createElement('div');
    b.className = 'chat-body';
    b.textContent = text;
    turn.appendChild(r); turn.appendChild(b);
    logEl.appendChild(turn);
    scrollToBottom();
    return b;
  }

  function setBusy(v) {
    busy = v;
    sendBtn.disabled = v;
    inputEl.disabled = v;
    sendBtn.textContent = v ? '...' : 'Send';
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text || busy) return;

    conversation.push({ role: 'user', content: text });
    addTurn('user', text);
    inputEl.value = '';
    autoResize();
    setBusy(true);

    const bodyEl = addTurn('assistant', '');
    const cursor = document.createElement('span');
    cursor.className = 'chat-cursor';
    bodyEl.appendChild(cursor);

    let assistantText = '';
    let errored = false;

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: conversation }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error('HTTP ' + resp.status);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      outer:
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // SSE events are separated by blank lines
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const raw = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const line = raw.split('\n').find(l => l.startsWith('data:'));
          if (!line) continue;
          let payload;
          try { payload = JSON.parse(line.slice(5).trim()); }
          catch { continue; }
          if (payload.type === 'text') {
            assistantText += payload.text;
            bodyEl.textContent = assistantText;
            bodyEl.appendChild(cursor);
            scrollToBottom();
          } else if (payload.type === 'error') {
            bodyEl.textContent = assistantText;
            if (cursor.parentNode) cursor.remove();
            // Replace the in-progress assistant turn with an error turn
            const turn = bodyEl.closest('.chat-turn');
            if (turn) turn.remove();
            addTurn('error', payload.text);
            errored = true;
          } else if (payload.type === 'done') {
            break outer;
          }
        }
      }
    } catch (err) {
      if (cursor.parentNode) cursor.remove();
      const turn = bodyEl.closest('.chat-turn');
      if (turn && !assistantText) turn.remove();
      addTurn('error', 'Network error: ' + (err.message || err));
      errored = true;
    }

    if (cursor.parentNode) cursor.remove();

    if (!errored && assistantText) {
      conversation.push({ role: 'assistant', content: assistantText });
    } else if (errored) {
      // Drop the user turn we just added since no assistant reply was recorded
      conversation.pop();
    }

    setBusy(false);
    inputEl.focus();
  }

  sendBtn.addEventListener('click', send);

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  clearBtn.addEventListener('click', () => {
    if (busy) return;
    conversation.length = 0;
    logEl.innerHTML = '<div class="chat-empty">Conversation cleared.</div>';
  });
})();

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
