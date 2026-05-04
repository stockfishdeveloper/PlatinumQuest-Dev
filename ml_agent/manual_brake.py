"""
Manual time-optimal braking controller for PlatinumQuest Hunt mode.

Hand-coded controller (no neural network) that demonstrates the "fast approach
+ brake at switching point" trajectory we've been trying to teach the trained
model. Use this to:
  1. Verify the time-optimal math actually works in this physics environment
  2. Calibrate A_MAX (the max-deceleration constant) — tune until pickup speeds
     are consistently near 0
  3. Establish a benchmark — how many gems can a perfect brake-controller score?

Control logic (every step):
  v_radial      = component of velocity in the gem direction (positive = approaching)
  stopping_dist = v_radial^2 / (2 * A_MAX)         # physics: v^2 = 2 * a * d
  if stopping_dist + BRAKE_MARGIN > gem_dist:
      BRAKE: joystick points anti-velocity, full throttle
  else:
      ACCELERATE: joystick points at gem (XY only), full throttle

The marble is always heading toward the gem unless it's committed to overshoot,
at which point it brakes (anti-velocity force). Brake naturally tapers as speed
drops — when |velocity| is tiny, the anti-velocity unit vector is meaningless
and we just hold position.

Run on a FLAT map only — this controller doesn't handle jumping, ramps, or
anything beyond direct gem chase. Recommended map: FlatGemTraining_Hunt or
FlatWithJump_Hunt with jumping ignored.

Usage:
    python manual_brake.py
    python manual_brake.py --a_max 30.0     # if marble decelerates faster than 20
    python manual_brake.py --a_max 15.0     # if marble decelerates slower than 20
    python manual_brake.py --margin 1.5     # extra braking distance buffer

IMPORTANT: stop train_ppo.py / play.py before running — they share port 8888.
"""

import socket
import json
import math
import random
import argparse
import os
import sys
import datetime


# Tee class — duplicates stdout writes to a log file. Used so a long-running
# auto-tune session writes its output to disk for later analysis without
# requiring the user to manually redirect.
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class AutoTuner:
    """Hill-climb (A_MAX, BRAKE_MARGIN, MIN_TARGET_SPEED) game by game.

    After each game we have a (gems, pickup_speed) outcome. Score is computed
    as `gems - weight * pickup_speed` — reward gem count, penalize pickup speed.
    Each game perturbs one parameter; if score improves, keep that direction
    next game (momentum). If score drops by more than REVERT_THRESHOLD, the
    change is reverted and we try a different parameter.

    This is coordinate-descent hill-climbing — simple but effective for a
    smooth low-dim parameter landscape. Track best-seen params separately so
    the search can keep exploring without losing progress.
    """

    REVERT_THRESHOLD = 4.0   # absorb noise; only revert if score drops by more than this
    STEP_SIZES = {
        'a_max': 1.0,
        'margin': 0.1,
        'min_target_speed': 0.05,
    }
    BOUNDS = {
        'a_max': (3.0, 25.0),
        'margin': (0.0, 2.0),
        'min_target_speed': (0.0, 1.0),
    }

    def __init__(self, initial_params, pickup_weight=0.0):
        self.params = dict(initial_params)
        self.pickup_weight = pickup_weight
        self.history = []
        self.best_score = -float('inf')
        self.best_params = dict(initial_params)
        self.last_change = None  # (param_name, delta_applied, score_before)
        # Last successful direction per param — used for momentum
        self.directions = {k: 0 for k in self.params.keys()}

    def score(self, gems, pickup_speed):
        # Default: optimize purely for gems. Pickup speed is a soft penalty
        # only if pickup_weight > 0 (off by default).
        return gems - self.pickup_weight * pickup_speed

    def update(self, gems, pickup_speed):
        """Called once per completed game. Returns (status_str, score)."""
        current_score = self.score(gems, pickup_speed)
        self.history.append({
            'params': dict(self.params),
            'gems': gems,
            'pickup': pickup_speed,
            'score': current_score,
        })

        new_best = current_score > self.best_score
        if new_best:
            self.best_score = current_score
            self.best_params = dict(self.params)

        # Decide next change
        if self.last_change is not None:
            param, delta, score_before = self.last_change
            improvement = current_score - score_before

            if improvement >= -self.REVERT_THRESHOLD:
                # Change kept score in the same neighborhood or improved.
                # Continue in the same direction (momentum).
                self.directions[param] = +1 if delta > 0 else -1
                next_param = param
                next_delta = delta
                action = f"CONTINUE {param} {'+' if delta > 0 else '-'} (Δscore {improvement:+.1f})"
            else:
                # Score dropped meaningfully — revert and try a different param.
                self.params[param] -= delta
                self.directions[param] = -1 if delta > 0 else +1  # next time try opposite
                next_param = random.choice([p for p in self.params.keys() if p != param])
                d = self.directions.get(next_param, 0)
                next_delta = self.STEP_SIZES[next_param] * (d if d != 0 else random.choice([-1, 1]))
                action = f"REVERT {param} (Δscore {improvement:+.1f}); try {next_param}"
        else:
            # No prior change — pick something at random
            next_param = random.choice(list(self.params.keys()))
            next_delta = self.STEP_SIZES[next_param] * random.choice([-1, 1])
            action = f"INITIAL probe: {next_param}"

        # Apply with bounds
        old_val = self.params[next_param]
        new_val = old_val + next_delta
        lo, hi = self.BOUNDS[next_param]
        new_val = max(lo, min(hi, new_val))
        actual_delta = new_val - old_val

        if abs(actual_delta) > 1e-9:
            self.params[next_param] = new_val
            self.last_change = (next_param, actual_delta, current_score)
            change_str = f"{next_param}: {old_val:.2f} -> {new_val:.2f}"
        else:
            # Hit a bound — randomize next time
            self.last_change = None
            change_str = f"{next_param} at bound ({old_val:.2f})"

        best_str = " [NEW BEST]" if new_best else f" [best score so far: {self.best_score:.1f}]"
        status = f"  AUTO-TUNE: {action} | {change_str}{best_str}"
        return status, current_score


def _print_benchmark_summary(configs, results):
    """Print a side-by-side summary comparing benchmark configs."""
    print()
    print("=" * 78)
    print("BENCHMARK RESULTS")
    print("=" * 78)
    summaries = []
    for cfg, res in zip(configs, results):
        if not res:
            print(f"\n{cfg['label']}: NO DATA")
            continue
        gems = [r['gems'] for r in res]
        pickup = [r['pickup_speed'] for r in res]
        n = len(gems)
        gem_avg = sum(gems) / n
        gem_min = min(gems)
        gem_max = max(gems)
        gem_std = (sum((g - gem_avg) ** 2 for g in gems) / n) ** 0.5
        pu_avg = sum(pickup) / n
        pu_min = min(pickup)
        pu_max = max(pickup)
        summaries.append({
            'cfg': cfg, 'n': n, 'gem_avg': gem_avg, 'gem_min': gem_min,
            'gem_max': gem_max, 'gem_std': gem_std, 'pu_avg': pu_avg,
            'pu_min': pu_min, 'pu_max': pu_max, 'gems': gems,
        })
        print(f"\n{cfg['label']}:")
        print(f"  Params:       a_max={cfg['a_max']:.2f}  margin={cfg['margin']:.2f}  min_target={cfg['min_target_speed']:.2f}")
        print(f"  Games:        {n}")
        print(f"  Gems:         avg={gem_avg:.1f}  std={gem_std:.1f}  min={gem_min}  max={gem_max}")
        print(f"  Pickup speed: avg={pu_avg:.2f}  min={pu_min:.2f}  max={pu_max:.2f}")
        print(f"  Per-game gems: {gems}")

    if len(summaries) >= 2:
        print()
        print("-" * 78)
        # Pick winner by avg gems
        best = max(summaries, key=lambda s: s['gem_avg'])
        print(f"WINNER (by avg gems): {best['cfg']['label']}  →  avg {best['gem_avg']:.1f} gems")
        for s in summaries:
            if s is best:
                continue
            delta = best['gem_avg'] - s['gem_avg']
            print(f"  vs {s['cfg']['label']}:  avg {s['gem_avg']:.1f} gems  ({delta:+.1f} gem advantage)")

        # Statistical significance check (simple): if difference < std of either, it's noise
        if len(summaries) == 2:
            a, b = summaries
            delta = abs(a['gem_avg'] - b['gem_avg'])
            combined_std = max(a['gem_std'], b['gem_std'])
            print()
            if delta > combined_std * 1.5:
                print(f"  Difference of {delta:.1f} gems is LIKELY meaningful (> 1.5 × max std {combined_std:.1f}).")
            elif delta > combined_std * 0.5:
                print(f"  Difference of {delta:.1f} gems is MAYBE meaningful (between 0.5x and 1.5x max std {combined_std:.1f}).")
                print(f"  Recommend more games per config for confidence.")
            else:
                print(f"  Difference of {delta:.1f} gems is WITHIN NOISE (< 0.5 × max std {combined_std:.1f}).")
                print(f"  Configs are statistically indistinguishable. Either is fine.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description='Manual time-optimal brake controller')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--a_max', type=float, default=20.0,
                        help='Max marble deceleration estimate (raw units/s^2). '
                             'Tune until pickup speeds are consistently near 0.')
    parser.add_argument('--margin', type=float, default=0.0,
                        help='Extra units added to stopping distance — start braking '
                             'this much earlier. Increase if you observe overshoot.')
    parser.add_argument('--max_speed', type=float, default=25.0,
                        help='Marble max speed estimate (units/s). Cap on target speed.')
    parser.add_argument('--min_target_speed', type=float, default=0.5,
                        help='Minimum target speed even at the gem itself. Ensures marble '
                             'always crawls forward to make contact instead of stopping '
                             'short. Lower = cleaner pickup speed but risks halting just '
                             'before pickup; higher = more guaranteed pickup but higher speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every step (sampled, every 10/30 steps)')
    parser.add_argument('--tune', action='store_true',
                        help='Auto-tune (a_max, margin, min_target_speed) after every game. '
                             'Hill-climbs to maximize score = gems - pickup_weight*pickup_speed.')
    parser.add_argument('--pickup_weight', type=float, default=0.0,
                        help='Weight on pickup_speed penalty in tuning score. Default 0 = pure '
                             'gem-count optimization. Set to 1-5 to also penalize high pickup speed.')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Write all stdout output to this log file in addition to terminal. '
                             'Default: auto-named manual_brake_YYYYMMDD_HHMMSS.log in cwd.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run a fixed comparison: BENCHMARK_CONFIGS (hardcoded list) × '
                             '--games_per_config games each, then print summary and exit. '
                             'Disables --tune. Use this to A/B test parameter sets reliably.')
    parser.add_argument('--games_per_config', type=int, default=10,
                        help='How many games to run per config in --benchmark mode (default 10).')
    args = parser.parse_args()

    if args.benchmark and args.tune:
        print("ERROR: --benchmark and --tune are mutually exclusive.")
        sys.exit(1)

    # Hardcoded configs for --benchmark mode. Edit this list to change what's tested.
    # These two were the 108-gem peaks discovered by the auto-tuner.
    BENCHMARK_CONFIGS = [
        {'label': 'A: aggressive (a=12, m=0.00, min=0.05)',
         'a_max': 12.0, 'margin': 0.00, 'min_target_speed': 0.05},
        {'label': 'B: slow-creep (a=13, m=0.00, min=0.85)',
         'a_max': 13.0, 'margin': 0.00, 'min_target_speed': 0.85},
    ]

    # Tee stdout to a log file so progress is recoverable later.
    log_path = args.logfile or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"manual_brake_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_fh = open(log_path, 'w', encoding='utf-8', buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    print(f"Log file: {log_path}")

    A_MAX = args.a_max
    BRAKE_MARGIN = args.margin
    MAX_SPEED = args.max_speed
    MIN_TARGET_SPEED = args.min_target_speed

    tuner = None
    if args.tune:
        tuner = AutoTuner(
            initial_params={
                'a_max': A_MAX,
                'margin': BRAKE_MARGIN,
                'min_target_speed': MIN_TARGET_SPEED,
            },
            pickup_weight=args.pickup_weight,
        )
        print(f"AUTO-TUNE: enabled. score = gems - {args.pickup_weight} * pickup_speed")
        print(f"AUTO-TUNE: bounds = {tuner.BOUNDS}")
        print(f"AUTO-TUNE: step sizes = {tuner.STEP_SIZES}")

    # Benchmark mode state
    benchmark_state = None
    if args.benchmark:
        # Override starting params from first config
        cfg0 = BENCHMARK_CONFIGS[0]
        A_MAX = cfg0['a_max']
        BRAKE_MARGIN = cfg0['margin']
        MIN_TARGET_SPEED = cfg0['min_target_speed']
        benchmark_state = {
            'config_idx': 0,
            'games_done_this_config': 0,
            'results': [[] for _ in BENCHMARK_CONFIGS],
        }
        print(f"BENCHMARK MODE: {len(BENCHMARK_CONFIGS)} configs × {args.games_per_config} games each = {len(BENCHMARK_CONFIGS) * args.games_per_config} games total")
        for i, cfg in enumerate(BENCHMARK_CONFIGS):
            print(f"  Config {i+1}/{len(BENCHMARK_CONFIGS)}: {cfg['label']}")
        print(f"  Starting with: {BENCHMARK_CONFIGS[0]['label']}")

    print(f"Manual time-optimal brake controller")
    print(f"  A_MAX             = {A_MAX:.1f}    (max-deceleration estimate)")
    print(f"  MAX_SPEED         = {MAX_SPEED:.1f}    (cap on target speed)")
    print(f"  MIN_TARGET_SPEED  = {MIN_TARGET_SPEED:.2f}    (creep speed at gem)")
    print(f"  BRAKE_MARGIN      = {BRAKE_MARGIN:.2f}    (effective stopping point: gem - margin)")
    print(f"  Target speed at d=10:  {min(MAX_SPEED, max(MIN_TARGET_SPEED, math.sqrt(2*A_MAX*max(0, 10-BRAKE_MARGIN)))):.2f}")
    print(f"  Target speed at d=5:   {min(MAX_SPEED, max(MIN_TARGET_SPEED, math.sqrt(2*A_MAX*max(0, 5-BRAKE_MARGIN)))):.2f}")
    print(f"  Target speed at d=2:   {min(MAX_SPEED, max(MIN_TARGET_SPEED, math.sqrt(2*A_MAX*max(0, 2-BRAKE_MARGIN)))):.2f}")
    print(f"  Target speed at d=1:   {min(MAX_SPEED, max(MIN_TARGET_SPEED, math.sqrt(2*A_MAX*max(0, 1-BRAKE_MARGIN)))):.2f}")
    print(f"  Target speed at d=0.5: {min(MAX_SPEED, max(MIN_TARGET_SPEED, math.sqrt(2*A_MAX*max(0, 0.5-BRAKE_MARGIN)))):.2f}")
    print(f"  Switching point at v=25:  stopping_dist = {25*25/(2*A_MAX) + BRAKE_MARGIN:.1f} units")
    print(f"  Switching point at v=15:  stopping_dist = {15*15/(2*A_MAX) + BRAKE_MARGIN:.1f} units")
    print(f"  Switching point at v=10:  stopping_dist = {10*10/(2*A_MAX) + BRAKE_MARGIN:.1f} units")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Run game at 1x speed (or higher for benchmarking). Make sure no other agent is on port 8888.")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((args.host, args.port))
    sock.listen(1)

    # Per-episode stats
    total_steps = 0
    brake_steps = 0
    episode_gems = 0
    pickup_speeds = []           # all pickup speeds in this episode
    total_episodes = 0
    all_episode_gems = []        # gems each episode
    all_episode_pickup_speeds = []  # avg pickup speed each episode

    try:
        while True:
            try:
                conn, addr = sock.accept()
                print(f"Game connected from {addr}")
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

            buffer_str = ""
            try:
                while True:
                    data = conn.recv(8192).decode('utf-8')
                    if not data:
                        print("Game disconnected")
                        break

                    buffer_str += data
                    while '\n' in buffer_str:
                        line, buffer_str = buffer_str.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue

                        # Protocol: obs_json|gem_delta|oob|done
                        parts = line.split('|')
                        if len(parts) != 4:
                            conn.sendall(b'0,0,0,0,0\n')
                            continue

                        obs_json, gem_delta_str, oob_str, done_str = parts
                        obs = json.loads(obs_json)
                        gem_delta = float(gem_delta_str)
                        done = int(float(done_str))

                        # Game-end signal handling. CS sends three different "done" events:
                        #   1. empty obs + done=1 + negative gem_delta = TRUE game end (5-min timer)
                        #   2. done=1 with non-empty obs = intermediate event (OOB respawn,
                        #      marble reset within a game). Stats should KEEP accumulating —
                        #      this is the user's same in-progress game.
                        #   3. obs non-empty + done=0 = normal step.
                        #
                        # Only case 1 should print summary stats and reset accumulators.
                        if len(obs) == 0:
                            if done:
                                # TRUE game end. Negative gem_delta is total game gems.
                                game_total_gems = int(abs(gem_delta))
                                total_episodes += 1
                                brake_pct = (brake_steps / max(total_steps, 1)) * 100
                                avg_pickup = (sum(pickup_speeds) / len(pickup_speeds)) if pickup_speeds else 0.0
                                min_pickup = min(pickup_speeds) if pickup_speeds else 0.0
                                max_pickup = max(pickup_speeds) if pickup_speeds else 0.0
                                print()
                                print(f"=== Game {total_episodes} done (timer expired) ===")
                                print(f"  params used     = a_max={A_MAX:.2f}  margin={BRAKE_MARGIN:.2f}  min_target={MIN_TARGET_SPEED:.2f}")
                                print(f"  gems            = {game_total_gems}pts")
                                print(f"  steps           = {total_steps}")
                                print(f"  brake %         = {brake_pct:.1f}%")
                                print(f"  pickup speed    = avg {avg_pickup:.2f}    min {min_pickup:.2f}    max {max_pickup:.2f}")
                                all_episode_gems.append(game_total_gems)
                                all_episode_pickup_speeds.append(avg_pickup)
                                if len(all_episode_gems) > 1:
                                    running_avg = sum(all_episode_gems) / len(all_episode_gems)
                                    running_pickup = sum(all_episode_pickup_speeds) / len(all_episode_pickup_speeds)
                                    print(f"  running avg gems         = {running_avg:.1f}")
                                    print(f"  running avg pickup speed = {running_pickup:.2f}")

                                # Auto-tune step
                                if tuner is not None:
                                    status, score_val = tuner.update(game_total_gems, avg_pickup)
                                    print(f"  score this game = {score_val:.1f}")
                                    print(status)
                                    # Apply new params for next game
                                    A_MAX = tuner.params['a_max']
                                    BRAKE_MARGIN = tuner.params['margin']
                                    MIN_TARGET_SPEED = tuner.params['min_target_speed']
                                    print(f"  next game params = a_max={A_MAX:.2f}  margin={BRAKE_MARGIN:.2f}  min_target={MIN_TARGET_SPEED:.2f}")
                                    print(f"  best params seen = a_max={tuner.best_params['a_max']:.2f}  margin={tuner.best_params['margin']:.2f}  min_target={tuner.best_params['min_target_speed']:.2f}  (score {tuner.best_score:.1f})")

                                # Benchmark step
                                if benchmark_state is not None:
                                    cfg_idx = benchmark_state['config_idx']
                                    cfg = BENCHMARK_CONFIGS[cfg_idx]
                                    benchmark_state['results'][cfg_idx].append({
                                        'gems': game_total_gems,
                                        'pickup_speed': avg_pickup,
                                        'min_pickup': min_pickup,
                                        'max_pickup': max_pickup,
                                        'steps': total_steps,
                                        'brake_pct': brake_pct,
                                    })
                                    benchmark_state['games_done_this_config'] += 1
                                    n_done = benchmark_state['games_done_this_config']
                                    n_total = args.games_per_config
                                    print(f"  BENCHMARK: {cfg['label']} -- game {n_done}/{n_total}")

                                    if n_done >= n_total:
                                        # Move to next config or finish
                                        cfg_idx += 1
                                        if cfg_idx >= len(BENCHMARK_CONFIGS):
                                            # All configs done — print summary
                                            _print_benchmark_summary(BENCHMARK_CONFIGS, benchmark_state['results'])
                                            print("Benchmark complete. Exiting.")
                                            sys.exit(0)
                                        else:
                                            # Switch to next config
                                            benchmark_state['config_idx'] = cfg_idx
                                            benchmark_state['games_done_this_config'] = 0
                                            new_cfg = BENCHMARK_CONFIGS[cfg_idx]
                                            A_MAX = new_cfg['a_max']
                                            BRAKE_MARGIN = new_cfg['margin']
                                            MIN_TARGET_SPEED = new_cfg['min_target_speed']
                                            print(f"  BENCHMARK: switching to {new_cfg['label']}")
                                            print(f"             new params: a_max={A_MAX:.2f}  margin={BRAKE_MARGIN:.2f}  min_target={MIN_TARGET_SPEED:.2f}")
                                print()
                                # Reset only here, on TRUE game end
                                episode_gems = 0
                                total_steps = 0
                                brake_steps = 0
                                pickup_speeds = []
                            conn.sendall(b'0,0,0,0,0\n')
                            continue

                        # Camera-relative velocity and gem-relative position. Same frame
                        # as train_ppo's compute_reward / Actor.action_to_joystick.
                        vx, vy, vz = obs[3], obs[4], obs[5]
                        gx_rel, gy_rel, gz_rel = obs[6], obs[7], obs[8]
                        gem_dist = obs[10]

                        speed = math.sqrt(vx*vx + vy*vy + vz*vz)

                        # No gem visible — idle (game state between spawns or post-pickup grace).
                        if gem_dist <= 0 or gem_dist >= 900:
                            conn.sendall(b'0,0,0,0,0\n')
                            continue

                        # Track pickup
                        if gem_delta > 0:
                            episode_gems += int(gem_delta)
                            pickup_speeds.append(speed)
                            recent_avg = sum(pickup_speeds[-10:]) / min(10, len(pickup_speeds))
                            print(f"  PICKUP: +{gem_delta:.0f} at speed {speed:.2f}    "
                                  f"(recent avg {recent_avg:.2f})    "
                                  f"gems_this_ep={episode_gems}")

                        # Radial velocity: kept for diagnostics only.
                        if gem_dist > 0.01:
                            v_radial = (vx * gx_rel + vy * gy_rel + vz * gz_rel) / gem_dist
                        else:
                            v_radial = 0.0

                        # ====================================================================
                        # UNIFIED CONTROL LAW (replaces separate brake/accel branches)
                        #
                        # Define:  target_speed(d) = sqrt(2 * A_MAX * max(0, d - margin))
                        #                            clamped to [MIN_TARGET_SPEED, MAX_SPEED]
                        #
                        # This is the max speed from which the marble can decelerate to zero
                        # by the time it reaches `gem_dist - margin`. As d → margin, target
                        # speed → 0, and the marble's velocity gets driven toward zero AT the
                        # gem (rather than firing brake at the last moment with leftover speed).
                        #
                        # Joystick = direction that drives current velocity → (gem_direction *
                        # target_speed). This is proportional navigation: it accounts for both
                        # tangential momentum (bleeds off sideways drift) AND distance-based
                        # deceleration (target speed shrinks as marble approaches).
                        #
                        # No brake/accel mode switch needed — the same law smoothly transitions
                        # from "accelerate to MAX_SPEED" (when far) to "decelerate to ~0" (when
                        # close). MIN_TARGET_SPEED prevents the marble from stalling out short
                        # of the gem due to friction taking over at zero target.
                        # ====================================================================

                        effective_dist = max(0.0, gem_dist - BRAKE_MARGIN)
                        physics_target_speed = math.sqrt(2.0 * A_MAX * effective_dist)
                        target_speed = max(MIN_TARGET_SPEED, min(MAX_SPEED, physics_target_speed))

                        # Direction toward gem (XY only — z handled by gravity)
                        dist_xy = math.sqrt(gx_rel*gx_rel + gy_rel*gy_rel)
                        if dist_xy > 0.01:
                            gem_dir_x = gx_rel / dist_xy
                            gem_dir_y = gy_rel / dist_xy
                        else:
                            gem_dir_x, gem_dir_y = 0.0, 0.0

                        # Target velocity vector
                        target_vx = gem_dir_x * target_speed
                        target_vy = gem_dir_y * target_speed

                        # Steering vector: how to change current velocity to match target
                        steer_x = target_vx - vx
                        steer_y = target_vy - vy
                        steer_mag = math.sqrt(steer_x*steer_x + steer_y*steer_y)

                        if steer_mag > 0.01:
                            dx = steer_x / steer_mag
                            dy = steer_y / steer_mag
                        else:
                            dx, dy = 0.0, 0.0
                        throttle = 1.0

                        # Diagnostic: count "braking" frames as those where marble is being
                        # asked to decelerate (target speed < current speed).
                        is_braking = target_speed < speed - 0.5
                        if is_braking:
                            brake_steps += 1

                        if args.verbose and total_steps % 30 == 0:
                            mode = "BRAKE" if is_braking else "ACCEL"
                            print(f"  {mode} @ step {total_steps}: speed={speed:.2f} "
                                  f"target={target_speed:.2f} v=({vx:.1f},{vy:.1f}) "
                                  f"target_v=({target_vx:.1f},{target_vy:.1f}) "
                                  f"joy=({dx:.2f},{dy:.2f}) gem_dist={gem_dist:.2f}")

                        # Convert (dx, dy, throttle) → joystick (fwd, back, left, right, jump).
                        # Convention: dx>0 = right, dy>0 = forward.
                        move_x = dx * throttle
                        move_y = dy * throttle

                        fwd   = round(max(move_y, 0.0), 6) + 0.0
                        back  = round(max(-move_y, 0.0), 6) + 0.0
                        right = round(max(move_x, 0.0), 6) + 0.0
                        left  = round(max(-move_x, 0.0), 6) + 0.0

                        action_tuple = (fwd, back, left, right, 0)  # never jump
                        conn.sendall((','.join(map(str, action_tuple)) + '\n').encode('utf-8'))

                        total_steps += 1

                        # `done=1` here means an intermediate event (OOB respawn etc.) — NOT
                        # the end of the user's 5-min game. The TRUE game-end signal arrives
                        # as empty-obs + done=1 (handled at the top of the loop). Don't reset
                        # stats here. Optionally print a marker so the user knows when an OOB
                        # / mid-game reset happened.
                        if done:
                            print(f"  [intermediate reset @ step {total_steps}, {episode_gems} gems so far — stats keep accumulating]")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
