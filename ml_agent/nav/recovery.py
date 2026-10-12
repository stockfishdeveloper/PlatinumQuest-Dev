"""Progress and escape telemetry shared by training workers and evaluation."""
import math
import os


def shaping_configuration(gamma=0.99):
    """Explicit reward metadata; old runs used legacy progress, not discounted shaping."""
    k = float(os.environ.get('NAV_POTENTIAL_K', '0.05'))
    mode = os.environ.get('NAV_SHAPING_MODE', 'legacy')
    if mode not in ('legacy', 'discounted') or not math.isfinite(k) or k < 0:
        raise ValueError('Expected NAV_SHAPING_MODE legacy/discounted and finite nonnegative NAV_POTENTIAL_K')
    cfg = dict(version=1, mode=mode if k else 'off', k=k,
               gamma=gamma if mode == 'discounted' and k else 1.0,
               floor_hold_u=3.0, practice_reward=('shared_discounted_potential_bootstrapped_v2'
                                                if mode == 'discounted' and k else 'game_points_bootstrapped_v1'))
    # 2026-10-11 (operator: "You can charge every fall"; log 40.95): legacy shaping charges a fall against the gem it was
    # chasing even when the target changes at the respawn. Part of the reward, so recorded here: a checkpoint trained
    # without it differs and the trainer re-warms the critic.
    if mode == 'legacy' and k and os.environ.get('NAV_FALL_CHARGE', '1') == '1':
        cfg['fall_charge'] = True
    return cfg


def potential_term(phi_before, phi_after, duration_steps, gamma, k, terminal=False, goal_changed=False):
    """Exact duration-discounted potential difference, including target changes.

    gamma must match the learner. Discounted trajectory sums telescope, with
    Phi(next)=0 only at a true terminal. A target switch is still a transition;
    dropping it breaks that identity. goal_changed remains a compatibility flag
    and never suppresses the term. The caller owns initialization/floor history.
    """
    if k <= 0.0 or phi_before is None:
        return 0.0
    after = 0.0 if terminal else (phi_before if phi_after is None else phi_after)
    return float(k) * ((gamma ** float(duration_steps)) * after - phi_before)


def legacy_progress_term(phi_before, phi_after, k, terminal=False, goal_changed=False):
    """Reproduce the overnight route-progress heuristic; no policy-invariance claim."""
    if goal_changed:
        return 0.0
    return potential_term(phi_before, phi_after, 1.0, 1.0, k, terminal=terminal)


def simulation_time(env):
    if env.replay_world is not None:
        return float(env.replay_world['clock_ms']) / 1000.0
    return float(env.info.get('round_ms', 0)) / 1000.0 - env.time_left_s()


class ProgressMonitor:
    def __init__(self, window_s=8.0, progress_u=1.0):
        self.window_s = float(window_s)
        self.progress_u = float(progress_u)
        self.reset()

    def reset(self):
        self.target = None
        self.best = math.inf
        self.last_progress = None
        self.last_now = None
        self.stalled_s = 0.0
        self.rescues = 0
        self.rescue_attempts = 0
        self.escape_uses = 0
        self.powerup_escapes = 0
        self.failed_escape_uses = 0
        self.pending_escape = None
        self.group_completions = 0
        self.last_gems = None

    def observe(self, now, target, distance, picked=0.0, fell=False, gems=None, groups_completed=None):
        now = float(now)
        if self.last_now is not None and now < self.last_now:
            self.last_progress, self.best = now, math.inf
        dt = max(0.0, now - self.last_now) if self.last_now is not None else 0.0
        if groups_completed is not None:
            self.group_completions = int(groups_completed)
        if gems is not None:
            self.last_gems = set(gems)
        changed = target != self.target
        if changed:
            self.target, self.best = target, float(distance)
            # Gem-chooser oscillation is not progress. First observation initializes the clock.
        progress = picked > 0 or (math.isfinite(distance) and distance <= self.best - self.progress_u)
        if self.last_progress is None or progress or fell:
            self.last_progress = now
        if progress:
            self.best = float(distance)
        idle = max(0.0, now - self.last_progress)
        if idle >= 1.0 and target is not None and not fell:
            self.stalled_s += dt
        if self.pending_escape is not None:
            started = self.pending_escape
            if fell or now - started >= self.window_s:
                self.failed_escape_uses += 1
                self.pending_escape = None
            elif progress:
                self.powerup_escapes += 1
                self.pending_escape = None
        self.last_now = now
        return idle

    def escape_use(self, now):
        if self.pending_escape is None:
            self.escape_uses += 1
            self.pending_escape = float(now)

    def rescued(self, now, distance):
        self.rescues += 1
        if self.pending_escape is not None:
            self.failed_escape_uses += 1
            self.pending_escape = None
        self.last_progress, self.last_now, self.best = float(now), float(now), float(distance)

    def stats(self):
        return dict(rescues=self.rescues, rescue_attempts=self.rescue_attempts,
                    stalled_s=round(self.stalled_s, 3), escape_uses=self.escape_uses,
                    powerup_escapes=self.powerup_escapes,
                    failed_escape_uses=self.failed_escape_uses,
                    group_completions=self.group_completions)
