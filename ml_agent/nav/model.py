"""Navigator actor-critic: CNN over the height crops + MLP over the vector -> GRU -> heads.

Heads and their distributions are the ones proven in train_ppo.Actor (copied,
not imported): 2D direction Gaussian on a unit-circle mean, throttle Gaussian
in logit space, jump and brake Bernoullis. The value head uses PopArt
normalization like train_ppo.Critic.

Action stored in the rollout buffer: (dx, dy, throttle_logit, jump, brake).
Action for the game: (dx, dy, throttle, jump, brake) -> action_to_joystick.
"""
import math
import torch
import torch.nn as nn

from nav.obs import VEC_DIM
from nav.terrain import CROP_SHAPE

HIDDEN = 256
ACTION_DIM = 5


class NavActorCritic(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, -0.5
    THROTTLE_LOG_STD_MIN, THROTTLE_LOG_STD_MAX = -3.0, -0.9
    THROTTLE_FLOOR = 0.15
    POPART_BETA = 0.05
    POPART_MIN_STD = 1e-2

    def __init__(self):
        super().__init__()
        c, hgt, wid = CROP_SHAPE
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (hgt // 4) * (wid // 4), 128), nn.ReLU(),
        )
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 64), nn.ReLU())
        self.pre = nn.Sequential(nn.Linear(128 + 64, HIDDEN), nn.ReLU())
        self.gru = nn.GRUCell(HIDDEN, HIDDEN)
        self.dir_head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
        nn.init.uniform_(self.dir_head[2].weight, -0.1, 0.1); nn.init.zeros_(self.dir_head[2].bias)
        self.throttle_head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.throttle_head[-1].bias, 2.0)
        self.jump_head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1))
        # -3 (4.7 %): with the old -1 (27 %) a fresh policy jumped every ~4 decisions, was airborne
        # 90 % of the time, had no traction, so neither the direction nor the airborne penalty
        # produced a gradient (12 k decisions, jump rate flat at 0.28, 2026-09-17).
        nn.init.constant_(self.jump_head[-1].bias, -3.0)
        self.brake_head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.brake_head[-1].bias, -3.0)
        self.value_head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((1,), -1.5))
        self.throttle_log_std = nn.Parameter(torch.full((1,), -1.0))
        self.register_buffer('value_mean', torch.zeros(1))
        self.register_buffer('value_std', torch.ones(1))
        self.register_buffer('value_sq_mean', torch.ones(1))
        self.register_buffer('value_stats_initialized', torch.zeros(1))

    # ------------------------------------------------------------------ core
    def initial_state(self, batch, device):
        return torch.zeros(batch, HIDDEN, device=device)

    def core(self, crop, vec, h):
        x = torch.cat([self.conv(crop), self.vec(vec)], dim=1)
        return self.gru(self.pre(x), h)

    def heads(self, h):
        mean_xy = self.dir_head(h)
        thr = self.throttle_head(h).squeeze(-1)
        jump = torch.clamp(self.jump_head(h).squeeze(-1), -7.0, 3.0)
        brake = torch.clamp(self.brake_head(h).squeeze(-1), -7.0, 3.0)
        vn = self.value_head(h).squeeze(-1)
        return mean_xy, thr, jump, brake, vn

    def dists(self, mean_xy, thr, jump, brake):
        # The tanh output is the Gaussian mean directly (NOT normalized to the unit circle:
        # normalizing a near-zero fresh output made the direction, and the PPO ratio,
        # chaotic). action_to_joystick normalizes the sampled direction.
        std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean_xy)
        d_dir = torch.distributions.Normal(mean_xy, std)
        d_thr = torch.distributions.Normal(thr, torch.clamp(self.throttle_log_std, self.THROTTLE_LOG_STD_MIN, self.THROTTLE_LOG_STD_MAX).exp())
        d_jump = torch.distributions.Bernoulli(logits=jump)
        d_brake = torch.distributions.Bernoulli(logits=brake)
        return d_dir, d_thr, d_jump, d_brake

    def value_from_norm(self, vn):
        return vn * self.value_std + self.value_mean

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, crop, vec, h, deterministic=False):
        """crop (B,6,32,32), vec (B,48), h (B,H). Returns dict with action_buf (B,5), action_game (B,5),
        logp (B,), value (B,), h_next (B,H)."""
        h1 = self.core(crop, vec, h)
        mean_xy, thr, jump, brake, vn = self.heads(h1)
        d_dir, d_thr, d_jump, d_brake = self.dists(mean_xy, thr, jump, brake)
        if deterministic:
            direction = d_dir.mean; thr_s = thr; j = (torch.sigmoid(jump) > 0.5).float(); b = (torch.sigmoid(brake) > 0.5).float()
        else:
            direction = d_dir.sample(); thr_s = d_thr.sample(); j = d_jump.sample(); b = d_brake.sample()
        logp = d_dir.log_prob(direction).sum(-1) + d_thr.log_prob(thr_s) + d_jump.log_prob(j) + d_brake.log_prob(b)
        throttle = self.THROTTLE_FLOOR + (1 - self.THROTTLE_FLOOR) * torch.sigmoid(thr_s)
        action_buf = torch.stack([direction[:, 0], direction[:, 1], thr_s, j, b], dim=1)
        action_game = torch.stack([direction[:, 0], direction[:, 1], throttle, j, b], dim=1)
        return {'action_buf': action_buf, 'action_game': action_game, 'logp': logp,
                'value': self.value_from_norm(vn), 'h_next': h1}

    # ------------------------------------------------------------------ training
    def evaluate_seq(self, crops, vecs, h0, actions, resets):
        """crops (T,B,6,32,32), vecs (T,B,48), h0 (B,H), actions (T,B,5),
        resets (T,B) float: 1 where the hidden state must be zeroed BEFORE step t
        (first step of a new segment). Returns logp (T,B), entropy (T,B), value_norm (T,B)."""
        T, B = actions.shape[:2]
        h = h0
        logps, ents, vns = [], [], []
        for t in range(T):
            h = h * (1.0 - resets[t]).unsqueeze(-1)
            h = self.core(crops[t], vecs[t], h)
            mean_xy, thr, jump, brake, vn = self.heads(h)
            d_dir, d_thr, d_jump, d_brake = self.dists(mean_xy, thr, jump, brake)
            a = actions[t]
            logp = d_dir.log_prob(a[:, 0:2]).sum(-1) + d_thr.log_prob(a[:, 2]) + d_jump.log_prob(a[:, 3]) + d_brake.log_prob(a[:, 4])
            ent = d_dir.entropy().sum(-1) + d_thr.entropy() + d_jump.entropy() + d_brake.entropy()
            logps.append(logp); ents.append(ent); vns.append(vn)
        return torch.stack(logps), torch.stack(ents), torch.stack(vns)

    @torch.no_grad()
    def update_value_stats(self, returns):
        """PopArt: refresh running mean/std of the returns and rescale the value head so V(s) is unchanged."""
        old_mean, old_std = self.value_mean.clone(), self.value_std.clone()
        bm, bsq = returns.mean(), (returns ** 2).mean()
        if self.value_stats_initialized.item() == 0:
            self.value_mean.copy_(bm); self.value_sq_mean.copy_(bsq); self.value_stats_initialized.fill_(1)
        else:
            b = self.POPART_BETA
            self.value_mean.mul_(1 - b).add_(b * bm); self.value_sq_mean.mul_(1 - b).add_(b * bsq)
        var = (self.value_sq_mean - self.value_mean ** 2).clamp(min=0)
        self.value_std.copy_(var.sqrt().clamp(min=self.POPART_MIN_STD))
        last = self.value_head[-1]
        last.weight.mul_(old_std / self.value_std)
        last.bias.copy_((last.bias * old_std + old_mean - self.value_mean) / self.value_std)


def action_to_joystick(dx, dy, throttle, jump=0, brake=0, vx=0.0, vy=0.0):
    """(dx, dy) world direction (+x right/east, +y forward/north), throttle [0,1]; brake overrides
    the direction to anti-velocity at full throttle. Returns (fwd, back, left, right, jump)."""
    if brake > 0.5:
        speed_xy = math.sqrt(vx * vx + vy * vy)
        if speed_xy > 0.1:
            dx, dy, throttle = -vx / speed_xy, -vy / speed_xy, 1.0
    else:
        n = math.sqrt(dx * dx + dy * dy)
        if n > 1e-6:
            dx, dy = dx / n, dy / n
    mx, my = dx * throttle, dy * throttle
    return (round(max(my, 0.0), 6), round(max(-my, 0.0), 6), round(max(-mx, 0.0), 6), round(max(mx, 0.0), 6), int(jump))
