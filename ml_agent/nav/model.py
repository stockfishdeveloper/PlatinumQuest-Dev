"""Navigator actor-critic: CNN over the height crops + MLP over the vector -> GRU -> heads.

Heads and their distributions are the ones proven in train_ppo.Actor (copied,
not imported): 2D direction Gaussian on a unit-circle mean, throttle Gaussian
in logit space, jump and brake Bernoullis. The value head uses PopArt
normalization like train_ppo.Critic.

Action stored in the rollout buffer: (dx, dy, throttle_logit, jump, brake).
Action for the game: (dx, dy, throttle, jump, brake) -> action_to_joystick.
"""
import math
import os
import torch
import torch.nn as nn

from nav.obs import VEC_DIM
from nav.terrain import CROP_SHAPE

HIDDEN = 256
ACTION_DIM = 5

# Jump prior (2026-09-17): a fixed, non-learned bonus on the jump logit when the edge ray
# toward the waypoint shows a drop within JUMP_PRIOR_DIST u, the marble is on the floor and
# moving. Without it the learned jump logit sat at its -7 clamp (0.1 %) after the flat maps,
# so a jump at a gap edge was never sampled and could never be learned. The learned head can
# cancel the bonus where jumping is bad. Map-independent: it only reads the rays.
JUMP_PRIOR = 6.0            # logit bonus (-7 -> -1 = 27 % per decision at a gap edge)
JUMP_PRIOR_DIST = 2.5       # edge within this many u along the line to the waypoint
JUMP_PRIOR_DROP = -0.15     # ray "beyond" value below this = a drop of > 1.5 u or void
JUMP_PRIOR_SPEED = 1.5      # u/s (was 2.0: braking dropped the marble under the gate and it rolled off)
JUMP_LANDING_MAX = 4.5      # a landing within this many u beyond the edge (fine crop, 0.5 u cells) = crossable
JUMP_LANDING_DZ = 0.15      # landing height within +/-1.5 u of the current floor (crop units are /10)
DIR_GOAL_GAIN = float(os.environ.get('NAV_DIR_GOAL_GAIN', '30.0'))   # gain on vec[0:2] into the
                            # direction head. 1.0 = the old behaviour. See heads() for the
                            # measurement that motivated it; 30 was the value that reached
                            # human-level aim steadiness in the offline replay (conc 0.89).
JUMP_DAMP = 1.0             # 3.0 -> 1.0 on 2026-09-22 (HANDOFF 28.13): jumping is being re-enabled as a
                            # learnable skill. At 3.0 the agent jumped on 0.1 % of real-round decisions
                            # and never in KOTM's centre block, where a hop over the 2x2 hole is 5.7 u
                            # against 12.5 u round it; the human jumps on 1.8 % of ticks. Together with
                            # JUMP_TAKEOFF 0.4 -> 0.1, DISCRETE_ENT_COEF 0.002 -> 0.01 and the jump
                            # curriculum (terrain.JUMP_GOAL_P). FALL stays 25: falls will rise in
                            # training while it practises; the 8-round eval is the judge.
                            # Snapshot before: models/nav/nav_eval_align1_1536.pth (update 15,590).
                            # (superseded) 2.0 -> 3.0 on 2026-09-19 02:00, chasing the fall gap against the human
                            # baseline (section 3c: human 0.046 falls per 100 u and 0.11 jumps/s;
                            # agent 0.46-0.53 and, per map, islands 0.52 takeoffs/s with 59 % at a
                            # real gap, KOTM 0.17/s with only 32 % at a real gap). Stray takeoffs
                            # still cause most remaining falls. 3.0 takes the stray base rate from
                            # 0.7 % to 0.25 % while a real gap edge still fires at ~50 % per
                            # decision, which stays near-certain over the several decisions spent
                            # at an edge. Before-snapshot: models/nav/nav_both90_20260919_0158.pth
                            # Constant subtracted from the jump logit, the mirror of BRAKE_SUPPRESS.
                            # 2026-09-18: measured on King of the Marble, a takeoff ends in a fall
                            # 40.7 % of the time (islands 12.0 %) and 83 % of its takeoffs happen with
                            # NO gap prior active, i.e. no gap to cross. Segments with no takeoff
                            # arrive 87 %, with any takeoff 29 %, and 70 % of arrivals need no jump at
                            # all. The residual takeoffs are exploration noise from the Bernoulli jump
                            # head (the policy had already pushed itself below its own -3 bias), so a
                            # reward penalty is the wrong lever -- the implicit cost is already
                            # 0.407 x FALL = 4.07 per takeoff. Damping the logit takes the stray base
                            # rate 4.7 % -> 0.7 % while a real gap edge still fires at 73 % per
                            # decision (near-certain over the several decisions spent at an edge).
                            # Map-independent: nothing about King of the Marble is baked in.
BRAKE_SUPPRESS = 1.0   # 6.0 -> 1.0 on 2026-09-20 11:55. This penalises the brake logit WHILE THE GAP
                       # PRIOR IS ACTIVE, i.e. at edges. Measured on KOTM: 78 % of takeoffs are edge
                       # roll-offs rather than jumps, and 76 % of falls follow one, so braking was
                       # suppressed precisely where the falls happen. The agent brakes on 0.22 % of
                       # decisions against a human 14.3 %. The degenerate 'sit at the edge and brake'
                       # policy this guard was added for (2026-09-17) is now separately priced by
                       # BRAKE = 0.05 per decision and FALL = 10, so the suppression looks redundant.
                       # Not set to 0: a little suppression still discourages braking mid-gap.
                            # at gaps came after braking, 77 % without any jump)
BRAKE_ENABLED = True   # re-enabled 2026-09-19 03:05. Disabled long ago because an early policy
                       # learned to sit still and brake for free; braking now costs BRAKE = 0.1 per
                       # decision and is suppressed at gap edges (BRAKE_SUPPRESS), so that strategy
                       # cannot pay. Motivation: 39-43 % of remaining falls are now ROLL-OFFS (the
                       # marble drives off an edge without jumping) at 7.5-8.0 u/s, and the human
                       # baseline brakes on 14.3 % of ticks. Braking is the control for arriving at
                       # an edge too fast. Snapshot: models/nav/nav_before_brake_0305.pth
                       # ABORT if brake share > ~30 % of decisions or mean speed drops > 15 %.       # 2026-09-17: the brake (full-throttle anti-velocity kick, inherited from the old
                            # trainer) preceded 85 % of falls on the islands and was used 4.7x per segment;
                            # the direction head can decelerate by steering, so the action is switched off
                            # (logit pinned at -20). The head stays in the model for checkpoint compatibility.
VEC_GOAL_DIST, VEC_SPEED, VEC_ON_FLOOR = 2, 7, 8
VEC_GOAL_RAY_CLEAR, VEC_GOAL_RAY_BEYOND = 10 + 32, 10 + 33


class NavActorCritic(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, -0.5
    THROTTLE_LOG_STD_MIN, THROTTLE_LOG_STD_MAX = -3.0, -0.9
    # 0.15 -> 0.90 on 2026-09-20 15:10. throttle = FLOOR + (1-FLOOR)*sigmoid(thr), so this
    # forces throttle into [0.90, 1.0]. MEASURED CHAIN: acceleration scales almost linearly
    # with input magnitude (0.5 -> +1.66 u/s^2, 0.7 -> +3.07, 0.9 -> +5.52, 1.0 -> +6.36 at
    # 4-7 u/s), and at full throttle the agent accelerates nearly as well as the human
    # (+6.36 vs +6.86-7.34). But it runs at 0.85 throttle on KOTM and 0.94 on Islands, and
    # its acceleration at speed is HALF the human's (8-9 u/s: +3.64 vs +6.00).
    # The floor must sit ABOVE the 0.85 the policy currently chooses or it is nullified:
    # the policy can hit any target below the floor by lowering its sigmoid output.
    # A keyboard player is pinned at 1.0 and still arrives at gems at 7.37 u/s against the
    # agent 5.4, so the caution is not buying precision they need. If arrivals collapse,
    # low throttle IS load-bearing for this policy and the floor goes back.
    THROTTLE_FLOOR = 0.90
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
        # dir_head reads [hidden state, observation vector], NOT the hidden state alone.
        # The unit vector to the waypoint is vec[0:2]: exact and perfectly smooth. Routing it
        # through a GRU that rewrites 26 % of itself every decision made the commanded heading
        # inherit that churn (37 deg/decision against a human 0.4; zeroing h cut the median 5x).
        # See HANDOFF sections 8, 9, 9a for the measurements and for what was ruled out.
        self.dir_head = nn.Sequential(nn.Linear(HIDDEN + VEC_DIM, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
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

    @staticmethod
    def gap_prior(crop, vec):
        """(B,) 1.0 where a crossable gap lies just ahead along the line to the waypoint:
        the edge ray ends within JUMP_PRIOR_DIST u with a drop beyond it, AND the fine crop
        shows floor at about the current height within JUMP_LANDING_MAX u past the edge, AND
        the marble is on the floor and moving. Fixed, map-independent (rays + crop only)."""
        B = vec.shape[0]
        ux, uy = vec[:, 0], vec[:, 1]
        edge_u = vec[:, VEC_GOAL_RAY_CLEAR] * vec[:, VEC_GOAL_DIST] * 50.0
        at_edge = (edge_u < JUMP_PRIOR_DIST) & (vec[:, VEC_GOAL_RAY_CLEAR] < 0.999) & (vec[:, VEC_GOAL_RAY_BEYOND] < JUMP_PRIOR_DROP)
        ready = (vec[:, VEC_ON_FLOOR] > 0.5) & (vec[:, VEC_SPEED] * 20.0 > JUMP_PRIOR_SPEED)
        # sample the fine crop (channel 0 rel height /10, channel 1 present; 0.5 u cells, centre 15.5)
        ds = edge_u.unsqueeze(1) + torch.arange(0.5, JUMP_LANDING_MAX + 0.01, 0.5, device=vec.device).unsqueeze(0)   # (B, K)
        ix = (15.5 + ux.unsqueeze(1) * ds / 0.5).round().long().clamp(0, 31)
        iy = (15.5 + uy.unsqueeze(1) * ds / 0.5).round().long().clamp(0, 31)
        bi = torch.arange(B, device=vec.device).unsqueeze(1).expand_as(ix)
        present = crop[:, 1][bi, iy, ix] > 0.5
        level = crop[:, 0][bi, iy, ix].abs() < JUMP_LANDING_DZ
        landing = (present & level).any(dim=1)
        return (at_edge & ready & landing).float()

    def heads(self, h, vec, crop=None):
        # DIR_GOAL_GAIN: amplify the GOAL BEARING on the way into the direction head.
        # Measured 2026-09-21 (HANDOFF 21a): the head was 65x more sensitive to the hidden state
        # than to vec[0:2], the exact smooth unit vector to the gem sitting right there in its
        # input, because training grew the hidden columns to 2.10x init and shrank the goal columns
        # to 0.83x. The command therefore pointed at the gem ON AVERAGE but scattered +-60-70 deg
        # around it, and useful thrust is the cosine of that scatter: 0.57 against a human 0.91.
        # An inference-only test of this gain on real rounds: aim concentration 0.46 -> 0.64 and
        # mean speed 8.29 -> 9.44 u/s (+14 %) with no training at all.
        # Applied as a GAIN on the input rather than by rescaling the weights, so it is structural
        # and visible: to switch it off, gradient descent would have to shrink those columns by the
        # same factor, which is detectable (see the effective-norm log line in train_nav).
        vec_in = vec
        if DIR_GOAL_GAIN != 1.0:
            vec_in = vec.clone()
            vec_in[..., 0:2] = vec_in[..., 0:2] * DIR_GOAL_GAIN
        mean_xy = self.dir_head(torch.cat([h, vec_in], dim=-1))
        thr = self.throttle_head(h).squeeze(-1)
        gp = self.gap_prior(crop, vec) if crop is not None else torch.zeros_like(thr)
        jump = torch.clamp(self.jump_head(h).squeeze(-1) - JUMP_DAMP + gp * JUMP_PRIOR, -7.0, 3.0)
        brake = torch.clamp(self.brake_head(h).squeeze(-1) - gp * BRAKE_SUPPRESS, -7.0, 3.0)
        if not BRAKE_ENABLED:
            brake = torch.full_like(brake, -20.0)
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
        mean_xy, thr, jump, brake, vn = self.heads(h1, vec, crop)
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
                'mean_dir': mean_xy,          # pre-sampling mean: TURN_COST is charged on THIS,
                                              # so it prices steering and not exploration noise
                'value': self.value_from_norm(vn), 'h_next': h1}

    # ------------------------------------------------------------------ training
    def evaluate_seq(self, crops, vecs, h0, actions, resets):
        """crops (T,B,6,32,32), vecs (T,B,48), h0 (B,H), actions (T,B,5),
        resets (T,B) float: 1 where the hidden state must be zeroed BEFORE step t
        (first step of a new segment). Returns logp (T,B), continuous entropy (T,B),
        discrete entropy (T,B), value_norm (T,B)."""
        T, B = actions.shape[:2]
        h = h0
        logps, ents, ents_d, vns = [], [], [], []
        for t in range(T):
            h = h * (1.0 - resets[t]).unsqueeze(-1)
            h = self.core(crops[t], vecs[t], h)
            mean_xy, thr, jump, brake, vn = self.heads(h, vecs[t], crops[t])
            d_dir, d_thr, d_jump, d_brake = self.dists(mean_xy, thr, jump, brake)
            a = actions[t]
            logp = d_dir.log_prob(a[:, 0:2]).sum(-1) + d_thr.log_prob(a[:, 2]) + d_jump.log_prob(a[:, 3]) + d_brake.log_prob(a[:, 4])
            # Entropy is returned SPLIT (2026-09-20). Lumping all four distributions together and
            # paying one coefficient on the sum let the entropy bonus buy its entropy from the
            # CHEAPEST head instead of the useful one: Bernoulli entropy rises very steeply from
            # p near 0, so raising the coefficient to widen steering instead randomised braking,
            # which went 0.34 % -> 17.25 % of decisions in 90 updates while direction std moved
            # 0.18 -> 0.25. Speed fell 4.9 -> 4.7 and arrivals 86 % -> 83 %. The caller now prices
            # the continuous heads (where exploration is wanted) and the discrete ones (which have
            # their own deliberate priors, JUMP_DAMP and BRAKE_SUPPRESS) separately.
            ent_cont = d_dir.entropy().sum(-1) + d_thr.entropy()
            ent_disc = d_jump.entropy() + d_brake.entropy()
            logps.append(logp); ents.append(ent_cont); ents_d.append(ent_disc); vns.append(vn)
        return torch.stack(logps), torch.stack(ents), torch.stack(ents_d), torch.stack(vns)

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


from nav.joystick import action_to_joystick   # noqa: E402,F401  (moved to a torch-free module; still importable from here)
