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

from nav.obs import VEC_DIM, VEC_GAP, VEC_POW, VEC_NEXT, WAYPOINT_DIST_SCALE, VEL_SCALE, SS_REDIRECT_U, SS_KEY_LAG_S, SS_FIRE_U, VEC_USE, USE_T_SCALE
from terrain_obs import RAY_RANGE
from nav.terrain import CROP_SHAPE

HIDDEN = 256
ACTION_DIM = 6              # dx, dy, throttle, jump, brake, use (2026-10-02, POWERUP_PLAN phase 4, log 40)
USE_PRIOR = 11.0            # 10 -> 11 on 2026-10-03 20:00 (log 40.17): approved uses sampled 50 % instead of 27 % (the head
                            # was at about -11 there; uses are revenue-neutral now and the data for learning them is the
                            # bottleneck: 40 % of pickups hold a Super Speed, 1 fire a round).
                            # 2026-10-02 23:30 (log 40.2): fixed bonus on the use logit where physics approves a use, the
                            # mirror of JUMP_PRIOR: random 0.1 % uses only ever taught 'do not' (the head went to its
                            # floor within 20 minutes) and never land in a situation where a use pays. Approved:
                            # Super Jump / blast at a lip within USE_LIP_U with a gap the plain jump cannot cross but the
                            # boosted flight can; Super Speed on the floor, heading within 30 deg of the waypoint, >= USE_RUN_U
                            # away, the edge ray to it clear. Floor -9 + 10 = +1: sampled 73 %, fires in deterministic
                            # play; the head can cancel it (up to +3 -> -6) where it hurts (it could not until 40.8, see
                            # USE_PRIOR_FLOOR). Rays, gap block and the
                            # powerup block only: map-independent.
USE_LIP_U = 2.5             # u to the lip along the heading
USE_RUN_U = 8.0             # u to the waypoint for a Super Speed run
USE_RUN_COS = 0.866         # heading within 30 deg of the waypoint bearing
USE_SJ_HANG = 2.0           # s in the air after a Super Jump (+20 u/s, g 20); a blast hangs sqrt(meter) s
USE_RANGE_K = 0.8           # the flight must cover lip + gap with this margin
USE_PRIOR_FLOOR = -1.0      # -4 -> -1 on 2026-10-03 21:40 (log 40.19): the operator's demo fires a Super Speed at EVERY
                            # chance (20 a round, 0.1-0.2 s after a pickup, a 100-130 deg turn to 16-19 u/s); the head had
                            # cancelled approved uses to the -4 floor before learning the control after the kick, so
                            # approved uses now stay at >= 27 % sampled (never deterministic until learned).
                            # 2026-10-03 13:10 (log 40.8, operator: "Apply use floor fix"): the lowest use logit the head
                            # can set where the prior approves (1.8 % a decision, no fire in deterministic play). Until
                            # then the prior was added AFTER the -9 clamp, so an approved use never fell below +1 (73 %
                            # sampled, always in deterministic play) and the head could not cancel it where it hurts:
                            # 8-round gates 132-138 all night against 160.1 without uses (log 40.3-40.7). The floor stays
                            # above the clamp's -9 so the approved situations keep being explored. Elsewhere unchanged.
USE_SS_ONLY = True          # 2026-10-04 (log 40.26): the first curriculum experiment isolates Super Speed: the approval
                            # mask (use_prior) keeps only its branch, so no Super Jump / blast / Mega use can be sampled
USE_EPS = 0.2               # 40.26: inside the approval mask a use is sampled with p = USE_EPS + (1 - USE_EPS) sigmoid(logit):
                            # a differentiable exploration floor (the hard floor of 40.8-40.22 had ZERO gradient below it,
                            # so a use that paid could never raise its own probability); PPO scores this exact mixture;
                            # deterministic play fires only where the LEARNED part sigmoid(logit) > 0.5. Outside the mask
                            # p = 0. USE_PRIOR / USE_PRIOR_FLOOR are no longer used.
USE_LOGIT_CAP = 12.0        # 40.26: soft bound on the use logit (cap tanh(raw / cap)) instead of a clamp: no dead zone
USE_BIAS_V10 = -1.0         # 40.26: a fresh use head starts unsaturated (learned p 0.27, sampled 0.41 where approved)
SS_RES_RECENT_S = 2.0       # 40.26: the Super Speed residual acts while one is held, a kick is pending, or for this long after
SS_RES_SCALE = 3.0          # 40.30: the residual's output x3, so each Adam step moves it 3x as far (it learned too slowly)
SS_APPROACH_U = 18.0        # preparation may start about two seconds before a turn; the fire mask is unchanged
                            # Super Speed held, on the floor, with a turn ahead or the existing redirect flag.
                            # Preparation need not wait for current-speed approval; firing still requires that approval.
SS_RES_APPROACH = False     # 2026-10-05 (log 40.51, operator-approved): the preparation term above is OFF. In the points run it
                            # turned harmful between 29250 (163.1 points, 0.50 falls/round) and 29275 (147.1, 4.62); 29375 with
                            # uses disabled, so preparation alone, 149.3 / 3.69; 29375 without it 162.8 / 0.50 (16 rounds each,
                            # GOAL_175_2026-10-05.md). True restores the morning gate: pending | recent | approach.
AUX_DIM = 10 + 37           # 40.40: the use head and the critic read, beside the frozen hidden state, the current observation's
                            # goal block (vec[0:10]: bearing, distance, dz, velocity, speed, on floor, airborne) and the whole
                            # powerup block (vec[VEC_POW:]: held type, meter, items, the kick aim / result / approvals, the
                            # use state). With the base frozen the hidden state carries NONE of this (28897's encoder
                            # columns for the powerup block are exactly zero), so the learned use preference and the value
                            # could not tell a held Super Speed from an empty slot; the mask did the knowing.
SS_RES_HELD = False         # 40.33: False = the residual acts only while a kick is pending or within SS_RES_RECENT_S of one.
                            # True (40.26-40.32) also gated it on "Super Speed held", which is ~60 % of a round once the
                            # policy stops firing: at 29585 it cost 3.6 points a round (141.5 vs 145.1 with it zeroed)
FREEZE_BASE = False         # 2026-10-05 22:15 (log 40.58, operator: "retrain the model so it knows the current map"): the
                            # navigator relearns on the corrected terrain map (its edges were ~0.5 u off; see
                            # logs/nav/goal175/terrain_map_old_vs_new_20261005.png), at ppo_recurrent.LR 1e-4.
                            # Before: True (40.33): train only use_head, ss_res and value_head on top of the frozen 28897 navigator (the
                            # plan's "small powerup residual + learnable use head"). 40.26-40.32 trained everything and the
                            # drills wore the navigator down: deterministic rounds 159.0 (28897) -> 149.0 (29235) -> 141.5
                            # (29585), slower (8.38 -> 7.85 u/s) with more falls; no Super Speed gain to show for it
SS_LR_MULT = 10.0           # 40.33: with the navigator frozen the use head and residual learn at LR x this (their own Adam
                            # group); at x1 the first 25 updates moved the policy by KL 0.000-0.001 per update
BRAKE_EPS_POST = 0.0      # 40.32: OFF again (0.15 for 40 min): forcing the brake for the first 4 / 8 decisions of the
                            # eval drills gave no first-pass gain and more falls (22 / 24 % vs 19 %), and the random brakes
                            # raised the training drills' falls 17-20 -> 22-23 %; the mechanism stays for later stages.
                            # 40.31: for SS_RES_RECENT_S after a Super Speed fire the brake (full thrust exactly against the
                            # velocity: it keeps the kick's line, the operator's backward key after 11 of 20 kicks) is
                            # sampled with p = BRAKE_EPS_POST + (1 - BRAKE_EPS_POST) sigmoid(logit), a differentiable floor
                            # PPO scores exactly; the navigator brakes ~150 deg off the line instead (bending it off the
                            # gem) and its brake logit sits at -7 (0.09 %), so straight braking was never tried.
                            # Deterministic play brakes only where the learned part sigmoid(logit) > 0.5.
STD_POST_MULT = 1.0         # 40.32: OFF again (2.5 for 1 h 20 min: no deterministic gain, more falls in the sampled drills)
                            # 40.30: steering noise x2.5 for SS_RES_RECENT_S after a fire: right after a kick the navigator
                            # brakes ~145 deg against its velocity and std 0.22 only tried +-12 deg around that, so a
                            # straight pass through the gem was hardly ever sampled. The entropy the controller sees
                            # excludes this boost; deterministic play is unaffected.
USE_BIAS = -11.0            # 2026-10-03 (log 40.11, fresh head from 28897): approved -11 + 10 = -1 (27 % sampled, never in
                            # deterministic play until the head learns a use is worth it), elsewhere the -9 floor. Was -3
                            # (4.7 %; with the prior +7 = every approved use fired until the head cancelled it).
                            # Earlier meaning: initial use logit (4.7 %): the policy learns from the game's points when to fire the held
                            # powerup (or the blast meter when nothing is held); the worker turns the bit into the
                            # bridge's use / blast words, a Super Speed fires along the commanded direction

# Jump prior (2026-09-17): a fixed, non-learned bonus on the jump logit when the edge ray
# toward the waypoint shows a drop within JUMP_PRIOR_DIST u, the marble is on the floor and
# moving. Without it the learned jump logit sat at its -7 clamp (0.1 %) after the flat maps,
# so a jump at a gap edge was never sampled and could never be learned. The learned head can
# cancel the bonus where jumping is bad. Map-independent: it only reads the rays.
JUMP_PRIOR = 8.5            # 6.0 -> 8.5 on 2026-09-24 (HANDOFF 28.34): with the head at ~-7 and JUMP_DAMP 1 the old bonus
                            # left the logit at ~-1 (27 % sampled, NEVER deterministic). 8.5 puts an approved gap at
                            # ~+0.5 (62 % sampled, jumps deterministically) while the flag is now physics-approved with a
                            # 1 u landing margin and an approved fall is priced at its true cost (waypoints 28.34).
                            # (superseded) logit bonus (-7 -> -1 = 27 % per decision at a gap edge)
JUMP_PRIOR_DIST = 2.5       # edge within this many u along the line to the waypoint
JUMP_PRIOR_DROP = -0.15     # ray "beyond" value below this = a drop of > 1.5 u or void
JUMP_EXPLORE_P = float(os.environ.get('NAV_JUMP_EXPLORE_P', '0.01'))   # 2026-10-07: sampling-time floor on the jump probability where it is safe (see heads); 0 = off.
                            # NAV_JUMP_EXPLORE_P / NAV_JUMP_EXPLORE_P_RISE (10-08 13:30): evaluation-only overrides, so a check can run the pure policy.
                            # 0.03 -> 0.01 on 10-08 00:05 (operator, "reason 1"): most random jumps mid-floor earn nothing and
                            # pull the head down; the exploration moves to where a rise is ahead:
JUMP_EXPLORE_BONUS = float(os.environ.get('NAV_JUMP_EXPLORE_BONUS', '0.0'))   # 1.0 -> 0.0 on 10-08 22:00 (operator): the moving case is over-learned (40 % of decisions above threshold); only the forced stall samples (0.03) stay   # 3.0 -> 1.0 on 10-08 21:40 (anneal step 1: the policy jumps on 40 % of decisions by itself and over-jumps at clusters, operator)   # 10-08 13:40: sampling-only LOGIT bonus where a rise (wall /
                            # box side) is within JUMP_EXPLORE_RISE_U along the heading (replaces the 0.15/0.30 mixture, see heads);
                            # anneal 3 -> 2 -> 1 -> 0 at the checks; evaluations run 0.
JUMP_EXPLORE_P_RISE = 0.0   # (retired 10-08 13:40; the mixture at rises starved the head, see JUMP_EXPLORE_BONUS)
JUMP_FORCE_P = float(os.environ.get('NAV_JUMP_FORCE_P', '0.03'))   # 0.10 -> 0.03 (10-08 21:40) -> 0.0 (10-09 00:30) -> back to 0.03 (10-09 03:10): at 0 the stall logits collapsed within 2.5 h (75 % -> 3 % above 0, held-out 108.9 -> 101.8, breaks 3 -> 13): the stopped case still needs the floor   # 10-08 15:55: forced jump sample probability where slow at a rise (see act); evaluations 0
JUMP_FORCE_SPEED = 2.0      # u/s: 'slow' for the forced samples (the stopped-at-a-box case)
JUMP_DET_THRESHOLD = float(os.environ.get('NAV_JUMP_DET_THRESHOLD', '0.5'))   # 10-08 22:40: deterministic-play jump threshold (evaluation/viewing knob, operator: 'simulate jumping less'); 0.5 = unchanged
                            # 0.15 -> 0.30 on 10-08 07:25 (overnight mandate): the held-out score sat at 46-47 for three
                            # checks and the head's rise slowed; more positive jump samples at rises, KOTM unaffected
                            # (the nudge cannot fire on 0 % of its floor)
JUMP_EXPLORE_RISE_U = 2.5   # u
JUMP_EXPLORE_RISE_DZ = 0.03 # edge_dz (/ DZ_SCALE) above this = a rise of 0.3 u or more
JUMP_EXPLORE_CLEAR_U = 10.0 # u: no edge with a drop beyond it within this along any of the 16 ray headings. 4.0 until
                            # 10-07 21:40: KOTM fell 166.5 -> 160.3 -> 152.2 in 4 h with training falls 3 -> 8 a round (the
                            # nudge's jumps near holes); at 10 u the nudge almost never fires on KOTM (holes everywhere)
                            # and always fires on the flat cluster maps (rays run the full 20 u). Map-agnostic rule.
JUMP_EXPLORE_DROP = -0.3    # edge_dz (/ DZ_SCALE) below this counts as a dangerous drop for the nudge (void reads -1; a 0.5-1.5 u box step reads -0.05..-0.15)
JUMP_FORCE_CLEAR_U = 3.0    # u: the FORCED STALL samples' own clearance (2026-10-09, Phase R part 4). A hop from under
                            # JUMP_FORCE_SPEED (2 u/s) lands within ~1.5 u, so a drop 3 u away cannot matter; the 10 u
                            # rule above kept every stall beside a drop untrained (ExampleMission's ledges, log 40.75).
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
        self.value_head = nn.Sequential(nn.Linear(HIDDEN + AUX_DIM, 64), nn.ReLU(), nn.Linear(64, 1))   # 40.40: [h, aux]
        self.log_std = nn.Parameter(torch.full((1,), -1.5))
        self.throttle_log_std = nn.Parameter(torch.full((1,), -1.0))
        # registered LAST so the parameter order of older checkpoints is unchanged and their Adam state loads
        self.use_head = nn.Sequential(nn.Linear(HIDDEN + AUX_DIM, 64), nn.ReLU(), nn.Linear(64, 1))   # 40.40: [h, aux]
        nn.init.constant_(self.use_head[-1].bias, USE_BIAS_V10)
        # 40.26: a Super-Speed-conditioned RESIDUAL on the steering (pre-tanh) and the brake logit, gated on (held, pending,
        # or a kick within SS_RES_RECENT_S): the control after a kick is learned here while ordinary driving, the base
        # heads, starts unchanged (zero init). Registered after use_head so older checkpoints keep their parameter order.
        self.ss_res = nn.Sequential(nn.Linear(HIDDEN + VEC_DIM, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.ss_res[-1].weight); nn.init.zeros_(self.ss_res[-1].bias)
        self.register_buffer('value_mean', torch.zeros(1))
        self.register_buffer('value_std', torch.ones(1))
        self.register_buffer('value_sq_mean', torch.ones(1))
        self.register_buffer('value_stats_initialized', torch.zeros(1))

    def load_state_dict(self, sd, strict=True):
        """A checkpoint from before the use head (ACTION_DIM 5) loads with the head at its fresh init."""
        own = self.state_dict()
        missing = [k for k in own if k not in sd]
        if missing and all(k.startswith('use_head.') or k.startswith('ss_res.') for k in missing):
            sd = dict(sd); sd.update({k: own[k] for k in missing})
        # 40.40: a checkpoint whose value_head / use_head read only the hidden state (HIDDEN columns): keep those weights
        # and add ZERO columns for the aux inputs, so the critic is unchanged until training moves them; a use head of
        # the old width is dropped (fresh init: the experiment restarts the use decision anyway)
        sd = dict(sd)
        for head in ('value_head', 'use_head'):
            k = f'{head}.0.weight'
            if k in sd and sd[k].shape[1] != own[k].shape[1]:
                if head == 'value_head' and sd[k].shape[1] < own[k].shape[1]:
                    w = torch.zeros_like(own[k]); w[:, :sd[k].shape[1]] = sd[k]; sd[k] = w
                else:
                    for kk in [q for q in sd if q.startswith(head + '.')]:
                        sd[kk] = own[kk]
        return super().load_state_dict(sd, strict)

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
        landing_short = (present & level).any(dim=1)          # the old short-hop test (<= JUMP_LANDING_MAX u)
        # 2026-09-23 (HANDOFF 28.27): OR the measured test: the gap along the marble's own heading is
        # within jump_range(current speed) per nav/physics (obs.py fills vec[VEC_GAP + 2]). This is what
        # lets the prior fire for a 7 u hole at 8 u/s and NOT at 5 u/s.
        # 2026-09-24 16:10 (HANDOFF 28.35): the crop short-hop test is DROPPED from the prior. It fired at every
        # lip with floor within 4.5 u (block-to-ring drop-offs included): 45 firings per instance-minute, 5 % of
        # decisions, so the head learned to ignore the prior entirely (P(jump | prior on) = 1 %). Only the
        # physics-approved flag (run-up + gap within jump_range(speed) with a 1 u margin, over floor) counts.
        landing = vec[:, VEC_GAP + 2] > 0.5
        return (at_edge & ready & landing).float()

    @staticmethod
    def use_prior(vec):
        """(B,) 1.0 where a powerup use is physics-approved (see USE_PRIOR). Fixed, map-independent."""
        held = vec[:, VEC_POW:VEC_POW + 7]
        held_sj, held_ss = held[:, 0] > 0.5, held[:, 1] > 0.5
        held_any = held.sum(-1) > 0.5
        meter, special = vec[:, VEC_POW + 7], vec[:, VEC_POW + 8] > 0.5
        on_floor = vec[:, VEC_ON_FLOOR] > 0.5
        speed = vec[:, VEC_SPEED] * VEL_SCALE
        lip, gap = vec[:, VEC_GAP] * RAY_RANGE, vec[:, VEC_GAP + 1] * RAY_RANGE
        has_gap = (vec[:, VEC_GAP] < 0.999) & (vec[:, VEC_GAP + 1] < 0.999)
        plain_ok = vec[:, VEC_GAP + 2] > 0.5
        at_lip = has_gap & (lip < USE_LIP_U) & on_floor & (speed > JUMP_PRIOR_SPEED) & ~plain_ok
        sj = held_sj & at_lip & (lip + gap < speed * USE_SJ_HANG * USE_RANGE_K)
        hang = torch.where(special, torch.ones_like(meter), meter.clamp(min=0.0).sqrt())
        bl = (~held_any) & ((meter >= 0.2) | special) & at_lip & (lip + gap < speed * hang * USE_RANGE_K)
        vx, vy = vec[:, 4] * VEL_SCALE, vec[:, 5] * VEL_SCALE
        cos = (vx * vec[:, 0] + vy * vec[:, 1]) / speed.clamp(min=1e-6)
        dist = vec[:, VEC_GOAL_DIST] * WAYPOINT_DIST_SCALE
        clear = vec[:, VEC_GOAL_RAY_CLEAR] >= 0.999
        run_clear = vec[:, VEC_POW + 26] > 0.5        # obs.py: the kick aimed at the waypoint is survivable (stop room, or the next gem on the line)
        # 40.15: no heading test: the kick is aimed so the resultant points at the waypoint, so this covers the turn
        # right after a pickup (the redirect) and the straight run alike; the floor check replaces the edge ray
        ss = held_ss & on_floor & (speed > 1.0) & (dist >= USE_RUN_U) & run_clear
        # V9 (log 40.12): the REDIRECT at the pickup: within SS_REDIRECT_U of the waypoint, the kick aimed by the
        # observation turns the velocity onto the line to the next waypoint with floor all the way (obs.py)
        # the pre-pickup redirect branch (40.12-40.14) is gone: a kick before the pickup missed the gem 40 % of the time
        if USE_SS_ONLY:
            return ss.float()                     # 40.26: Super Speed isolated for the curriculum experiment
        return (sj | bl | ss).float()

    @staticmethod
    def ss_gate(vec):
        """(B,) 1.0 while a kick is pending or one fired within SS_RES_RECENT_S (V10 use state); also on the approach to a
        turn if SS_RES_APPROACH, and while a Super Speed is held if SS_RES_HELD."""
        pending = vec[:, VEC_USE + 2] > 0.5
        recent = vec[:, VEC_USE + 5] < (SS_RES_RECENT_S / USE_T_SCALE)
        g = pending | recent
        if SS_RES_APPROACH:
            # Preparation must be possible before the current-speed forecast approves firing.
            # The next target's coordinates relative to this target describe the upcoming turn.
            goal_xy = vec[:, 0:2] * vec[:, 2:3]
            outgoing = vec[:, VEC_NEXT:VEC_NEXT + 2] * vec[:, VEC_NEXT + 2:VEC_NEXT + 3] - goal_xy
            length = outgoing.norm(dim=-1).clamp_min(1e-6)
            turn_cos = (outgoing * vec[:, 0:2]).sum(-1) / length
            turn_ahead = (vec[:, VEC_NEXT + 4] > .5) & (turn_cos < .866)
            distance = vec[:, VEC_GOAL_DIST] * WAYPOINT_DIST_SCALE
            approach = ((vec[:, VEC_POW + 1] > .5) & (vec[:, VEC_ON_FLOOR] > .5)
                        & (turn_ahead | (vec[:, VEC_POW + 27] > .5)) & (distance <= SS_APPROACH_U))
            g = g | approach
        if SS_RES_HELD:
            g = g | (vec[:, VEC_POW + 1] > 0.5)
        return g.float()

    TRAINABLE_WHEN_FROZEN = ('use_head.', 'ss_res.', 'value_head.')

    def freeze_base(self):
        """40.33 FREEZE_BASE: only the Super Speed parts and the critic's head learn. Returns the trainable count."""
        n = 0
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith(self.TRAINABLE_WHEN_FROZEN)
            n += p.numel() if p.requires_grad else 0
        return n

    @staticmethod
    def aux(vec):
        """40.40: the observation slice the use head and the critic read directly (AUX_DIM)."""
        return torch.cat([vec[:, 0:10], vec[:, VEC_POW:VEC_POW + 37]], dim=-1)

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
        x_dir = torch.cat([h, vec_in], dim=-1)
        g = self.ss_gate(vec)                                    # 40.26: the Super Speed residual's gate
        res = self.ss_res(x_dir) * (SS_RES_SCALE * g).unsqueeze(-1)   # zero at init and wherever the gate is off
        g_post = (vec[:, VEC_USE + 5] < (SS_RES_RECENT_S / USE_T_SCALE)).float()   # 40.30: just after a kick
        # 40.29: the residual is added AFTER the tanh: in the states after a kick the base steering is saturated (|pre-tanh|
        # median 1.5, 39 % above 2, tanh slope < 0.07 there), which starved a pre-tanh residual of gradient; the Gaussian
        # mean may leave [-1, 1] (action_to_joystick normalises the sampled direction)
        mean_xy = torch.tanh(self.dir_head[:3](x_dir)) + res[..., 0:2]
        thr = self.throttle_head(h).squeeze(-1)
        gp = self.gap_prior(crop, vec) if crop is not None else torch.zeros_like(thr)
        # 28.35: the prior is added AFTER the clamp, so the head cannot cancel it by going more negative (it
        # had reached ~-12 against a +8.5 prior). An approved gap now sits at >= -7 + 8.5 = +1.5: ~82 % sampled
        # in training and a jump in deterministic play. The policy's job is the approach, not the decision.
        jump_raw = self.jump_head(h).squeeze(-1) - JUMP_DAMP
        self.last_jump_raw = jump_raw.detach()          # 10-08 15:10: pre-clamp head output, read by real_run's trace (diagnostic)
        # 10-08 15:55 (log 40.67): the LOWER clamp is gone. Check 8's trace showed the raw head below -7 on 88 % of decisions
        # and 99 % of stalls (median -38 stopped at a box side); torch.clamp passes no gradient there, so no exploration
        # of any form could teach the head the stopped-at-a-box case. Below -7 the sampled probability is < 0.001 either
        # way and deterministic play (> 0.5) is unchanged, so only the gradient changes. The upper clamp stays.
        jump = torch.clamp(jump_raw, max=3.0) + gp * JUMP_PRIOR
        # 2026-10-07 (operator: "jump nudge is fine"; GENERALIZATION_PLAN skill 4, the discovery run): the head has learned
        # 'never' (~-7, one press in a thousand), so on a map where the gems sit on boxes the jump would almost never be
        # tried. JUMP_EXPLORE_P floors the SAMPLING probability of a jump wherever it is safe: on the floor, and no edge ray
        # ends within JUMP_EXPLORE_CLEAR_U with a drop beyond it. The floor is part of the distribution (act and evaluate_seq
        # see the same logits), so PPO's ratio stays exact. Deterministic play (sigmoid > 0.5) is untouched.
        self.last_force_mask = None
        if JUMP_EXPLORE_P > 0.0 or JUMP_EXPLORE_BONUS != 0.0 or JUMP_FORCE_P > 0.0:
            rays = vec[:, 10:10 + 32].reshape(-1, 16, 2)                      # (B, 16, [edge_dist/RAY_RANGE, edge_dz])
            near_drop = ((rays[..., 0] < JUMP_EXPLORE_CLEAR_U / 20.0) & (rays[..., 1] < JUMP_EXPLORE_DROP)).any(dim=1)   # deep drops only: a box step down is fine
            safe = (vec[:, VEC_ON_FLOOR] > 0.5) & ~near_drop
            # 21:40 FIX (check 2, log 40.64): the first version floored the LOGIT with torch.maximum, which passes no
            # gradient to the head while the head sits below the floor, so the nudged jumps could never teach it (4 h
            # flat). Now the mixture form the brake floor uses: p = p_learn + eps (1 - p_learn) where safe, returned as
            # the exact logit, so the gradient always flows through p_learn.
            # 10-08 00:05 (operator: "do reason 1"): explore the jump mostly where it can pay. A RISE ahead (edge_dz > 0:
            # a wall or a box side) within JUMP_EXPLORE_RISE_U along the heading (velocity, or the goal direction when slow)
            # gets JUMP_EXPLORE_P_RISE; the rest of the safe floor gets JUMP_EXPLORE_P. Rays are 16 fixed headings from +x
            # every 22.5 deg: the heading's ray and its two neighbours are checked. Exploration only: nothing deterministic.
            hx = torch.where(vec[:, VEC_SPEED] * 20.0 > 1.0, vec[:, 4], vec[:, 0])
            hy = torch.where(vec[:, VEC_SPEED] * 20.0 > 1.0, vec[:, 5], vec[:, 1])
            ridx = torch.round(torch.atan2(hy, hx) / (2.0 * math.pi / 16)).long() % 16
            ar = torch.arange(rays.shape[0], device=rays.device)
            rise = torch.zeros_like(safe)
            for off in (-1, 0, 1):
                k = (ridx + off) % 16
                rise = rise | ((rays[ar, k, 0] < JUMP_EXPLORE_RISE_U / 20.0) & (rays[ar, k, 1] > JUMP_EXPLORE_RISE_DZ))
            # 10-08 13:40 (log 40.66): the MIXTURE form starves the head. With p = p_learn + eps (1 - p_learn) the gradient of
            # log p w.r.t. the head logit is (1 - eps) p_learn (1 - p_learn) / p: 0.36 at eps 0.03 but 0.04 at eps 0.30, so the
            # more exploration, the less the head learns, and after 10 h the pure policy still jumped on 0.2 % of decisions
            # (every held-out gain was the exploration sampled inside the stuck-breaker's detours). Now a LOGIT BONUS at rises,
            # sampling only: p = sigmoid(logit + JUMP_EXPLORE_BONUS) there, gradient (1 - p) = 0.73 at a +3 bonus. The bonus is
            # annealed by hand at the checks (3 -> 2 -> 1 -> 0) as the pure policy's jump rate at box sides rises; NAV_JUMP_EXPLORE_BONUS
            # overrides it (evaluations run it at 0, so the checks measure the policy, breaker detours included). The plain-floor
            # mixture (JUMP_EXPLORE_P) stays tiny; deterministic play reads the raw head, never the bonus.
            jump = jump + (rise & safe).float() * JUMP_EXPLORE_BONUS
            near_drop_stall = ((rays[..., 0] < JUMP_FORCE_CLEAR_U / 20.0) & (rays[..., 1] < JUMP_EXPLORE_DROP)).any(dim=1)
            safe_stall = (vec[:, VEC_ON_FLOOR] > 0.5) & ~near_drop_stall
            self.last_force_mask = (rise & safe_stall & (vec[:, VEC_SPEED] * 20.0 < JUMP_FORCE_SPEED)).detach()
            if JUMP_EXPLORE_P > 0.0:
                # only the plain-floor rows go through the mixture; the others keep the raw logit (the mixture's 1e-6 clamp
                # turned a -31 logit into a constant and zeroed the gradient of the forced samples, 10-08 16:05)
                mix = safe & ~rise
                pj_learn = torch.sigmoid(jump)
                pj = (pj_learn + JUMP_EXPLORE_P * (1.0 - pj_learn)).clamp(1e-6, 1.0 - 1e-6)
                jump = torch.where(mix, torch.log(pj) - torch.log1p(-pj), jump)
        brake_l = torch.clamp(self.brake_head(h).squeeze(-1) - gp * BRAKE_SUPPRESS, -7.0, 3.0) + res[..., 2]
        if not BRAKE_ENABLED:
            brake_l = torch.full_like(brake_l, -20.0)
        # 40.31: the post-kick brake floor: p = p_learn + eps (1 - p_learn) just after a kick, p_learn elsewhere (unchanged)
        pb_learn = torch.sigmoid(brake_l)
        pb = (pb_learn + g_post * BRAKE_EPS_POST * (1.0 - pb_learn)).clamp(1e-6, 1.0 - 1e-6)
        brake = torch.log(pb) - torch.log1p(-pb)
        brake_det = (pb_learn > 0.5).float()
        # 40.26: the physics approval is the action MASK; inside it the use probability is the differentiable mixture
        # USE_EPS + (1 - USE_EPS) sigmoid(logit) (exploration that never stops the gradient), returned as the exact logit
        # of that probability so PPO's ratio uses the distribution actually sampled; outside the mask p = 0 (clamped).
        up = self.use_prior(vec)
        h_aux = torch.cat([h, self.aux(vec)], dim=-1)          # 40.40
        raw = self.use_head(h_aux).squeeze(-1)
        p_learn = torch.sigmoid(USE_LOGIT_CAP * torch.tanh(raw / USE_LOGIT_CAP))
        p = (up * (USE_EPS + (1.0 - USE_EPS) * p_learn)).clamp(1e-6, 1.0 - 1e-6)
        use = torch.log(p) - torch.log1p(-p)
        use_det = up * (p_learn > 0.5).float()                # deterministic play: only where the head has learned it
        vn = self.value_head(h_aux).squeeze(-1)
        return mean_xy, thr, jump, brake, use, vn, use_det, 1.0 + (STD_POST_MULT - 1.0) * g_post, brake_det

    def dists(self, mean_xy, thr, jump, brake, use, std_mult=None):
        # The tanh output is the Gaussian mean directly (NOT normalized to the unit circle:
        # normalizing a near-zero fresh output made the direction, and the PPO ratio,
        # chaotic). action_to_joystick normalizes the sampled direction.
        std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean_xy)
        if std_mult is not None:
            std = std * std_mult.unsqueeze(-1)          # 40.30: wider steering noise just after a kick
        d_dir = torch.distributions.Normal(mean_xy, std)
        d_thr = torch.distributions.Normal(thr, torch.clamp(self.throttle_log_std, self.THROTTLE_LOG_STD_MIN, self.THROTTLE_LOG_STD_MAX).exp())
        d_jump = torch.distributions.Bernoulli(logits=jump)
        d_brake = torch.distributions.Bernoulli(logits=brake)
        d_use = torch.distributions.Bernoulli(logits=use)
        return d_dir, d_thr, d_jump, d_brake, d_use

    def value_from_norm(self, vn):
        return vn * self.value_std + self.value_mean

    # ------------------------------------------------------------------ acting
    JUMP_ON_PRIOR = os.environ.get('NAV_JUMP_ON_PRIOR', '0') == '1'   # 2026-09-24 (HANDOFF 28.33) inference-only test:
                                  # deterministic runs jump whenever the gap prior is on. The learned jump logit is
                                  # ~-7 and the prior lifts it to ~-1, so the mean action NEVER jumps (893 prior
                                  # firings, 1 jump, in the 3x eval of 20945). Off by default; the eval sets it.

    @torch.no_grad()
    def act(self, crop, vec, h, deterministic=False):
        """crop (B,6,32,32), vec (B,48), h (B,H). Returns dict with action_buf (B,6), action_game (B,6),
        logp (B,), value (B,), h_next (B,H)."""
        h1 = self.core(crop, vec, h)
        mean_xy, thr, jump, brake, use, vn, use_det, std_mult, brake_det = self.heads(h1, vec, crop)
        d_dir, d_thr, d_jump, d_brake, d_use = self.dists(mean_xy, thr, jump, brake, use, std_mult)
        if deterministic:
            direction = d_dir.mean; thr_s = thr; j = (torch.sigmoid(jump) > JUMP_DET_THRESHOLD).float(); b = brake_det   # 40.31: learned part
            u = use_det                           # 40.26: the learned preference, not the exploration mixture
            if self.JUMP_ON_PRIOR and crop is not None:
                j = torch.maximum(j, self.gap_prior(crop, vec))
        else:
            direction = d_dir.sample(); thr_s = d_thr.sample(); j = d_jump.sample(); b = d_brake.sample(); u = d_use.sample()
            # 10-08 15:55: FORCED jump samples where the marble is slow at a rise (the stopped-at-a-box case the head has
            # never learned: raw logit ~-38 there, so neither a bonus nor a mixture ever samples it). With probability
            # JUMP_FORCE_P the jump is set regardless of the head; the log-prob below is the POLICY's own for that action, so
            # the PPO update pushes the head with full strength (1 - p ~ 1) when the forced jump pays. Off-policy by design,
            # sampling only, never on deterministic play; the mask cannot fire on KOTM's floor (no rises).
            if JUMP_FORCE_P > 0.0 and getattr(self, 'last_force_mask', None) is not None:
                fm = self.last_force_mask & (torch.rand_like(j) < JUMP_FORCE_P)
                j = torch.where(fm, torch.ones_like(j), j)
        logp = d_dir.log_prob(direction).sum(-1) + d_thr.log_prob(thr_s) + d_jump.log_prob(j) + d_brake.log_prob(b) + d_use.log_prob(u)
        throttle = self.THROTTLE_FLOOR + (1 - self.THROTTLE_FLOOR) * torch.sigmoid(thr_s)
        action_buf = torch.stack([direction[:, 0], direction[:, 1], thr_s, j, b, u], dim=1)
        action_game = torch.stack([direction[:, 0], direction[:, 1], throttle, j, b, u], dim=1)
        return {'action_buf': action_buf, 'action_game': action_game, 'logp': logp,
                'mean_dir': mean_xy,          # pre-sampling mean: TURN_COST is charged on THIS,
                                              # so it prices steering and not exploration noise
                'value': self.value_from_norm(vn), 'h_next': h1}

    # ------------------------------------------------------------------ training
    def evaluate_seq(self, crops, vecs, h0, actions, resets, h_reset=None):
        """crops (T,B,6,32,32), vecs (T,B,48), h0 (B,H), actions (T,B,6),
        resets (T,B) float: 1 where the hidden state restarts BEFORE step t
        (first step of a new segment). h_reset (T,B,H): the state the actor restarted from at
        those steps (zeros for an ordinary segment, a live start's warmed-up state, log 40.42);
        None = zeros. Returns logp (T,B), continuous entropy (T,B),
        discrete entropy (T,B), value_norm (T,B)."""
        T, B = actions.shape[:2]
        h = h0
        logps, ents, ents_d, vns = [], [], [], []
        for t in range(T):
            r = resets[t].unsqueeze(-1)
            h = h * (1.0 - r) if h_reset is None else h * (1.0 - r) + h_reset[t] * r
            h = self.core(crops[t], vecs[t], h)
            mean_xy, thr, jump, brake, use, vn, _, std_mult, _ = self.heads(h, vecs[t], crops[t])
            d_dir, d_thr, d_jump, d_brake, d_use = self.dists(mean_xy, thr, jump, brake, use, std_mult)
            a = actions[t]
            logp = (d_dir.log_prob(a[:, 0:2]).sum(-1) + d_thr.log_prob(a[:, 2]) + d_jump.log_prob(a[:, 3]) + d_brake.log_prob(a[:, 4])
                    + d_use.log_prob(a[:, 5]))
            # Entropy is returned SPLIT (2026-09-20). Lumping all four distributions together and
            # paying one coefficient on the sum let the entropy bonus buy its entropy from the
            # CHEAPEST head instead of the useful one: Bernoulli entropy rises very steeply from
            # p near 0, so raising the coefficient to widen steering instead randomised braking,
            # which went 0.34 % -> 17.25 % of decisions in 90 updates while direction std moved
            # 0.18 -> 0.25. Speed fell 4.9 -> 4.7 and arrivals 86 % -> 83 %. The caller now prices
            # the continuous heads (where exploration is wanted) and the discrete ones (which have
            # their own deliberate priors, JUMP_DAMP and BRAKE_SUPPRESS) separately.
            # 40.30: minus the post-kick noise boost (Normal entropy rises by log k at k x std), so the entropy
            # controller keeps seeing the base exploration and does not cut it everywhere to pay for the boost
            ent_cont = (d_dir.entropy() - torch.log(std_mult).unsqueeze(-1)).sum(-1) + d_thr.entropy()
            ent_disc = d_jump.entropy() + d_brake.entropy() + d_use.entropy()
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
