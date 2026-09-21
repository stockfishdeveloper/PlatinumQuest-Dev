"""PPO for the recurrent navigator: rollout storage, GAE, sequence minibatches.

The rollout is one long time series per environment (T steps). It is cut into
sequences of SEQ_LEN steps; each sequence is replayed from the GRU state that
was recorded at its first step (no burn-in), with the state zeroed inside the
sequence wherever a new segment started (`resets`). Value targets are
PopArt-normalized by the model.
"""
import numpy as np
import torch
import torch.nn.functional as F

from nav.model import HIDDEN, ACTION_DIM
from nav.terrain import CROP_SHAPE
from nav.obs import VEC_DIM

SEQ_LEN = 32
GAMMA = 0.99          # 0.97 was tried and reverted immediately (2026-09-19 night): the claim that a
                      # shorter horizon pays more for speed is WRONG. Both discount terms shrink
                      # together, so the absolute gain from arriving 39 decisions sooner is 2.19 at
                      # 0.99 and 2.12 at 0.97 -- unchanged. Lower gamma only makes distant gems
                      # matter less. Do not retry this as a speed lever.
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 3             # was 4: with 8 instances the update, not collection, bounds throughput (2026-09-18)
MINIBATCH_SEQS = 16
ENTROPY_COEF = 0.01    # RAISED 0.002 -> 0.01 on 2026-09-20 08:15, deliberately, to break a points
                       # plateau: 650 updates of training moved real-round score 72.2 -> 74.1
                       # (t ~ 0.83, not significant) while the training metrics moved a lot. The
                       # policy had stopped finding new ways to drive.
                       #
                       # READ THE ENTROPY NUMBER CORRECTLY. `ent=` in the logs is DIFFERENTIAL
                       # entropy: it can be negative and 0.0 does NOT mean "no exploration". Over
                       # the overnight run it read 0.335 -> 0.01, which sounds like a collapse but
                       # is direction std 0.220 -> 0.183, a 17 % reduction. Convert before reacting:
                       #     ent = 2*(ln sd_dir + 1.4189) + (ln sd_thr + 1.4189)
                       # The real argument for raising it is the CLAMP HEADROOM in model.py:
                       # direction std was 0.183 against an allowed [0.135, 0.607], so 3.3x of
                       # exploration was available and unused. (Throttle std 0.408 is already pinned
                       # at its 0.407 ceiling and cannot contribute more.)
                       #
                       # Calibration: 0.002 lets std slowly fall, 0.005 made it rise monotonically
                       # 0.23 -> 0.28 over 8.5 h. 0.01 is chosen to be decisively in the rising
                       # regime rather than near break-even. This is a DELIBERATE, MONITORED
                       # randomisation -- if arrivals fall while std climbs, lower it again.
                       # Revert point: models/nav/nav_wander_20260920_0805.pth (74.1 pts, update 9415).

# ---------------------------------------------------------------- entropy controller
# ENTROPY_COEF above is only the STARTING value. A fixed coefficient cannot hold exploration
# where it belongs: the same number that is neutral for a fresh policy is far too small for a
# converged one, whose advantage signal has shrunk. Overnight 2026-09-19/20 that let direction
# std sag to 0.183 and the score flatline, and the operator had to notice and intervene by hand.
# It now self-corrects: after every update the coefficient moves until `ent` is back in band.
ENT_BAND_LO = 0.2      # differential entropy, NOT a probability. See the conversion above.
ENT_BAND_HI = 0.8      # band corresponds to direction std ~0.206 .. 0.278 (clamp allows 0.607),
                       # so it is reachable from either side.
ENT_UP = 1.05          # per update when below the band: coefficient doubles in ~14 updates (~4 min
ENT_DOWN = 0.95        # at 3.5 upd/min), fast enough to catch a sag, slow enough not to oscillate.
DISCRETE_ENT_COEF = 0.002   # FIXED bonus for the jump and brake heads, never adapted. The
                            # controller must not be able to reach these: Bernoulli entropy is
                            # the cheapest on the menu, so a rising coefficient buys it there
                            # first and randomises braking instead of widening steering. Held
                            # at the long-standing 0.002 so jump exploration keeps the level it
                            # had before 2026-09-20; the island maps need jumping to stay live.
ENT_COEF_MIN = 5e-4
ENT_COEF_MAX = 0.25
_ent_coef = ENTROPY_COEF      # live value, adapted by _adapt_entropy()
_ent_ema = None               # smoothed entropy the controller reacts to


def _adapt_entropy(ent):
    """Nudge the entropy coefficient so `ent` stays inside [ENT_BAND_LO, ENT_BAND_HI].

    Called once per PPO update with that update's mean entropy. Deliberately slow and
    multiplicative: the coefficient has to span a wide range over a run, and a proportional
    controller on a noisy signal oscillates. Returns (coefficient, smoothed entropy) for logging.
    """
    global _ent_coef, _ent_ema
    _ent_ema = ent if _ent_ema is None else 0.8 * _ent_ema + 0.2 * ent
    if _ent_ema < ENT_BAND_LO:
        _ent_coef = min(ENT_COEF_MAX, _ent_coef * ENT_UP)
    elif _ent_ema > ENT_BAND_HI:
        _ent_coef = max(ENT_COEF_MIN, _ent_coef * ENT_DOWN)
    return _ent_coef, _ent_ema
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.3
LR = 3e-4              # with the un-normalized direction mean the KL per update is ~0.005 at 1e-4: too slow


class Rollout:
    """Fixed-size storage for one environment (extend to (T, N) when there are N instances)."""

    def __init__(self, T, device, slack=0):
        # `slack` extra rows beyond T: when the pipelined trainer settles its in-flight groups
        # before an update, a worker can deliver one transition after full() already went true.
        # The slack keeps that transition instead of dropping it (see nav/train_nav.py).
        self.T = T; self.device = device; self.n = 0
        T = T + slack
        self.crop = np.zeros((T,) + CROP_SHAPE, dtype=np.float32)
        self.vec = np.zeros((T, VEC_DIM), dtype=np.float32)
        self.action = np.zeros((T, ACTION_DIM), dtype=np.float32)
        self.logp = np.zeros(T, dtype=np.float32)
        self.value = np.zeros(T, dtype=np.float32)
        self.reward = np.zeros(T, dtype=np.float32)
        self.done = np.zeros(T, dtype=np.float32)      # segment ended after this step
        self.reset = np.zeros(T, dtype=np.float32)     # hidden state was zero before this step
        self.h = np.zeros((T, HIDDEN), dtype=np.float32)

    def add(self, crop, vec, action, logp, value, reward, done, reset, h):
        i = self.n
        self.crop[i] = crop; self.vec[i] = vec; self.action[i] = action; self.logp[i] = logp
        self.value[i] = value; self.reward[i] = reward; self.done[i] = done; self.reset[i] = reset; self.h[i] = h
        self.n += 1

    def full(self):
        return self.n >= self.T

    def clear(self):
        self.n = 0

    def truncate(self, k):
        """Drop the last k stored steps (a segment whose game state turned out to be corrupted)."""
        self.n = max(0, self.n - k)

    def gae(self, last_value):
        T = self.n
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else self.value[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.reward[t] + GAMMA * nv * nonterminal - self.value[t]
            gae = delta + GAMMA * LAMBDA * nonterminal * gae
            adv[t] = gae
        return adv, adv + self.value[:T]


def ppo_update(model, opt, rolls, last_values, log=print):
    """One PPO update over one rollout or a list of them (one per game instance). Each rollout is
    cut into SEQ_LEN sequences on its own (its own GAE bootstrap), then all sequences are pooled.
    Returns a dict of stats."""
    if not isinstance(rolls, (list, tuple)):
        rolls, last_values = [rolls], [last_values]
    dev = rolls[0].device
    parts = []
    for roll, lv in zip(rolls, last_values):
        T = roll.n - (roll.n % SEQ_LEN)
        if T < SEQ_LEN:
            continue
        adv, ret = roll.gae(lv)
        parts.append((roll, T, adv[:T], ret[:T]))
    if not parts:
        return {}
    cat = lambda f: np.concatenate([f(r)[:T] for r, T, _, _ in parts])
    adv = np.concatenate([a for _, _, a, _ in parts]); ret = np.concatenate([r for _, _, _, r in parts])
    T = len(adv)
    ret_t = torch.as_tensor(ret, device=dev)
    model.update_value_stats(ret_t)
    ret_norm = (ret_t - model.value_mean) / model.value_std
    adv_t = torch.as_tensor(adv, device=dev)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    nseq = T // SEQ_LEN
    to = lambda a: torch.as_tensor(a, device=dev)
    crop = to(cat(lambda r: r.crop)).view(nseq, SEQ_LEN, *CROP_SHAPE)
    vec = to(cat(lambda r: r.vec)).view(nseq, SEQ_LEN, VEC_DIM)
    act = to(cat(lambda r: r.action)).view(nseq, SEQ_LEN, ACTION_DIM)
    old_logp = to(cat(lambda r: r.logp)).view(nseq, SEQ_LEN)
    old_vn = ((to(cat(lambda r: r.value)) - model.value_mean) / model.value_std).view(nseq, SEQ_LEN)
    resets = to(cat(lambda r: r.reset)).view(nseq, SEQ_LEN)
    h0 = to(cat(lambda r: r.h)).view(nseq, SEQ_LEN, HIDDEN)[:, 0, :]
    adv_s = adv_t.view(nseq, SEQ_LEN); ret_s = ret_norm.view(nseq, SEQ_LEN)
    # two minibatches per epoch however large the pooled rollout is: each optimizer step is a
    # 32-step GRU unroll and costs the same whatever the batch, and the GPU is shared with the
    # game instances (16 s per 8192-sample update at 8 minibatches x 4 epochs, 2026-09-18)
    mb = max(MINIBATCH_SEQS, nseq // 2)
    stats = {'pl': 0.0, 'vl': 0.0, 'ent': 0.0, 'ent_d': 0.0, 'kl': 0.0, 'clipfrac': 0.0, 'gn': 0.0, 'n': 0}
    stop = False
    for epoch in range(EPOCHS):
        perm = torch.randperm(nseq, device=dev)
        for start in range(0, nseq, mb):
            idx = perm[start:start + mb]
            # (B, T, ...) -> (T, B, ...)
            logp, ent, ent_d, vn = model.evaluate_seq(crop[idx].transpose(0, 1), vec[idx].transpose(0, 1), h0[idx],
                                                      act[idx].transpose(0, 1), resets[idx].transpose(0, 1))
            logp, ent, ent_d, vn = logp.transpose(0, 1), ent.transpose(0, 1), ent_d.transpose(0, 1), vn.transpose(0, 1)
            ratio = torch.exp(logp - old_logp[idx])
            a = adv_s[idx]
            pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
            v_clipped = old_vn[idx] + torch.clamp(vn - old_vn[idx], -CLIP, CLIP)
            vl = torch.max(F.mse_loss(vn, ret_s[idx], reduction='none'), F.mse_loss(v_clipped, ret_s[idx], reduction='none')).mean()
            loss = pl + VALUE_COEF * vl - _ent_coef * ent.mean() - DISCRETE_ENT_COEF * ent_d.mean()
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            with torch.no_grad():
                kl = (old_logp[idx] - logp).mean().item()
                stats['pl'] += pl.item(); stats['vl'] += vl.item(); stats['ent'] += ent.mean().item(); stats['ent_d'] += ent_d.mean().item()
                stats['kl'] += kl; stats['clipfrac'] += ((ratio - 1).abs() > CLIP).float().mean().item()
                stats['gn'] += float(gn); stats['n'] += 1
            if kl > TARGET_KL:
                stop = True; break
        if stop:
            log(f'  KL early stop at epoch {epoch + 1}')
            break
    n = max(stats.pop('n'), 1)
    out = {k: v / n for k, v in stats.items()}
    out['epochs'] = epoch + 1; out['seqs'] = nseq
    out['ent_coef'], out['ent_ema'] = _adapt_entropy(out['ent'])
    return out
