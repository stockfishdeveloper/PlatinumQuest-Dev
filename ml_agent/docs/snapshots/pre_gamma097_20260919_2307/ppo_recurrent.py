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
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 3             # was 4: with 8 instances the update, not collection, bounds throughput (2026-09-18)
MINIBATCH_SEQS = 16
ENTROPY_COEF = 0.002   # was 0.005 until 2026-09-18 22:45. Entropy rose monotonically 0.46 -> 0.83
                       # over 8.5 h (direction spread 0.23 -> 0.28) while reward stayed flat: the
                       # near-converged policy's advantage signal had shrunk enough that the fixed
                       # entropy bonus dominated the gradient and was slowly randomising the policy.
                       # Performance had not dropped yet, but the drift was unbroken and the run was
                       # about to go unattended overnight. Target band: entropy roughly stable
                       # ~0.4-0.7 with arrivals held. Revert point: models/nav/nav_good_20260918_2245.pth
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
    stats = {'pl': 0.0, 'vl': 0.0, 'ent': 0.0, 'kl': 0.0, 'clipfrac': 0.0, 'gn': 0.0, 'n': 0}
    stop = False
    for epoch in range(EPOCHS):
        perm = torch.randperm(nseq, device=dev)
        for start in range(0, nseq, mb):
            idx = perm[start:start + mb]
            # (B, T, ...) -> (T, B, ...)
            logp, ent, vn = model.evaluate_seq(crop[idx].transpose(0, 1), vec[idx].transpose(0, 1), h0[idx],
                                               act[idx].transpose(0, 1), resets[idx].transpose(0, 1))
            logp, ent, vn = logp.transpose(0, 1), ent.transpose(0, 1), vn.transpose(0, 1)
            ratio = torch.exp(logp - old_logp[idx])
            a = adv_s[idx]
            pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
            v_clipped = old_vn[idx] + torch.clamp(vn - old_vn[idx], -CLIP, CLIP)
            vl = torch.max(F.mse_loss(vn, ret_s[idx], reduction='none'), F.mse_loss(v_clipped, ret_s[idx], reduction='none')).mean()
            loss = pl + VALUE_COEF * vl - ENTROPY_COEF * ent.mean()
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            with torch.no_grad():
                kl = (old_logp[idx] - logp).mean().item()
                stats['pl'] += pl.item(); stats['vl'] += vl.item(); stats['ent'] += ent.mean().item()
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
    return out
