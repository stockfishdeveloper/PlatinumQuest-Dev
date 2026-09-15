"""
Human demonstrations for behavior cloning (BC).

Loads every demos/demo_*.npz written by record_demos.py and exposes them as
(state, action) tensors in exactly the format the Actor consumes:

  state  (N, obs_dim)  obs_model from the recording: the same normalized obs +
                       tick frame history + terrain sample the trainer builds
                       (verified bit-for-bit against train_ppo.py).
  action (N, 5)        [dx, dy, throttle, jump, brake]: dx,dy is the human's key
                       direction as a unit vector in the agent's fixed frame,
                       throttle is 1 when any movement key was held, jump is the
                       jump key, brake is the derived label (intent opposes
                       velocity).

bc_loss() scores the actor's heads against those targets:
  direction  1 - cos(policy mean direction, human direction), only on ticks
             where the human was moving and not braking (when braking, the
             policy's brake head overrides the direction, so the direction head
             should not be taught to point backwards).
  throttle   BCE(sigmoid(throttle logit), moving?)
  jump       BCE(jump logit, jump key)
  brake      BCE(brake logit, derived brake)
All four terms are O(1), so the total is easy to weigh against the PPO loss.

Used by bc_pretrain.py (pure BC on the actor) and by train_ppo.py (decaying BC
auxiliary term inside the PPO update).
"""
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demos')
MIN_HOLDOUT_TICKS = 5000       # the validation round must be (most of) a full round
HIST = slice(35, 59)           # frame-history block of obs_model (4 x pos+vel, t-8..t-32 ticks)

# Copycat guard (2026-09-14). A human holds keys for long stretches, so the
# recent trajectory (frame history) predicts the human's current action almost
# by itself, and a policy cloned from raw recordings learns "continue what you
# were doing" instead of "look at the gems and the edges". Live, its own noisy
# motion then confirms whatever it was doing and it drives straight off the
# map (measured: direction agreement 0.97 on recordings, 0.54 with the history
# swapped for another tick's; 7-14 pts and ~20 OOB per game live). Training
# with the history zeroed or swapped on most samples, plus small input noise,
# forces the decision onto the current state. Applies wherever bc_loss is used.
AUG_P_ZERO_HIST = 0.35         # history zeroed (as in the first 32 ticks of a game)
AUG_P_SWAP_HIST = 0.35         # history taken from a random other tick
AUG_NOISE_SD = 0.03            # gaussian noise on every input (normalized units)


def augment(states, p_zero=AUG_P_ZERO_HIST, p_swap=AUG_P_SWAP_HIST, noise_sd=AUG_NOISE_SD):
    """Copycat-guard augmentation of a batch of demo states (see above)."""
    x = states.clone()
    n = len(x)
    r = torch.rand(n)
    zero = torch.where(r < p_zero)[0]
    swap = torch.where((r >= p_zero) & (r < p_zero + p_swap))[0]
    x[zero, HIST] = 0.0
    if len(swap):
        x[swap, HIST] = states[torch.randint(0, n, (len(swap),)), HIST]
    if noise_sd > 0:
        x = x + torch.randn_like(x) * noise_sd
    return x


class DemoSet:
    def __init__(self, obs_dim, demo_dir=DEMO_DIR, holdout=True, log=print):
        files = sorted(glob.glob(os.path.join(demo_dir, 'demo_*.npz')))
        obs, act, gid = [], [], []
        self.files = []
        next_game = 0
        for f in files:
            try:
                a = np.load(f, allow_pickle=False)
                om, ac, g = a['obs_model'], a['action'], a['game'].astype(np.int64)
            except Exception as e:
                log(f"  demos: skipping {os.path.basename(f)} ({e})")
                continue
            if len(om) == 0:
                continue
            if om.shape[1] != obs_dim:
                log(f"  demos: skipping {os.path.basename(f)}: obs dim {om.shape[1]} != {obs_dim}")
                continue
            obs.append(om.astype(np.float32))
            act.append(ac.astype(np.float32))
            gid.append(g + next_game)
            next_game += int(g.max()) + 1
            self.files.append(os.path.basename(f))
        if not obs:
            self.obs = torch.zeros((0, obs_dim)); self.act = torch.zeros((0, 5))
            self.game = np.zeros(0, dtype=np.int64)
            self.train_idx = np.zeros(0, dtype=np.int64); self.val_idx = np.zeros(0, dtype=np.int64)
            return
        self.obs = torch.from_numpy(np.concatenate(obs))
        self.act = torch.from_numpy(np.concatenate(act))
        self.game = np.concatenate(gid)
        # Hold out the last full round for validation so BC metrics are measured
        # on a round the policy never trained on (ticks within a round are
        # highly correlated, so a random split would overstate generalization).
        self.val_game = None
        if holdout:
            for gidx in sorted(np.unique(self.game))[::-1]:
                n_round = int((self.game == gidx).sum())
                # only hold a round out if enough training data remains without it
                if n_round >= MIN_HOLDOUT_TICKS and len(self.game) - n_round >= MIN_HOLDOUT_TICKS:
                    self.val_game = int(gidx)
                    break
        mask_val = self.game == self.val_game if self.val_game is not None else np.zeros(len(self.game), bool)
        self.val_idx = np.where(mask_val)[0]
        self.train_idx = np.where(~mask_val)[0]

    def __len__(self):
        return len(self.game)

    @property
    def n_train(self):
        return len(self.train_idx)

    @property
    def n_val(self):
        return len(self.val_idx)

    def describe(self):
        if len(self) == 0:
            return "no demonstrations"
        rounds = len(np.unique(self.game))
        a = self.act
        return (f"{len(self):,} ticks ({len(self) * 0.016 / 60:.1f} min) in {rounds} rounds from {len(self.files)} file(s); "
                f"train {self.n_train:,} / val {self.n_val:,} (val = round {self.val_game}); "
                f"moving {100 * (a[:, 2] > 0.5).float().mean():.0f}% jump {100 * a[:, 3].mean():.1f}% brake {100 * a[:, 4].mean():.1f}%")

    def sample(self, batch_size, split='train', augmented=True):
        idx = self.train_idx if split == 'train' else self.val_idx
        pick = idx[np.random.randint(0, len(idx), size=batch_size)]
        states = self.obs[pick]
        return (augment(states) if augmented else states), self.act[pick]

    def batches(self, batch_size, split='train', shuffle=True):
        idx = self.train_idx if split == 'train' else self.val_idx
        if shuffle:
            idx = np.random.permutation(idx)
        for s in range(0, len(idx), batch_size):
            pick = idx[s:s + batch_size]
            yield self.obs[pick], self.act[pick]


def bc_loss(actor, states, actions):
    """Behavior-cloning loss of `actor` on demo (states, actions).

    Returns (loss, metrics). metrics are floats: dir_cos (mean cosine between
    the policy mean direction and the human direction on moving, non-braking
    ticks), dir_within30 (fraction within 30 deg), thr_acc, jump_ll, brake_ll
    (per-head log losses), jump_rate / brake_rate (predicted probabilities),
    and the four loss terms.
    """
    mean_xy, thr_logit, jump_logit, brake_logit = actor.forward(states)
    target_dir = actions[:, 0:2]
    thr_t, jump_t, brake_t = actions[:, 2], actions[:, 3], actions[:, 4]

    unit = mean_xy / torch.norm(mean_xy, dim=1, keepdim=True).clamp(min=1e-6)
    cos = (unit * target_dir).sum(dim=1)
    dir_mask = ((thr_t > 0.5) & (brake_t < 0.5)).float()
    n_dir = dir_mask.sum().clamp(min=1.0)
    dir_loss = ((1.0 - cos) * dir_mask).sum() / n_dir

    # Throttle: do not teach "at rest -> keep no keys held". The demonstrator's
    # deliberate stops (recorded on purpose to cover low-speed states) are
    # labelled throttle 0 while stationary, and a clone that learns them
    # freezes next to gems (2026-09-14 trace: 46% of a round dithering at
    # ~1 u/s with throttle 0.2). Keep the resume ticks (at rest, key held) and
    # every moving tick; drop only stationary no-key ticks from the throttle
    # term. Velocity is obs_model[3:5] * 20 (u/s).
    speed = torch.norm(states[:, 3:5], dim=1) * 20.0
    thr_w = 1.0 - ((speed < 1.0) & (thr_t < 0.5)).float()
    thr_loss = (F.binary_cross_entropy_with_logits(thr_logit.squeeze(-1), thr_t, reduction='none') * thr_w).sum() / thr_w.sum().clamp(min=1.0)
    jump_loss = F.binary_cross_entropy_with_logits(jump_logit.squeeze(-1), jump_t)
    brake_loss = F.binary_cross_entropy_with_logits(brake_logit.squeeze(-1), brake_t)
    loss = dir_loss + thr_loss + jump_loss + brake_loss

    with torch.no_grad():
        metrics = {
            'loss': loss.item(),
            'dir_loss': dir_loss.item(), 'thr_loss': thr_loss.item(),
            'jump_ll': jump_loss.item(), 'brake_ll': brake_loss.item(),
            'dir_cos': ((cos * dir_mask).sum() / n_dir).item(),
            'dir_within30': (((cos > 0.866).float() * dir_mask).sum() / n_dir).item(),
            'thr_acc': ((((thr_logit.squeeze(-1) > 0).float() == thr_t).float() * thr_w).sum() / thr_w.sum().clamp(min=1.0)).item(),
            'jump_rate': torch.sigmoid(jump_logit).mean().item(),
            'brake_rate': torch.sigmoid(brake_logit).mean().item(),
            'n_dir': int(n_dir.item()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(actor, demos, split='val', batch_size=4096, history='clean'):
    """Average bc_loss metrics over a whole split (weighted by batch size).
    history: 'clean' (as recorded), 'zero' or 'swap' (copycat probes: the
    policy should keep agreeing with the human without its own past)."""
    actor.eval()
    tot, n = {}, 0
    for states, actions in demos.batches(batch_size, split=split, shuffle=False):
        if history == 'zero':
            states = augment(states, p_zero=1.0, p_swap=0.0, noise_sd=0.0)
        elif history == 'swap':
            states = augment(states, p_zero=0.0, p_swap=1.0, noise_sd=0.0)
        _, m = bc_loss(actor, states, actions)
        b = len(states)
        for k, v in m.items():
            tot[k] = tot.get(k, 0.0) + v * b
        n += b
    actor.train()
    return {k: v / max(n, 1) for k, v in tot.items()}
