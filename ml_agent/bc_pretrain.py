"""
Behavior-cloning pretraining of the actor on human demonstrations.

    python bc_pretrain.py                 # newest checkpoint + every demos/demo_*.npz

What it does
  1. loads the latest models/checkpoints/update_*.pth (actor + critic + stats)
  2. scores the current actor against the demos (baseline)
  3. trains the actor's four heads to imitate the human (demo_data.bc_loss),
     keeping the epoch with the best validation loss (last full round held out)
  4. writes models/checkpoints/update_<N+1>.pth so `python train_ppo.py`
     resumes from it with no flags. The new checkpoint:
       - keeps the critic, its optimizer and all training counters
       - drops the actor's Adam state (the weights moved; stale moments would
         misfire) so PPO starts the actor optimizer fresh
       - clears warmup_start_update, so the trainer runs its critic-only
         warmup (WARMUP_ROLLOUTS) first: the critic relearns V(s) under the
         new, much faster policy before the actor moves on PPO advantages
       - records bc_updates_done = 0 so train_ppo.py starts its decaying BC
         auxiliary loss from the top
Nothing here touches the game; run it between training sessions.
"""
import os
import sys
import glob
import copy
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
from train_ppo import Actor, widen_state_dict, resolve_terrain_path
from terrain_obs import TerrainMap, PERCEPTION_DIM
from demo_data import DemoSet, bc_loss, evaluate, augment

EPOCHS = 30
BATCH = 512
LR = 1e-4
CKPT_DIR = os.path.join(HERE, 'models', 'checkpoints')


def latest_checkpoint():
    """Newest update_*.pth that is not itself a BC-pretrained checkpoint, so a
    re-run of this script starts again from the PPO policy, not from its own
    previous output."""
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'update_*.pth')),
                   key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]), reverse=True)
    for f in files:
        try:
            if not torch.load(f, weights_only=False, map_location='cpu').get('bc_pretrained_from'):
                return f
        except Exception:
            continue
    return None


def fmt(m):
    return (f"loss {m['loss']:.3f} | dir cos {m['dir_cos']:.3f} within30 {100 * m['dir_within30']:.0f}% "
            f"| thr acc {100 * m['thr_acc']:.1f}% | jump ll {m['jump_ll']:.3f} rate {100 * m['jump_rate']:.2f}% "
            f"| brake ll {m['brake_ll']:.3f} rate {100 * m['brake_rate']:.1f}%")


def report(actor, demos, tag):
    """Clean fit plus the two copycat probes (history zeroed / swapped)."""
    for split in ('train', 'val'):
        if (demos.n_train if split == 'train' else demos.n_val) == 0:
            continue
        c = evaluate(actor, demos, split); z = evaluate(actor, demos, split, history='zero'); w = evaluate(actor, demos, split, history='swap')
        print(f"{tag:7s} {split:5s}: {fmt(c)}")
        print(f"{'':7s} {'':5s}  copycat probe: dir cos with history zeroed {z['dir_cos']:.3f}, swapped {w['dir_cos']:.3f} (clean {c['dir_cos']:.3f})")
    return evaluate(actor, demos, 'val' if demos.n_val else 'train')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_checkpoint()
    if not path or not os.path.exists(path):
        print("No checkpoint found in models/checkpoints/."); return
    ck = torch.load(path, weights_only=False)
    saved = ck.get('actor_state_dict', ck.get('model_state_dict'))
    obs_dim = 35 + 24 + PERCEPTION_DIM          # current layout; older checkpoints are widened
    actor = Actor(obs_dim=obs_dim)
    actor.load_state_dict(widen_state_dict(actor, saved), strict=False)
    actor.LOG_STD_MAX = float(ck.get('log_std_max', Actor.LOG_STD_MAX))
    if not ck.get('throttle_floor_active', True):
        actor.THROTTLE_FLOOR = 0.0
    n_upd = int(ck.get('total_updates', 0))
    print(f"Checkpoint: {os.path.basename(path)} (update {n_upd}, obs_dim {obs_dim})")

    terrain = TerrainMap(resolve_terrain_path())
    demos = DemoSet(obs_dim, terrain=terrain)
    print(f"Demos: {demos.describe()}")
    if demos.n_train == 0:
        print("Nothing to train on."); return
    human = demos.act
    print(f"Human rates: jump {100 * human[:, 3].mean():.2f}% brake {100 * human[:, 4].mean():.1f}% moving {100 * (human[:, 2] > 0.5).float().mean():.1f}%")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor.to(device)
    demos.obs = demos.obs.to(device); demos.act = demos.act.to(device)

    print()
    report(actor, demos, 'BEFORE')
    sel_split = 'val' if demos.n_val else 'train'

    # The trunks and heads train; the exploration stds (log_std, throttle_log_std)
    # are PPO's and get no gradient from bc_loss, so they stay put.
    params = [p for n, p in actor.named_parameters() if n not in ('log_std', 'throttle_log_std')]
    opt = torch.optim.Adam(params, lr=LR)
    best_val, best_state, best_epoch = float('inf'), copy.deepcopy(actor.state_dict()), 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tot, n = 0.0, 0
        for states, actions in demos.batches(BATCH, 'train', shuffle=True):
            loss, _ = bc_loss(actor, augment(states), actions)     # copycat guard (demo_data.augment)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item() * len(states); n += len(states)
        val = evaluate(actor, demos, sel_split)
        swp = evaluate(actor, demos, sel_split, history='swap')
        crit = 0.5 * (val['loss'] + swp['loss'])       # select on clean AND history-swapped fit
        tag = ''
        if crit < best_val:
            best_val, best_state, best_epoch, tag = crit, copy.deepcopy(actor.state_dict()), epoch, '  *best*'
        print(f"epoch {epoch:2d}  train {tot / n:.3f}  {sel_split} {fmt(val)} | swapped-history cos {swp['dir_cos']:.3f}{tag}")
    actor.load_state_dict(best_state)
    print(f"\nKept epoch {best_epoch} ({time.time() - t0:.0f} s).")
    after_val = report(actor, demos, 'AFTER')

    actor.to('cpu')
    with torch.no_grad():
        split = obs_dim - PERCEPTION_DIM
        w = actor.features[0].weight
        print(f"Terrain-column weight ratio (actor trunk): {(w[:, split:].abs().mean() / w[:, :split].abs().mean()).item():.3f}")

    out = dict(ck)
    out['actor_state_dict'] = actor.state_dict()
    out.pop('actor_optimizer_state_dict', None)
    out['warmup_start_update'] = None
    out['total_updates'] = n_upd + 1
    out['bc_updates_done'] = 0
    out['bc_pretrained_from'] = os.path.basename(path)
    out['bc_val_metrics'] = {k: float(v) for k, v in after_val.items()}
    out['log_std_max'] = float(actor.LOG_STD_MAX)
    out_path = os.path.join(CKPT_DIR, f'update_{n_upd + 1}.pth')
    torch.save(out, out_path)
    print(f"\nSaved {out_path}")
    print("Next: python train_ppo.py  (it resumes from this checkpoint: critic warmup first, then PPO with the decaying BC term)")


if __name__ == '__main__':
    main()
