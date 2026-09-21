"""Warm-start a NAV_OBS_V1 checkpoint into the wider NAV_OBS_V2 observation.

V2 appends NEXT_DIM numbers (the next gem of the group) after the edge rays, so exactly one
tensor changes shape: `vec.0.weight`, 64 x 48 -> 64 x 53, which is 0.4 % of the network. Every
other tensor -- convolution, GRU, all four heads, the PopArt value statistics -- is copied
unchanged. The appended input columns start at ZERO, so the converted policy computes
bit-identically to the one it came from, and learns to use the new inputs from the first update
(a zero weight still receives gradient: dL/dW = delta * input).

Adam's moments are carried over too, except for the one resized tensor, whose moments are reset
to zero: its shape changed, so the stored moments no longer describe it.

    python convert_ckpt_v2.py                              # nav_latest.pth -> nav_latest.pth (backed up first)
    python convert_ckpt_v2.py --in models/nav/nav_best_20260919_0715.pth --out models/nav/nav_v2.pth
"""
import os
import sys
import shutil
import argparse
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default=os.path.join(HERE, 'models', 'nav', 'nav_latest.pth'))
    ap.add_argument('--out', dest='dst', default=None, help='default: overwrite the input (a .v1backup copy is kept)')
    a = ap.parse_args()
    dst = a.dst or a.src

    from nav.obs import NAV_OBS_VERSION, VEC_DIM, VEC_NEXT, NEXT_DIM
    from nav.model import NavActorCritic

    ck = torch.load(a.src, map_location='cpu')
    old_ver = ck.get('obs_version')
    print(f'source     {a.src}')
    print(f'  version  {old_ver} -> {NAV_OBS_VERSION}')
    print(f'  update   {ck.get("update")}, steps {ck.get("steps"):,}')
    if old_ver == NAV_OBS_VERSION:
        raise SystemExit('already at the current observation version; nothing to do')

    sd = ck['model']
    key = 'vec.0.weight'
    old_w = sd[key]
    old_dim = old_w.shape[1]
    if old_dim != VEC_NEXT:
        raise SystemExit(f'{key} has {old_dim} input columns but VEC_NEXT is {VEC_NEXT}; layouts do not line up')

    fresh = NavActorCritic()                    # built at the NEW width
    new_sd = fresh.state_dict()
    if new_sd[key].shape[1] != VEC_DIM:
        raise SystemExit(f'new model expects {new_sd[key].shape[1]} columns, VEC_DIM is {VEC_DIM}')

    copied, resized = 0, []
    for k, v in new_sd.items():
        if k not in sd:
            raise SystemExit(f'tensor {k} missing from the checkpoint; the architecture changed by more than the width')
        if sd[k].shape == v.shape:
            new_sd[k] = sd[k].clone(); copied += 1
        elif k == key:
            w = torch.zeros_like(v)
            w[:, :old_dim] = sd[k]              # carry the learned columns, leave the appended ones at zero
            new_sd[k] = w; resized.append(k)
        else:
            raise SystemExit(f'unexpected shape change on {k}: {tuple(sd[k].shape)} -> {tuple(v.shape)}')
    print(f'  tensors  {copied} copied unchanged, {len(resized)} widened {resized}')
    print(f'           {NEXT_DIM} appended input columns start at zero -> policy is unchanged at conversion')

    # optimiser: keep every moment except for the tensor whose shape changed
    opt = ck.get('opt')
    if opt and 'state' in opt:
        names = [n for n, _ in fresh.named_parameters()]
        idx = names.index(key)
        st = opt['state']
        if idx in st:
            for mom in ('exp_avg', 'exp_avg_sq'):
                if mom in st[idx]:
                    st[idx][mom] = torch.zeros_like(new_sd[key])
            print(f'  adam     moments reset for {key} (shape changed), kept for the other {len(st) - 1} tensors')

    if dst == a.src:
        backup = a.src.replace('.pth', f'.v1backup_{datetime.now():%Y%m%d_%H%M}.pth')
        shutil.copy2(a.src, backup)
        print(f'  backup   {os.path.basename(backup)}')

    ck['model'] = new_sd
    ck['obs_version'] = NAV_OBS_VERSION
    ck['converted'] = f'{old_ver}->{NAV_OBS_VERSION} at {datetime.now().isoformat()}'
    torch.save(ck, dst + '.tmp'); os.replace(dst + '.tmp', dst)
    print(f'written    {dst}')


if __name__ == '__main__':
    main()
