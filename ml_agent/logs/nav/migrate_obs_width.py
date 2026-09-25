"""Expand a checkpoint's vec input width to the current VEC_DIM with zero columns (vec.0.weight, dir_head.0.weight
and their Adam moments), so the migrated model behaves identically until training moves the new weights.
Usage: python logs/nav/migrate_obs_width.py <src.pth> <dst.pth>"""
import sys, torch
sys.path.insert(0, '.')
from nav.model import NavActorCritic
from nav.obs import NAV_OBS_VERSION, VEC_DIM
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
sd = ck['model']; old = sd['vec.0.weight'].shape[1]; add = VEC_DIM - old
assert add > 0, (old, VEC_DIM)
def expand(w):
    return torch.cat([w, torch.zeros(w.shape[0], add, dtype=w.dtype)], dim=1)
sd['vec.0.weight'] = expand(sd['vec.0.weight']); sd['dir_head.0.weight'] = expand(sd['dir_head.0.weight'])
m = NavActorCritic(); m.load_state_dict(sd)
names = [n for n, p in m.named_parameters() if p.requires_grad]
opt = ck.get('opt')
if opt:
    for idx, n in enumerate(names):
        if n in ('vec.0.weight', 'dir_head.0.weight') and idx in opt['state']:
            st = opt['state'][idx]
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].dim() == 2 and st[k].shape[1] == old + (256 if n.startswith('dir_head') else 0):
                    st[k] = expand(st[k])
ck['obs_version'] = NAV_OBS_VERSION; ck['migrated'] = f'{src} width {old} -> {VEC_DIM}'
torch.save(ck, dst)
print('wrote', dst, 'update', ck['update'], 'vec width', old, '->', VEC_DIM)
