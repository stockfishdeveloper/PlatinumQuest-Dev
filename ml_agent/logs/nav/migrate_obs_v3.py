"""Expand a NAV_OBS_V2 checkpoint (VEC_DIM 53) to V3 (VEC_DIM 56): zero columns for the 3 new gap inputs in
vec.0.weight and dir_head.0.weight (+ their Adam moments), so the migrated model behaves identically until
training moves the new weights. Usage: python logs/nav/migrate_obs_v3.py <src.pth> <dst.pth>"""
import sys, torch
sys.path.insert(0, '.')
from nav.model import NavActorCritic
from nav.obs import NAV_OBS_VERSION, VEC_DIM
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
assert ck['obs_version'] == 'NAV_OBS_V2', ck['obs_version']
sd = ck['model']; add = VEC_DIM - 53
def expand(w):
    return torch.cat([w, torch.zeros(w.shape[0], add, dtype=w.dtype)], dim=1)
sd['vec.0.weight'] = expand(sd['vec.0.weight'])
sd['dir_head.0.weight'] = expand(sd['dir_head.0.weight'])
m = NavActorCritic(); m.load_state_dict(sd)          # shape check against the V3 model
names = [n for n, p in m.named_parameters() if p.requires_grad]
opt = ck.get('opt')
if opt:
    for idx, n in enumerate(names):
        if n in ('vec.0.weight', 'dir_head.0.weight') and idx in opt['state']:
            st = opt['state'][idx]
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].dim() == 2 and st[k].shape[1] == 53 + (256 if n.startswith('dir_head') else 0):
                    st[k] = expand(st[k])
ck['obs_version'] = NAV_OBS_VERSION
ck['migrated'] = f'{src} -> V3 (gap block) 2026-09-23'
torch.save(ck, dst)
print('wrote', dst, 'update', ck['update'], 'vec.0', tuple(sd['vec.0.weight'].shape), 'dir_head.0', tuple(sd['dir_head.0.weight'].shape))
