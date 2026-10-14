"""Complete-map inference cache for a frozen motion student; no geometry crop.

Run equivalence/timing with RANDOM weights only; no fitting or gameplay change:
python -m nav.marble_dynamics.student_cache --output <new-report.json>
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.physics_student import MotionStudent
from nav.marble_dynamics.student_data import STATE_FEATURES, SURFACE_FEATURES
from nav.marble_dynamics.student_dryrun import weight_digest
from nav.marble_dynamics.record import ML, sha


class CompleteMapMotionCache:
    """One immutable map and frozen weight version; states can vary freely."""
    def __init__(self, model, surfaces, bbox_lo, bbox_hi):
        if type(model) is not MotionStudent:
            raise ValueError('Cache supports only the original MotionStudent forward; variants require separate equivalence checks')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Inference cache requires an explicitly frozen eval model')
        if surfaces.ndim != 2 or surfaces.shape[1] != len(SURFACE_FEATURES):
            raise ValueError('Full prepared surface schema required')
        if not len(surfaces) or bbox_lo.shape != (len(surfaces), 3) or bbox_hi.shape != bbox_lo.shape:
            raise ValueError('Every surface must have aligned bounds')
        if any(not bool(torch.isfinite(x).all()) for x in (surfaces, bbox_lo, bbox_hi)):
            raise ValueError('Nonfinite map geometry')
        self.model = model
        self.versions = tuple(p._version for p in model.parameters())
        # Own the geometry; caller mutation must not silently alter cached keys.
        self.bbox_lo = bbox_lo.detach().clone()
        self.bbox_hi = bbox_hi.detach().clone()
        self.bound_versions = (self.bbox_lo._version, self.bbox_hi._version)
        self.surface_count = len(surfaces)
        with torch.inference_mode():
            encoded = model.surface(surfaces)
            self.keys = model.key(encoded)
            self.values = model.value(encoded)

    def __call__(self, states):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Model is no longer frozen; rebuild after fitting')
        if self.versions != tuple(p._version for p in self.model.parameters()):
            raise ValueError('Weights changed; rebuild the complete-map cache')
        if self.bound_versions != (self.bbox_lo._version, self.bbox_hi._version):
            raise ValueError('Cached geometry changed; rebuild')
        with torch.inference_mode():
            state = self.model.state(states)
            logits = self.model.query(state) @ self.keys.T / math.sqrt(self.model.width)
            p = states[:, :3]
            scale = states[:, STATE_FEATURES.index('map_scale')] * 100.
            reach = states[:, STATE_FEATURES.index('radius_map')] + (
                torch.linalg.vector_norm(states[:, 3:6], dim=1)*25.+45.)*.016/scale+.05/scale
            distance = torch.maximum(torch.maximum(self.bbox_lo[None]-p[:, None],
                p[:, None]-self.bbox_hi[None]),
                torch.zeros((), device=states.device, dtype=states.dtype))
            prior = -torch.clamp(torch.sum(distance**2, dim=-1)/reach[:, None]**2, max=40.)
            attention = torch.softmax(logits+prior, dim=-1)
            return self.model.motion(torch.cat([state, attention @ self.values], dim=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ML/'dynamics_reports/native_motion_student_expanded_20261012')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve old evidence; choose a new output')
    manifest = json.loads((args.data/'manifest.json').read_text())
    if not manifest['preparation_complete']:
        raise ValueError('Preparation incomplete')
    for name, value in manifest['source_hashes'].items():
        if sha(Path(__file__).with_name(name+'.py')) != value:
            raise ValueError('Prepared input source changed: '+name)
    for file, value in manifest['output_hashes'].items():
        if sha(args.data/file) != value:
            raise ValueError('Prepared array changed: '+file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1)
    torch.manual_seed(121014)
    model = MotionStudent().to(device).eval().requires_grad_(False)
    before = weight_digest(model)
    protected = {str(path): sha(path) for path in [
        ML/'models/nav/real_maps_20261010_A2/nav_latest.pth',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe',
        ML.parent/'Marble Blast Platinum/marbleblast_mbx.exe']}
    results = []

    def timed(fn):
        times = []
        for i in range(7):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            if device.type == 'cuda': torch.cuda.synchronize()
            if i >= 2: times.append(time.perf_counter()-start)
        return out, float(np.median(times))

    for record in manifest['map_records']:
        name = record['map']
        with np.load(args.data/(name+'.surfaces.npz'), allow_pickle=False) as geo, \
             np.load(args.data/(name+'.steps.npz'), allow_pickle=False) as steps:
            surface, lo, hi = [torch.as_tensor(geo[k], device=device, dtype=torch.float32)
                for k in ('features', 'bbox_lo', 'bbox_hi')]
            if len(surface) != record['surfaces']:
                raise ValueError('A map surface was discarded')
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            cache = CompleteMapMotionCache(model, surface, lo, hi)
            if device.type == 'cuda': torch.cuda.synchronize()
            build = time.perf_counter()-start
            for batch in (1, 16, 64):
                indexes = np.linspace(0, len(steps['x'])-1, batch, dtype=int)
                states = torch.as_tensor(steps['x'][indexes], device=device)
                with torch.inference_mode():
                    original, uncached = timed(lambda: model(states, surface, lo, hi))
                    cached, hot = timed(lambda: cache(states))
                if not torch.equal(original, cached):
                    raise ValueError('Cached output differs: '+name+'/'+str(batch))
                results.append(dict(map=name, batch=batch, surfaces=cache.surface_count,
                    exact_output_equality=True, build_s=build,
                    uncached_median_s=uncached, cached_median_s=hot,
                    uncached_over_cached=uncached/hot))
            # Own bounds remain independent of the caller's geometry tensors.
            saved = cache.bbox_lo.clone()
            lo.add_(1)
            if not torch.equal(saved, cache.bbox_lo):
                raise ValueError('Caller mutated the cached geometry')
            cache.bbox_lo.add_(1)
            try: cache(states)
            except ValueError as exc:
                if 'Cached geometry changed' not in str(exc): raise
            else: raise ValueError('Modified cached bounds were silently accepted')
        print(json.dumps(dict(map=name, exact=True,
            batches=[1,16,64], all_surfaces=record['surfaces'])), flush=True)
    if weight_digest(model) != before:
        raise ValueError('Random model weights changed during equivalence checks')
    # A real weight mutation must invalidate an already-built cache. It occurs
    # only in this local discarded random model, after unchanged-weight proof.
    with torch.no_grad(): next(model.parameters()).add_(1)
    try: cache(states)
    except ValueError as exc:
        if 'Weights changed' not in str(exc): raise
    else: raise ValueError('Stale weights were silently accepted')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('Protected artifact changed')
    atomic_json(args.output, dict(results=results, device=str(device),
        data_manifest_sha256=sha(args.data/'manifest.json'),
        source_hashes={str(p):sha(p) for p in [Path(__file__), Path(__file__).with_name('physics_student.py')]},
        random_weights_unchanged_during_equivalence=True, stale_weight_guard_passed=True,
        caller_mutation_guard_passed=True, cached_bounds_mutation_guard_passed=True,
        protected_artifacts=protected, fitting=False, gameplay_integration=False,
        limitations='Random-network single-step inference equivalence/feasibility only. No accuracy, autoregressive continuation, runtime controller or end-to-end planner speed claim.'))


if __name__ == '__main__':
    main()
