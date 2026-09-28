"""Stage 3 M2 item: calibrate the step ensemble's predicted spread.

    python -m nav.learned_nav.calibrate3        -> models/learned_nav/step3_calibration.json

The combined spread (the members' own spread and their disagreement) was too wide: 85-94 % of validation errors fell
inside one predicted standard deviation instead of 68 %. One scale per output dimension is fitted on the validation
blocks of the training maps (the 68.3 % quantile of |error| / predicted std) and checked on the two held-out maps,
which are not used for the fit.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D                                     # noqa: E402
from nav.learned_nav import eval3 as E                                         # noqa: E402

OUT = os.path.join(D.MODEL_DIR, 'step3_calibration.json')


def coverage(pr, T, scale):
    y = T[:, :D.N_CONT]
    std = np.sqrt(pr['std_alea'] ** 2 + pr['std_epi'] ** 2) * scale
    z = np.abs(pr['mu'] - y) / std
    return (z <= 1).mean(0), (z <= 2).mean(0)


def main():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = D.load_ensemble(dev)
    Fva, Tva = D.load_features(D.TRAIN_MAPS, 'val')
    pr = E.predict(models, Fva, dev)
    std = np.sqrt(pr['std_alea'] ** 2 + pr['std_epi'] ** 2)
    ratio = np.abs(pr['mu'] - Tva[:, :D.N_CONT]) / std
    scale = np.quantile(ratio, 0.683, axis=0)
    out = {'scale': scale.round(4).tolist(), 'fit_on': 'validation blocks of the training maps',
           'validation': {'before_1std': coverage(pr, Tva, 1.0)[0].round(3).tolist(), 'after_1std': coverage(pr, Tva, scale)[0].round(3).tolist(),
                          'after_2std': coverage(pr, Tva, scale)[1].round(3).tolist()}}
    for m in D.DEV_MAPS:
        d = np.load(os.path.join(D.FEAT_DIR, f'{m}.npz'))
        p2 = E.predict(models, d['F'], dev)
        b1, _ = coverage(p2, d['T'], 1.0); a1, a2 = coverage(p2, d['T'], scale)
        out[m] = {'before_1std': b1.round(3).tolist(), 'after_1std': a1.round(3).tolist(), 'after_2std': a2.round(3).tolist()}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
