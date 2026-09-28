"""Stage 3 final run after collection: features, the step ensemble, the flight head, every evaluation.

    python -m nav.learned_nav.final3              -> logs/learned_nav/final3.txt and the result JSONs

Settings chosen on validation blocks of the training maps (2026-09-28 night): step model 768 x 4 residual blocks,
the previous reply and a fine 0.2 u crop as inputs, targets as deviations from the ballistic step, 3 bootstrap
members, 6 epochs; flight head with the start-velocity prior and a landing-decision head, 10 epochs. The held-out
development maps (Gems Ahoy, Acropolis 2) were never used to choose anything.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, 'logs', 'learned_nav', 'final3.txt')


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run(args, name):
    log(f'start {name}')
    t = time.time()
    with open(LOG, 'a') as f:
        rc = subprocess.run([sys.executable, '-u'] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                            env=dict(os.environ, STEP3_HIDDEN='768', STEP3_BLOCKS='4')).returncode
    log(f'end {name}: exit {rc}, {time.time() - t:.0f} s')
    return rc


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log('final3: stage 3 final run')
    run(['-m', 'nav.learned_nav.dynamics3', 'build'], 'step features')
    # the flight features build on the CPU while the step ensemble trains on the GPU
    fb = subprocess.Popen([sys.executable, '-u', '-c', 'from nav.learned_nav import flight3 as F; F.build(workers=10)'],
                          cwd=HERE, stdout=open(LOG.replace('.txt', '_flightbuild.txt'), 'w'), stderr=subprocess.STDOUT)
    run(['-c', 'from nav.learned_nav import dynamics3 as D; D.train(epochs=6)'], 'step ensemble training')
    fb.wait(); log(f'flight features built (exit {fb.returncode})')
    run(['-c', "from nav.learned_nav import flight3 as F; F.train(epochs=10, out_name='flight3.pth')"], 'flight head training')
    run(['-m', 'nav.learned_nav.eval3'], 'step model evaluation')
    run(['-m', 'nav.learned_nav.eval_flight3', 'flight3.pth'], 'flight head evaluation')
    run(['-m', 'nav.learned_nav.p0cross', 'flight3.pth'], 'P0 cross-check (flight head)')
    run(['-m', 'nav.learned_nav.p0cross', 'rollout'], 'P0 cross-check (step rollouts)')
    run(['-m', 'nav.learned_nav.edge_check'], 'edge release check')
    log('final3: done')


if __name__ == '__main__':
    main()
