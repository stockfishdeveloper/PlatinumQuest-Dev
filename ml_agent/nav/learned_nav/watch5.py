"""Watch a whole kotmjump round at 1x: the navigator drives, the planner takes the floating gems (hybrid.py, memory
kept current). Viewing only (the round is logged under the tag 'watch').

    powershell -ExecutionPolicy Bypass -File nav\\learned_nav\\watch5.ps1

Every frame is drawn and each decision is paced to 64 ms of wall time. The navigator's decisions take milliseconds, so
its driving plays at normal speed; the planner needs ~0.3-0.4 s a decision, so while it owns the marble (the run-up,
jump and landing at each floating gem) the round plays in slow motion. The game waits for every reply (lockstep), so
nothing is skipped or changed by the slowdown.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('NAV_RENDER_EVERY', '1')

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import hybrid as HY                                         # noqa: E402
from nav.learned_nav.session import Session                                      # noqa: E402

_init, _step = Session.__init__, Session.step
_clock = {'t': None}


def _paced_init(self, *a, **k):
    _init(self, *a, **k)
    self.env.set_speed(1)
    self.contact_on()


def _paced_step(self, js):
    now = time.perf_counter()
    _clock['t'] = now if _clock['t'] is None else max(_clock['t'] + 0.064, now)
    if _clock['t'] > now:
        time.sleep(_clock['t'] - now)
    return _step(self, js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--map', default='kotmjump')           # e.g. KingOfTheMarble_Hunt (tag watch_<map>)
    a = ap.parse_args()
    Session.__init__, Session.step = _paced_init, _paced_step
    HY.play(a.port, a.map, a.rounds, 'current', 'watch', lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
