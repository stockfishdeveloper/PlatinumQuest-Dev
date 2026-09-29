"""Watch the stage 4 planner at 1x on a drill map (viewing only; nothing is recorded).

    powershell -ExecutionPolicy Bypass -File nav\\learned_nav\\watch4.ps1                  # kotmjump_p0
    powershell -ExecutionPolicy Bypass -File nav\\learned_nav\\watch4.ps1 -Map kotmjump_p2

Planning takes ~0.3-0.4 s a decision, several times the 64 ms a decision lasts, so two games run: a hidden one where
the planner plays each trial in lockstep (the game waits for every reply), and the visible one, where the same start
is placed and the same replies are sent paced to 64 ms. The engine is deterministic (stage 3: repeated trials match
exactly), so the visible run is the planner's own run at real speed. The next trial is planned while one is shown.
The console prints each outcome and how far the replay ended from the planned run.
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import time

os.environ.setdefault('NAV_RENDER_EVERY', '1')

import numpy as np                                                               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import drills as DR                                         # noqa: E402
from nav.learned_nav import planner as PL                                        # noqa: E402
from nav.learned_nav import seeds as SEEDS                                       # noqa: E402
from nav.learned_nav.session import RoundOver                                    # noqa: E402
from nav.protocol import NOOP_ACTION, RAW_POS                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)            # the visible game
    ap.add_argument('--port2', type=int, required=True)           # the hidden planning game
    ap.add_argument('--map', default='kotmjump_p0')
    a = ap.parse_args()
    say = lambda m: print(m, flush=True)
    S = json.load(open(DR.starts_file(a.map, 'dev')))
    plan_d = DR.Drill(a.port2, a.map, log=say)                    # connects first (the launcher starts its game first)
    plan_d.s.env.control('RENDEREVERY 100')
    view_d = DR.Drill(a.port, a.map, log=say)
    vs = view_d.s
    vs.env.set_speed(1); vs.contact_on()
    rng = np.random.default_rng(int(time.time()) % 100000)
    planner = PL.Planner(plan_d.g, plan_d.gem, seeds=SEEDS.load(a.map), rng=rng)
    q = queue.Queue(maxsize=2)

    def plan_loop():
        order = rng.permutation(len(S['starts']))
        k = 0
        while True:
            st = S['starts'][int(order[k % len(order)])]; k += 1
            try:
                r = plan_d.trial(st, planner, S['floor_ref'])
            except RoundOver:
                continue
            if r.get('status') == 'bad_teleport':
                continue
            q.put((st, r))
    threading.Thread(target=plan_loop, daemon=True).start()

    clock = {'t': 0.0}

    def paced_step(js):
        clock['t'] = max(clock['t'] + 0.064, time.perf_counter())
        dt = clock['t'] - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return vs.step(js)

    shown = 0
    while True:
        st, r = q.get()
        o = view_d.place(st)
        if o is None:
            continue
        clock['t'] = time.perf_counter()
        for js in r['replies']:
            msg, info = paced_step(js)
            if info['fell']:
                break
        end = np.asarray(vs.obs()[RAW_POS], float)
        planned = np.asarray(r['traj'][-1][:3], float) if r['traj'] else end
        shown += 1
        say(f'{shown:3d}: start {st["id"]:2d} ({st["cat"]}, {st["speed"]:.1f} u/s, {st["dist"]:.1f} u from the gem): '
            f'{r["status"]} after {len(r["replies"]) * 0.064:.1f} s, {r["n_jumps"]} jump press(es); '
            f'replay ended {np.linalg.norm(end - planned):.3f} u from the planned run')
        for _ in range(16):                                        # a second to see how it ended
            paced_step(NOOP_ACTION)


if __name__ == '__main__':
    main()
