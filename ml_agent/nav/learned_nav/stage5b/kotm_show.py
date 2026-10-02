"""A real KOTM round, navigator + planner with the planner as jump-happy as it gets, recorded to a real-speed video.

    python nav/learned_nav/stage5b/kotm_show.py --port 9621
    (then: marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport 9621; keep its window un-minimized)

The planner needs ~0.4 s a decision, so live play stutters whenever it drives. Instead one frame of the game window
is grabbed after every 64 ms decision (wincap.py: works while other windows cover it) and written at 15.625 fps, so
the video plays the round at true speed. A red banner marks the decisions the planner sent.
Settings (this viewer only): route jump legs on, shortcuts taken at any predicted gain, rescue on.
Output: logs/learned_nav/kotm_show_<time>.mp4 (and the usual round log under the tag 'show').
"""
import argparse
import os
import sys
import time

os.environ['NAV_RENDER_EVERY'] = '1'
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2                                                                        # noqa: E402
import wincap                                                                     # noqa: E402
from nav.learned_nav import hybrid as HY                                          # noqa: E402
from nav.learned_nav import route as RT                                           # noqa: E402
from nav.learned_nav import planner as PL                                         # noqa: E402
from nav.learned_nav.session import Session                                       # noqa: E402

# jump-happy: any predicted gain is enough
RT.ROUTE_GAIN_S = 0.0
RT.JUMP_EXTRA_S = 0.1
HY.SHORTCUT_MARGIN_S = 0.0
HY.SHORTCUT_GAIN_U = 1.5

state = {'hwnd': None, 'vw': None, 'plan_js': None, 'n': 0, 'planner_n': 0, 'jumps': 0}
_step = Session.step
_plan = PL.Planner.plan


def _plan_rec(self, *a, **k):
    js, info = _plan(self, *a, **k)
    state['plan_js'] = tuple(round(float(q), 4) for q in js[:5])
    return js, info


def _step_rec(self, js):
    by_planner = state['plan_js'] is not None and tuple(round(float(q), 4) for q in js[:5]) == state['plan_js']
    state['plan_js'] = None
    out = _step(self, js)
    if state['hwnd'] is None:
        pid = find_game_pid()
        state['hwnd'] = wincap.window_of_pid(pid) if pid else None
    if state['hwnd'] is not None:
        img = wincap.grab(state['hwnd'])
        if state['vw'] is None:
            h, w = img.shape[:2]
            path = os.path.join(HERE, 'logs', 'learned_nav', time.strftime('kotm_show_%H%M.mp4'))
            state['path'] = path
            state['vw'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 1000.0 / 64.0, (w, h))
        if by_planner:
            state['planner_n'] += 1
            cv2.rectangle(img, (img.shape[1] // 2 - 170, 70), (img.shape[1] // 2 + 170, 118), (0, 0, 200), -1)
            label = 'PLANNER: JUMP!' if int(js[4]) else 'PLANNER'
            if int(js[4]):
                state['jumps'] += 1
            cv2.putText(img, label, (img.shape[1] // 2 - 150, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        state['vw'].write(img)
        state['n'] += 1
    return out


def find_game_pid():
    import subprocess
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq marbleblast_mbx.exe', '/FO', 'CSV', '/NH'],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        parts = line.split('","')
        if len(parts) > 1 and 'marbleblast' in parts[0]:
            return int(parts[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    a = ap.parse_args()
    PL.Planner.plan = _plan_rec
    Session.step = _step_rec
    try:
        HY.play(a.port, 'KingOfTheMarble_Hunt', 1, 'current', 'show', lambda m: print(m, flush=True),
                shortcuts=1, rescue=True, route_on=True)
    finally:
        if state['vw'] is not None:
            state['vw'].release()
            print('video: %s, %d frames (%.0f s), planner frames %d, planner jump presses %d' % (
                state['path'], state['n'], state['n'] * 0.064, state['planner_n'], state['jumps']), flush=True)


if __name__ == '__main__':
    main()
