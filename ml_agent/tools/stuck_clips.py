"""Film recorded stuck stretches in the real game (operator, 2026-10-16: "make a video of the marble getting stuck").

For each clip (a recorded round folder and a decision range) the game is put into the recorded state at the clip's
first decision: TELEPORT (position, velocity, spin), GEMSET (exactly the gems that were up), GIVEPOW (the powerup held)
and VIEWYAW (a fixed view from the marble toward the gem it was after). The recorded actions are then replayed one
64 ms decision at a time, snapping the marble back onto the recorded path whenever it is more than 0.2 u off, and the
game window is filmed after every decision (round_video.WindowRecorder, 15.625 fps = 1x). Each frame carries the
recorded round time, the planner's choice and the distance / height to the target gem. The Blast meter is not set,
so a recorded Blast may show no effect (the snap keeps the path).

usage (repo root): PYTHONPATH=ml_agent python ml_agent/tools/stuck_clips.py clips.json OUTDIR [--port 9080]
clips.json: [{"journal": "<round folder>", "start": D0, "end": D1, "title": "..."}, ...] (clips on one map share a game)
"""
import argparse
import gzip
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from nav.protocol import NOOP_ACTION
from nav.marble_dynamics.record import game, command
from nav.marble_dynamics.round_video import WindowRecorder

POW_DATABLOCK = {1: 'SuperJumpItem_MBU', 2: 'SuperSpeedItem_MBU', 5: 'HelicopterItem_MBU'}


class OverlayRecorder(WindowRecorder):
    def capture(self, lines=()):
        from PIL import ImageGrab
        import cv2
        try:
            if self.hwnd is None:
                self.start()
            image = ImageGrab.grab(bbox=self._bbox(), all_screens=True)
        except Exception:
            self.misses += 1
            return
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w - w % 2, h - h % 2)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        for k, text in enumerate(lines):
            y = 84 + 26 * k     # below the game's own timer
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.write(frame)
        self.frames += 1


def load_rows(folder):
    rows = []
    with gzip.open(Path(folder) / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    rows.sort(key=lambda d: d['decision'])
    return rows


def state_words(state):
    return ' '.join(format(float(v), '.9g') for v in [*state['position'], *state['velocity'], *state.get('spin', [0., 0., 0.])])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips', type=Path)
    p.add_argument('outdir', type=Path)
    p.add_argument('--port', type=int, default=9080)
    p.add_argument('--snap-u', type=float, default=0.2)
    a = p.parse_args()
    clips = json.loads(a.clips.read_text(encoding='utf-8'))
    a.outdir.mkdir(parents=True, exist_ok=True)
    by_map = OrderedDict()
    for n, clip in enumerate(clips, 1):
        folder = Path(clip['journal']).resolve()
        clip['n'] = n; clip['folder'] = folder; clip['map'] = folder.parts[-4]
        by_map.setdefault(clip['map'], []).append(clip)
    for name, group in by_map.items():
        report = json.loads((group[0]['folder'] / 'report.json').read_text(encoding='utf-8'))
        seed = int(report.get('native_seed') or group[0]['folder'].parts[-3])
        print(f'== {name}: {len(group)} clips', flush=True)
        with game(name, a.port, obs_ms=64, seed=seed, process_cap=8, random_start=True, watch=True) as (env, proc):
            env.set_game_rules(False)
            while not env.round_ended:
                _, outcome = env.step(NOOP_ACTION, repeat=1)
                if outcome['reconnected']:
                    raise RuntimeError('Reconnect during pre-round warmup')
            env.wait_new_round()
            env.control('FIXEDSTEP 16 64')
            env.speed = 1
            env.control('SPEED 1')
            env.control('RENDEREVERY 1')
            command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
            for _ in range(160):     # through the countdown: the marble must be live (mode 0) before it is placed
                st = ((env.msg.dynamics or {}).get('native') or {}).get('state') if isinstance(env.msg.dynamics, dict) else None
                if isinstance(st, dict) and st.get('mode', 0) == 0 and not st.get('oob'):
                    break
                env.step(NOOP_ACTION, repeat=1)
            else:
                raise RuntimeError('the marble never became live after the countdown')
            for clip in group:
                rows = load_rows(clip['folder'])
                index = {r['decision']: i for i, r in enumerate(rows)}
                i0, i1 = index[clip['start']], index[clip['end']]
                first = rows[i0]['before']
                st = first['native']['state']
                gems = [it for it in first['items'] if it.get('gem')]
                dest = (rows[i0]['choice'].get('goal_context') or {}).get('destination') or (gems[0]['position'] if gems else st['position'])
                yaw = math.atan2(float(dest[0]) - float(st['position'][0]), float(dest[1]) - float(st['position'][1]))
                env.control('VIEWYAW ' + format(yaw, '.6f'))
                env.control('GEMSET ' + ' '.join(format(float(v), '.6g') for g in gems for v in g['position']))
                held = int(st.get('held', 0))
                env.control('GIVEPOW ' + POW_DATABLOCK.get(held, 'none'))
                env.control('TELEPORT ' + state_words(st))
                live_gems = [it for it in (env.msg.dynamics or {}).get('items', []) if it.get('gem')] if getattr(env.msg, 'dynamics', None) else []
                print(f'clip {clip["n"]}: {clip.get("title", "")}: gems asked {len(gems)}, live gems now {len(live_gems)}', flush=True)
                path = a.outdir / f'{clip["n"]:02d}_{name}_{clip["start"]}.mp4'
                rec = OverlayRecorder(proc.pid, str(path))
                snaps = 0
                for i in range(i0, i1 + 1):
                    row = rows[i]; c = row['choice']
                    rs = row['before']['native']['state']
                    d = (c.get('goal_context') or {}).get('destination')
                    pos = np.asarray(rs['position'], dtype=float)
                    goal_txt = ''
                    if d is not None:
                        g = np.asarray(d[:3], dtype=float)
                        goal_txt = f'target gem {np.linalg.norm((g - pos)[:2]):.1f} u away, {g[2] - pos[2]:+.1f} u up'
                    lines = [f'{clip.get("title", name)}',
                             f'round time {row["decision"] * .064:6.1f} s   ({(row["decision"] - clip["start"]) * .064:4.1f} s into the clip)',
                             f'planner: {c.get("label")} {c.get("reason") or ""}   {goal_txt}']
                    live_state = env.msg.dynamics['native']['state']
                    live = np.asarray(live_state['position'], dtype=float)
                    drift = float(np.linalg.norm(live - pos))
                    if (a.snap_u and drift > a.snap_u and not live_state.get('oob') and not rs.get('oob')
                            and live_state.get('mode', 0) == 0 and rs.get('mode', 0) == 0):
                        target = rows[i + 1]['before']['native']['state'] if i + 1 < len(rows) else rs
                        if target.get('oob') or target.get('mode', 0) != 0:
                            target = rs
                        env.control('TELEPORT ' + state_words(target))
                        snaps += 1
                    else:
                        action = row['action']
                        _, outcome = env.step(action[:6], repeat=1, use_pow=int(action[6]),
                                              pow_yaw=action[7] if action[6] else None, use_blast=int(action[8]))
                        if outcome['round_ended'] or outcome['reconnected']:
                            print('round ended during the clip', flush=True)
                            break
                        if outcome['fell'] or row['outcome'].get('fell'):
                            env.control('OOBCLICK')
                    rec.capture(lines)
                info = rec.close()
                print(f'clip {clip["n"]} written: {info["path"]} ({info["frames"]} frames, {info["seconds"]:.1f} s, snaps {snaps}, missed {info["misses"]})', flush=True)
            for _ in range(4):
                env.step(NOOP_ACTION, repeat=1)


if __name__ == '__main__':
    main()
