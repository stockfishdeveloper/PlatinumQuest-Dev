"""Exercise timed fall/recovery windows with fixed controls; no model or optimizer."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=9978)
    p.add_argument('--starts', required=True)
    p.add_argument('--out', default='logs/nav/replay_recovery_probe.json')
    args = p.parse_args()
    os.environ['NAV_DRILL_PLAN'] = '0:4'
    os.environ['NAV_DRILL_EVAL'] = '1'
    os.environ['NAV_LIVE_RECORD'] = '0'
    from nav.vec_worker import InstanceWorker
    from nav.replay import load_starts, WINDOW_MS
    from nav.protocol import RAW_SCORE
    starts = load_starts(args.starts)
    w = InstanceWorker(0, args.port, 777, .65, .6)
    w.eval_starts = starts
    rows = []
    try:
        w.connect()
        for _ in range(2):
            w.begin_replay(starts[0])
            durations = []
            while True:
                # Drive straight until falling. Recovery must remain inside the window.
                rep = w.step([0, 1, 1, 0, 0, 0, 0, 1])
                durations.append(rep['duration_steps'])
                if rep['skip']:
                    raise RuntimeError('unexpected reset')
                if rep['done']:
                    break
            row = dict(w.last_drill_result)
            row['duration_ms'] = sum(durations) * 64
            row['score_delta'] = float(w.env.msg.obs[RAW_SCORE]) - starts[0]['world']['header'][3]
            if row['elapsed_ms'] != WINDOW_MS or row['duration_ms'] != WINDOW_MS:
                raise RuntimeError('recovery lost simulation time')
            if row['points'] != row['score_delta'] or row['falls'] < 1:
                raise RuntimeError('recovery probe missed a fall or lost game points')
            rows.append(row)
        if rows[0] != rows[1]:
            raise RuntimeError('recovery windows differ across identical restores')
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print('RECOVERY PROBE passed: ' + json.dumps(rows[0]), flush=True)
    finally:
        w.env.close()


if __name__ == '__main__':
    main()
