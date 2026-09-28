"""Connect to one game, report mission, round length, spawn position and contact telemetry, then exit.

    python -m nav.learned_nav.check_load --port 8981 --expect Sprawl_Hunt_phys
"""
import argparse
import json
import sys

from nav.env import HuntEnv
from nav.protocol import RAW_POS, NOOP_ACTION, CONTACT_FIELDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--expect', required=True)
    a = ap.parse_args()
    env = HuntEnv(a.port, speed=3, log=lambda m: None)
    env.connect()
    info = dict(env.info)
    for _ in range(80):                          # through the countdown
        env.step(NOOP_ACTION, repeat=1)
    env.control('CONTACT 1')
    for _ in range(3):
        msg, _ = env.step(NOOP_ACTION, repeat=1)
    tel = dict(zip(CONTACT_FIELDS, msg.extra)) if msg.extra is not None else None
    out = {'expect': a.expect, 'mission': info.get('mission'), 'round_s': float(info.get('round_ms', 0)) / 1000.0,
           'pos': [round(float(v), 2) for v in msg.obs[RAW_POS]], 'telemetry': tel,
           'ok': info.get('mission') == a.expect and float(info.get('round_ms', 0)) >= 3.5e6 and tel is not None}
    print(json.dumps(out), flush=True)
    env.close()
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
