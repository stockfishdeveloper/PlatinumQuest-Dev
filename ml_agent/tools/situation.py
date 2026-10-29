"""Situations from real-time rounds (2026-10-29, operator: pause a 1x round, get that exact moment set up again, play it
yourself and show the planner a better way). A situation is one decision of a real-time journal
(dynamics_reports/realtime/<date>/<map>_<stamp>.jsonl): the map, the marble's position, velocity and spin, the held
powerup, the gems in play, the Blast meter and the round clock, plus what the planner itself did from there (its next
pickups) for the comparison. record_situation.py sets it up in a game you control (play_situation.ps1).

usage (repo root, PYTHONPATH=ml_agent):
  python ml_agent/tools/situation.py capture [--journal PATH] [--seconds-back S | --decision N] [--out PATH]
  python ml_agent/tools/situation.py show PATH
capture takes the newest real-time journal by default and its last decision (the pause), or S seconds before it."""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
JOURNALS = ML / 'dynamics_reports' / 'realtime'
SITUATIONS = ML / 'situations'
ITEMS = {1: 'SuperJumpItem', 2: 'SuperSpeedItem', 3: 'SuperBounceItem', 4: 'ShockAbsorberItem', 5: 'HelicopterItem'}   # powerUpId
AFTER_S = 15.             # the planner's own play recorded after the moment (pickups within this long)


def newest_journal():
    files = sorted(glob.glob(str(JOURNALS / '*' / '*.jsonl')), key=os.path.getmtime)
    if not files:
        raise SystemExit('no real-time journal under ' + str(JOURNALS))
    return Path(files[-1])


def map_name(journal):
    m = re.match(r'(.+?)_\d{8}_\d{6}\.jsonl$', Path(journal).name)
    if not m:
        raise SystemExit('not a real-time journal name: ' + str(journal))
    return m.group(1)


def journal_rows(journal):
    return [json.loads(line) for line in open(journal, encoding='utf-8') if line.strip()]


def playing(journal):
    return [r for r in journal_rows(journal) if r.get('label') != 'countdown']


def round_length(journal):
    """The round length (s): the clock during Ready/Set (the journal's countdown lines)."""
    countdown = [r['time_left_s'] for r in journal_rows(journal) if r.get('label') == 'countdown']
    return round(max(countdown), 1) if countdown else None


def pick(rows, decision=None, seconds_back=None):
    if not rows:
        raise SystemExit('the journal has no played decisions yet')
    if decision is not None:
        if not 0 <= decision < len(rows):
            raise SystemExit('decision %d outside 0..%d' % (decision, len(rows) - 1))
        return decision
    if seconds_back:
        want = rows[-1]['time_left_s'] + float(seconds_back)
        return min(range(len(rows)), key=lambda i: abs(rows[i]['time_left_s'] - want))
    return len(rows) - 1


def planner_after(rows, i):
    """The planner's pickups after decision i (seconds after the moment, points, where), within AFTER_S."""
    t0 = rows[i]['time_left_s']; out = []
    for r in rows[i + 1:]:
        t = t0 - r['time_left_s']
        if t > AFTER_S:
            break
        if r.get('gem_delta'):
            out.append(dict(t_s=round(t, 2), points=r['gem_delta'], position=[round(v, 2) for v in r['pos'][:3]]))
    return out


def capture(journal=None, decision=None, seconds_back=None, out=None):
    journal = Path(journal) if journal else newest_journal()
    rows = playing(journal); i = pick(rows, decision, seconds_back); r = rows[i]
    if 'gems' not in r:
        raise SystemExit('decision %d has no gem list: the journal predates JOURNAL_VIEW' % i)
    held = int(r.get('held') or 0)
    sit = dict(map=map_name(journal), source=str(journal), decision=i, time_left_s=r['time_left_s'], round_s=round_length(journal),
               position=r['pos'][:3], velocity=r['pos'][3:6], spin=r.get('spin'), held=held, held_item=ITEMS.get(held),
               blast=r.get('blast'), gems=r['gems'], planner_plan=r.get('label'), planner_after=planner_after(rows, i),
               captured=time.strftime('%Y-%m-%d %H:%M:%S'))
    if out is None:
        SITUATIONS.mkdir(parents=True, exist_ok=True)
        out = SITUATIONS / ('%s_%s_d%d.json' % (sit['map'], time.strftime('%Y%m%d_%H%M%S'), i))
    Path(out).write_text(json.dumps(sit, indent=1), encoding='utf-8')
    return Path(out), sit


def describe(sit):
    p, v = sit['position'], sit['velocity']
    lines = ['%s, decision %d (%.1f s left): marble at (%.1f, %.1f, %.1f), moving %.1f u/s, holding %s, %d gems up, Blast %s'
             % (sit['map'], sit['decision'], sit['time_left_s'], p[0], p[1], p[2], (v[0] ** 2 + v[1] ** 2) ** .5,
                sit['held_item'] or 'nothing', len(sit['gems']), sit['blast']),
             'planner from there: ' + (', '.join('%+.1f s %g pt' % (a['t_s'], a['points']) for a in sit['planner_after'])
                                      or 'no pickup within %.0f s' % AFTER_S)]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--journal'); c.add_argument('--decision', type=int); c.add_argument('--seconds-back', type=float)
    c.add_argument('--out')
    s = sub.add_parser('show'); s.add_argument('path')
    a = ap.parse_args()
    if a.cmd == 'capture':
        path, sit = capture(a.journal, a.decision, a.seconds_back, a.out)
        print(path); print(describe(sit))
    else:
        print(describe(json.loads(Path(a.path).read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
