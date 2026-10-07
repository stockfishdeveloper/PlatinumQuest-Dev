"""Summarize authoritative full-round Super Speed gates; never trains or stops games."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t


def read_rounds(path):
    marker = Path(str(path).removesuffix('.csv.rounds.jsonl') + '.invalid.json')
    if marker.exists():
        return [], [{'reason': 'entire attempt invalidated', 'marker': str(marker)}]
    valid, rejected = [], []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected.append({'line': line_number, 'reason': 'incomplete or invalid JSON'})
            continue
        if not row.get('completed') or row.get('engine_points') != row.get('points'):
            rejected.append({'line': line_number, 'reason': 'incomplete round or score mismatch'})
            continue
        valid.append(row)
    return valid, rejected


def summarize(path):
    rows, rejected = read_rounds(path)
    points = [r['engine_points'] for r in rows]
    opportunities = Counter()
    peaks = []
    for r in rows:
        opportunities.update(r.get('ss_opportunities', {}))
        kicks = [math.hypot(e['velocity_after'][0] - e['velocity_before'][0],
                            e['velocity_after'][1] - e['velocity_before'][1])
                 for e in r.get('ss_events', [])]
        if (r['engine_points'] > 175 and not r.get('no_use', True)
                and r.get('ss_fires', 0) and max(kicks, default=0) > 10):
            peaks.append({'round': r['round'], 'points': r['engine_points'], 'ss_fires': r['ss_fires'],
                          'largest_observed_kick_dv': max(kicks), 'checkpoint': r['checkpoint'],
                          'force_use': r.get('force_use'), 'ss_route': r.get('ss_route', False),
                          'ss_prepare': r.get('ss_prepare', False), 'ss_follow': r.get('ss_follow', False),
                          'ss_command_aim': r.get('ss_command_aim', False)})
    result = {'path': str(path), 'n': len(rows), 'rejected': rejected,
              'mean': statistics.mean(points) if points else None, 'best': max(points, default=None),
              'se': statistics.stdev(points) / math.sqrt(len(points)) if len(points) > 1 else None,
              'fires': sum(r.get('ss_fires', 0) for r in rows), 'falls': sum(r['falls'] for r in rows),
              'opportunities': dict(opportunities), 'verified_peak_milestones': peaks}
    checkpoints = {}
    for name in sorted({r['checkpoint'] for r in rows}):
        file = Path(__file__).resolve().parents[1] / 'models' / 'nav' / name
        checkpoints[name] = hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
    result['checkpoint_sha256'] = checkpoints
    return result, points


def compare(a, b):
    """Welch interval for separate rounds, not a paired replay estimate."""
    if min(len(a), len(b)) < 2:
        return None
    va = statistics.variance(a) / len(a)
    vb = statistics.variance(b) / len(b)
    delta = statistics.mean(a) - statistics.mean(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1)) if se else math.inf
    margin = float(student_t.ppf(.975, df)) * se if se else 0.
    return {'a_minus_b': delta, 'se': se, 'ci95': [delta - margin, delta + margin],
            'method': 'Welch, independent rounds; no correction for repeated checkpoint screening'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='+')
    p.add_argument('--out')
    args = p.parse_args()
    arms = [summarize(path) for path in args.paths]
    report = {'arms': [a for a, _ in arms]}
    if len(arms) == 2:
        report['comparison'] = compare(arms[0][1], arms[1][1])
    data = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(data)
    print(data)


if __name__ == '__main__':
    main()
