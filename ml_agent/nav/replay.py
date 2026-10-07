"""Versioned, real-game Hunt fixtures and matched-window accounting (no torch)."""
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = 2
REWARD_VERSION = 'game_points_v1'
WINDOW_MS = 12032                 # exactly 188 ordinary 64 ms decisions
APPROACH_MS = 1536


def validate_start(start, window_ms=WINDOW_MS):
    if start.get('schema') != SCHEMA:
        raise ValueError('legacy synthetic start: recollect with REPLAYCAPTURE; it has no restorable game state')
    world = start['world']
    if not world.get('supported') or world.get('schema') != SCHEMA:
        raise ValueError('unsupported physical/game state')
    if not start.get('id') or not start.get('split_group'):
        raise ValueError('start needs a stable id and round-level split_group')
    catalog = start['catalog']
    if len(catalog) != len(world['items']) or not catalog:
        raise ValueError('catalog/item state mismatch')
    if len(world['pose']) != 9 or len(world['header']) != 9:
        raise ValueError('incomplete pose or game header')
    if world['header'][1] < window_ms + 1000:
        raise ValueError('not enough round time for the entire evaluation window')
    if not re.fullmatch(r'[A-Za-z0-9_]+', world['held']):
        raise ValueError('invalid held datablock')
    numbers = world['pose'] + world['header'] + world['groups']
    for row in world['items']:
        if len(row) != 4:
            raise ValueError('invalid item state')
        numbers += row
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in numbers):
        raise ValueError('non-finite fixture state')
    if len(start['goal']) != 3 or len(start['raw']) != 61:
        raise ValueError('missing original target/observation')
    return start


def restore_command(start, window_ms=WINDOW_MS):
    validate_start(start, window_ms)
    w = start['world']
    words = lambda xs: ' '.join(format(float(x), '.12g') for x in xs)
    return 'REPLAYRESTORE ' + '|'.join((words(w['header']), words(w['pose']), w['held'],
                                      ';'.join(words(row) for row in w['items']), words(w['groups'])))


def restore(env, start, window_ms=WINDOW_MS):
    """One requested fixture, one restoration attempt. Never substitute a different id."""
    validate_start(start, window_ms)
    if env.replay_catalog != start['catalog']:
        raise ValueError('mission catalog differs from the recorded fixture')
    env.debug = None
    env.control(restore_command(start, window_ms))
    from nav.protocol import NOOP_ACTION, RAW_SCORE
    for _ in range(16):
        if env.debug and env.debug[0] == 'replay_error':
            raise RuntimeError('restore rejected: ' + '|'.join(env.debug[1:]))
        if env.debug and env.debug[0] == 'replay_restored':
            break
        env.step(NOOP_ACTION, repeat=1)
    else:
        raise RuntimeError('restore acknowledgment missing')
    actual, expected = env.replay_world, start['world']
    if actual is None:
        raise RuntimeError('restore returned no world state')
    err = max(abs(a - b) for a, b in zip(actual['pose'], expected['pose']))
    if err > 0.02:
        raise RuntimeError(f'restored position/velocity/spin differs by {err:.4f}')
    if actual['held'] != expected['held'] or actual['items'] != expected['items']:
        raise RuntimeError('restored inventory/items differ from fixture')
    if actual['groups'] != expected['groups'] or actual['header'][:8] != expected['header'][:8]:
        raise RuntimeError('restored clock, spawn state, RNG, or blast differs from fixture')
    if abs(float(env.msg.obs[RAW_SCORE]) - expected['header'][3]) > 0.01:
        raise RuntimeError('restored game score differs from fixture')
    return actual['clock_ms']


def split_for(start):
    # Nearby events in one round share a split; no overlapping rollout leakage.
    key = start['split_group'].encode()
    return 'eval' if int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % 4 == 0 else 'dev'


def load_starts(path, window_ms=WINDOW_MS):
    starts = json.loads(Path(path).read_text(encoding='utf-8'))
    seen = set()
    for s in starts:
        validate_start(s, window_ms)
        if s['id'] in seen:
            raise ValueError('duplicate start id: ' + s['id'])
        seen.add(s['id'])
    if not starts:
        raise ValueError('empty real-game fixture set; collect fresh starts first')
    return starts


def paired_summary(rows, reference='no_use'):
    """Retain failures and report paired estimates, grouping related round starts.

    Legacy rows without split_group retain their original per-start summary.
    Grouped estimates weight source rounds equally, after averaging their starts.
    """
    import statistics
    by = {}
    for r in rows:
        branch = by.setdefault(r['branch'], {})
        if r['id'] in branch:
            raise ValueError('duplicate evaluation id/branch')
        branch[r['id']] = r
    common = set.intersection(*(set(d) for d in by.values())) if by else set()
    complete = sorted(k for k in common if all(d[k]['status'] == 'ok' for d in by.values()))
    out = {'attempts': len(rows), 'paired_n': len(complete), 'paired_ids': complete,
           'failed': [r for r in rows if r['status'] != 'ok'], 'comparisons': {}}
    if reference not in by:
        return out
    groups = {}
    for k in complete:
        labels = {d[k].get('split_group') for d in by.values()}
        if len(labels) > 1:
            raise ValueError('inconsistent source round for evaluation id: ' + k)
        group = next(iter(labels))
        if group is not None:
            groups.setdefault(group, []).append(k)
    grouped = len(complete) > 0 and sum(map(len, groups.values())) == len(complete)
    if grouped:
        out['round_grouped'] = {'n': len(groups), 'comparisons': {},
                                'method': 'equal-weight source-round means of paired start differences'}
    for arm in by:
        if arm == reference:
            continue
        metrics = {}
        for key in ('points', 'falls'):
            ds = [by[arm][k][key] - by[reference][k][key] for k in complete]
            metrics[key] = {'mean': statistics.mean(ds) if ds else None,
                            'se': statistics.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 1 else None}
        out['comparisons'][arm + '-' + reference] = metrics
        if grouped:
            group_metrics = {}
            for key in ('points', 'falls'):
                ds = [statistics.mean(by[arm][k][key] - by[reference][k][key] for k in ids)
                      for ids in groups.values()]
                group_metrics[key] = {'mean': statistics.mean(ds),
                                      'se': statistics.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 1 else None}
            out['round_grouped']['comparisons'][arm + '-' + reference] = group_metrics
    return out
