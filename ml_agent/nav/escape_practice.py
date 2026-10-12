"""Collect restorable escape practice from normal gameplay; never grant inventory."""
import copy
import json
import math
import os
from pathlib import Path
import numpy as np

from nav.obs import VEC_POW
from nav.protocol import RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL
from nav.replay import SCHEMA, validate_start, split_for
from nav.routing import goal_key


class EscapeRecorder:
    def __init__(self, directory, session, mission):
        self.directory = Path(directory) if directory else None
        self.session, self.mission = session, mission
        self.history = []
        self.last_capture = -math.inf
        self.count = 0
        self.capture_held = os.environ.get('NAV_PRACTICE_CAPTURE', 'escape') == 'held'
        self.capture_interval_ms = int(os.environ.get('NAV_HELD_CAPTURE_MS', '2000'))
        self.capture_limit = int(os.environ.get('NAV_HELD_CAPTURE_LIMIT', '0'))
        if self.capture_interval_ms <= 0 or self.capture_limit < 0:
            raise ValueError('held capture interval must be positive and limit nonnegative')

    def observe(self, env, goal, nav_goal, next_goal, vec, use_state, round_index):
        if self.directory is None:
            return
        raw = env.msg.obs
        row = dict(raw=raw.tolist(), goal=list(nav_goal), gem_goal=list(goal),
                   next=list(next_goal) if next_goal is not None else None, use=list(use_state))
        world = env.replay_world
        held = int(raw[RAW_POW_HELD])
        rays = np.asarray(vec[10:42]).reshape(16, 2)
        direction = int(round(math.atan2(float(vec[1]), float(vec[0])) / (2 * math.pi / 16))) % 16
        rise = any(rays[(direction + off) % 16, 0] < .125 and rays[(direction + off) % 16, 1] > .03
                   for off in (-1, 0, 1))
        eligible = (vec[8] > .5 and vec[7] * 20 < 2 and (rise or vec[3] > .03)
                    and (held == 1 or (held == 0 and (raw[RAW_POW_BLAST] >= .2 or raw[RAW_POW_SPECIAL] > .5))))
        interval = 8000
        if self.capture_held:
            eligible = (vec[8] > .5 and held > 0 and len(self.history) == 32
                        and (not self.capture_limit or self.count < self.capture_limit))
            interval = self.capture_interval_ms
        if eligible and world is not None and world.get('supported') and world['clock_ms'] - self.last_capture >= interval:
            group = f'{self.session}:{self.mission}:{round_index}'
            start = dict(schema=SCHEMA, id=f'{group}:{world["clock_ms"]}', split_group=group,
                         mission=self.mission, kind='held_approach' if self.capture_held else 'escape',
                         resource={1:'super_jump',2:'super_speed',3:'super_bounce',4:'shock_absorber',5:'helicopter',6:'mega',7:'special_blast'}.get(held,'blast'),
                         catalog=copy.deepcopy(env.replay_catalog), world=copy.deepcopy(world),
                         raw=raw.tolist(), goal=list(goal), next=row['next'], hist=copy.deepcopy(self.history))
            try:
                validate_start(start)
            except (ValueError, TypeError):
                pass  # round tail / unsupported effects are not replaced with invented starts
            else:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f'{self.session}_{self.mission}.jsonl'
                with path.open('a') as f:
                    f.write(json.dumps(start) + '\n')
                self.last_capture = world['clock_ms']; self.count += 1
        self.history.append(row)
        del self.history[:-32]


def assemble(directory, allowed_maps):
    """Keep source-round splits intact and exclude regression/held-out maps."""
    directory = Path(directory)
    starts = {}
    for path in sorted(directory.glob('*.jsonl')):
        for line in path.read_text().splitlines():
            start = json.loads(line)
            if start['mission'] in allowed_maps:
                validate_start(start)
                # Paired checkpoint baselines share per-map round seeds. Keep both
                # trajectories from that source round together, not one in each split.
                parts = start['split_group'].split(':')
                if len(parts) == 3 and parts[0].startswith('phase_r_'):
                    start['split_group'] = ':'.join(('phase_r_paired', *parts[1:]))
                starts[start['id']] = start
    for split in ('dev', 'eval'):
        rows = [s for s in starts.values() if split_for(s) == split]
        (directory / f'starts_{split}.json').write_text(json.dumps(rows, indent=1))
    return dict(total=len(starts), dev=sum(split_for(s) == 'dev' for s in starts.values()),
                eval=sum(split_for(s) == 'eval' for s in starts.values()))
