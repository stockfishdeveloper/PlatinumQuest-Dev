"""The socket line format between the game (mlAgent.cs) and Python, in one place.

Game -> Python, one line per 16 ms script update:
    obs_json|gemDelta|oob|done[|humanInputs]|tick
    obs_json: 35 numbers (see RAW_* below), or [] on the round-end message
    tick:     monotonic update counter (always the last field)
Other lines the game sends (answers to control words, never need a reply):
    STATS|onTime,late,held,delay,tick
    INFO|<mission file base>|<round ms>|<time scale>

Python -> game, one line per message:
    fwd,back,left,right,jump,camYaw,usePow[,t<tick>]     an action
    SPEED n | TELEPORT x y z [vx vy vz] | STATS | INFO | DELAY n | SLEEPTIME n | MAXFPS n | RECORD
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

RAW_DIM = 35
RAW_POS = slice(0, 3)
RAW_VEL = slice(3, 6)
RAW_GEMS = slice(6, 31)          # 5 gems x [dx, dy, dz, value, dist]; absent gems have value/dist <= -500
RAW_TIME_LEFT = 31               # ms LEFT in the round (PlayGui.currentTime counts DOWN in Hunt mode)
RAW_TIME_ELAPSED = 31            # (old name kept; same field, see above)
RAW_TIME_REMAINING = 32          # MissionInfo.time - PlayGui.currentTime, i.e. ms ELAPSED
RAW_SCORE = 33
RAW_GEMS_REMAINING = 34

NOOP_ACTION = (0.0, 0.0, 0.0, 0.0, 0)


@dataclass
class GameMessage:
    kind: str                                   # 'obs' | 'end' | 'stats' | 'info' | 'other'
    obs: Optional[np.ndarray] = None            # (35,) float32 for kind == 'obs'
    gem_delta: float = 0.0
    oob: int = 0
    done: int = 0
    tick: str = ''
    fields: List[str] = field(default_factory=list)   # raw '|' fields, for stats/info/other
    raw: str = ''


def parse_message(line: str) -> GameMessage:
    line = line.strip()
    parts = line.split('|')
    head = parts[0]
    if head == 'STATS':
        return GameMessage('stats', fields=parts[1:], raw=line)
    if head == 'INFO':
        return GameMessage('info', fields=parts[1:], raw=line)
    if len(parts) < 4:
        return GameMessage('other', fields=parts, raw=line)
    try:
        obs = json.loads(head)
    except ValueError:
        return GameMessage('other', fields=parts, raw=line)
    tick = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
    try:
        gem_delta = float(parts[1]); oob = int(float(parts[2])); done = int(float(parts[3]))
    except ValueError:
        return GameMessage('other', fields=parts, raw=line)
    if len(obs) == 0:
        return GameMessage('end', gem_delta=gem_delta, oob=oob, done=done, tick=tick, fields=parts, raw=line)
    if len(obs) < RAW_DIM:
        return GameMessage('other', fields=parts, raw=line)
    return GameMessage('obs', obs=np.asarray(obs[:RAW_DIM], dtype=np.float32), gem_delta=gem_delta,
                       oob=oob, done=done, tick=tick, fields=parts, raw=line)


def format_action(fwd, back, left, right, jump, cam_yaw=0.0, use_pow=0, tick=''):
    s = f'{fwd:.6f},{back:.6f},{left:.6f},{right:.6f},{int(jump)},{cam_yaw:.6f},{int(use_pow)}'
    if tick:
        s += f',t{tick}'
    return s


def format_teleport(x, y, z, vx=0.0, vy=0.0, vz=0.0):
    return f'TELEPORT {x:.4f} {y:.4f} {z:.4f} {vx:.4f} {vy:.4f} {vz:.4f}'
