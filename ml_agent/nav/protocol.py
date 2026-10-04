"""The socket line format between the game (mlAgent.cs) and Python, in one place.

Game -> Python, one line per 16 ms script update:
    obs_json|gemDelta|oob|done[|humanInputs]|tick
    obs_json: 38 numbers (see RAW_* below), or [] on the round-end message
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

RAW_DIM = 61                     # 35 -> 38 on 2026-09-26 (NAV_OBS_V6): the marble's spin appended; 38 -> 61 on
                                 # 2026-10-02 (NAV_OBS_V7): the powerup block appended (RAW_POW_* below)
RAW_DIM_BEFORE_SPIN = 35         # what an observer.cs without the spin fields sends (see parse_message)
RAW_POS = slice(0, 3)
RAW_VEL = slice(3, 6)
RAW_GEMS = slice(6, 31)          # 5 gems x [dx, dy, dz, value, dist]; absent gems have value/dist <= -500
RAW_TIME_LEFT = 31               # ms LEFT in the round (PlayGui.currentTime counts DOWN in Hunt mode)
RAW_TIME_ELAPSED = 31            # (old name kept; same field, see above)
RAW_TIME_REMAINING = 32          # MissionInfo.time - PlayGui.currentTime, i.e. ms ELAPSED
RAW_SCORE = 33
RAW_GEMS_REMAINING = 34
RAW_SPIN = slice(35, 38)         # angular velocity, rad/s, same frame as RAW_VEL (observer.cs collectSelfState)
# NAV_OBS_V7 (2026-10-02, POWERUP_PLAN phase 1): the powerup block, observer.cs collectPowerups. Types: 1 Super Jump,
# 2 Super Speed, 3 Super Bounce, 4 Shock Absorber, 5 Helicopter, 6 Mega Marble, 7 Blast pickup (only these seven).
RAW_DIM_V6 = 38                  # what an observer.cs without the powerup block sends (padded, see parse_message)
RAW_POW_HELD = 38                # held powerup type (0 none)
RAW_POW_BLAST = 39               # the regular blast meter 0..1 (usable from BLAST_REQUIRED; impulse 10 x sqrt(meter) up)
RAW_POW_SPECIAL = 40             # 1 while a picked-up Blast is armed (fires at BLAST_SPECIAL_POWER instead)
RAW_POW_MEGA = 41                # 1 while mega
RAW_POW_MEGA_LEFT = 42           # seconds of mega left
RAW_POW_BOUNCE_LEFT = 43         # seconds of Super Bounce left
RAW_POW_SHOCK_LEFT = 44          # seconds of Shock Absorber left
RAW_POW_HELI_LEFT = 45           # seconds of Helicopter left
RAW_POW_ITEMS = slice(46, 61)    # 3 nearest in-scope items x [type, dx, dy, dz, seconds to respawn (0 = there)]; absent -999
POW_ITEM_N = 3
POW_NAMES = ('none', 'super_jump', 'super_speed', 'super_bounce', 'shock_absorber', 'helicopter', 'mega', 'blast')
BLAST_REQUIRED = 0.2             # shared/mp/defaults.cs: the meter must reach this; it fills in BLAST_CHARGE_S
BLAST_CHARGE_S = 25.0
BLAST_POWER = 10.0               # client/scripts/blast.cs performBlast: impulse = BLAST_POWER x sqrt(meter) along -gravity
BLAST_SPECIAL_POWER = 1.03       # ... or BLAST_POWER x this for a picked-up Blast
POW_RESPAWN_S = 7.0              # item.cs / powerups.cs default respawn
# Contact telemetry (2026-09-27, jump physics stage 3): after the CONTACT 1 control word the game appends these
# 13 numbers (Marble::getContactTelemetry, covering the physics since the previous observation); they arrive in
# GameMessage.extra and never enter the 38-number observation.
CONTACT_FIELDS = ('sub_steps', 'contact_steps', 'support_steps', 'collisions', 'max_contacts', 'max_approach',
                  'nx', 'ny', 'nz', 'friction', 'restitution', 'force', 'last_step_contact')

NOOP_ACTION = (0.0, 0.0, 0.0, 0.0, 0)


@dataclass
class GameMessage:
    kind: str                                   # 'obs' | 'end' | 'stats' | 'info' | 'other'
    obs: Optional[np.ndarray] = None            # (RAW_DIM,) float32 for kind == 'obs'
    gem_delta: float = 0.0
    oob: int = 0
    done: int = 0
    tick: str = ''
    fields: List[str] = field(default_factory=list)   # raw '|' fields, for stats/info/other
    raw: str = ''
    extra: Optional[np.ndarray] = None          # numbers after the RAW_DIM observation (CONTACT telemetry), or None


def parse_message(line: str) -> GameMessage:
    line = line.strip()
    parts = line.split('|')
    head = parts[0]
    if head == 'STATS':
        return GameMessage('stats', fields=parts[1:], raw=line)
    if head == 'INFO':
        return GameMessage('info', fields=parts[1:], raw=line)
    if head == 'DEBUG':
        return GameMessage('debug', fields=parts[1:], raw=line)
    if len(parts) < 4:
        return GameMessage('other', fields=parts, raw=line)
    try:
        obs = json.loads(head)
    except ValueError:
        return GameMessage('other', fields=parts, raw=line)
    if not isinstance(obs, list):             # a partial line (seen once at connect, 2026-09-28): skip it
        return GameMessage('other', fields=parts, raw=line)
    tick = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
    try:
        gem_delta = float(parts[1]); oob = int(float(parts[2])); done = int(float(parts[3]))
    except ValueError:
        return GameMessage('other', fields=parts, raw=line)
    if len(obs) == 0:
        return GameMessage('end', gem_delta=gem_delta, oob=oob, done=done, tick=tick, fields=parts, raw=line)
    if len(obs) < RAW_DIM:
        if len(obs) >= RAW_DIM_BEFORE_SPIN:
            # An old observer.cs.dso without the spin fields. Failing loudly beats training on missing spin
            # (or hanging, which is what an 'other' message here would do: every observation would be skipped).
            raise ValueError(f'the game sends {len(obs)} observation numbers but NAV_OBS_V7 needs {RAW_DIM} '
                             f'(spin and powerups). Delete client/scripts/ai/observer.cs.dso so the game recompiles observer.cs.')
        return GameMessage('other', fields=parts, raw=line)
    return GameMessage('obs', obs=np.asarray(obs[:RAW_DIM], dtype=np.float32), gem_delta=gem_delta,
                       oob=oob, done=done, tick=tick, fields=parts, raw=line,
                       extra=np.asarray(obs[RAW_DIM:], dtype=np.float64) if len(obs) > RAW_DIM else None)


def format_action(fwd, back, left, right, jump, cam_yaw=0.0, use_pow=0, tick='', pow_yaw=None, use_blast=0):
    """pow_yaw (2026-10-02): the camera yaw the held powerup is fired along on a use tick (Super Speed boosts along
    the marble's camera yaw: world direction (dx, dy) -> yaw atan2(dx, dy)); use_blast: fire the regular blast.
    Both go as words 8 and 9, written only when used so old game scripts still parse the line."""
    s = f'{fwd:.6f},{back:.6f},{left:.6f},{right:.6f},{int(jump)},{cam_yaw:.6f},{int(use_pow)}'
    if pow_yaw is not None or use_blast:
        s += f',{(cam_yaw if pow_yaw is None else pow_yaw):.6f},{int(use_blast)}'
    if tick:
        s += f',t{tick}'
    return s


def format_teleport(x, y, z, vx=0.0, vy=0.0, vz=0.0, spin=None):
    """spin = (wx, wy, wz) rad/s sets the marble's angular velocity (mlAgent.cs words 7-9, 2026-09-26);
    without it the game zeroes the spin, as before."""
    # 6 decimals (2026-09-28, was 4): the marble's reported state is now full precision (observer.cs), so a replayed
    # state can match it
    s = f'TELEPORT {x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}'
    if spin is not None:
        s += f' {spin[0]:.6f} {spin[1]:.6f} {spin[2]:.6f}'
    return s
