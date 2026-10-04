"""The seven powerups as numbers, from the engine source (marble.cc, powerups.cs, blast.cs) and the measurements of
2026-10-02 (log 38, nav/learned_nav/powdrill.py). Everything the planner and the route chooser need; the policy learns
when to use them from the game's points. Types as the observation codes them (protocol.POW_NAMES):
1 Super Jump, 2 Super Speed, 3 Super Bounce, 4 Shock Absorber, 5 Helicopter, 6 Mega Marble, 7 Blast pickup.
"""
import math

KEY_LAG = 2                  # decisions between the use key sent and the fire (the jump key too, log 36.2)

SUPER_JUMP_VZ = 20.0         # u/s added along -gravity at the fire tick (18.6 seen a decision later)
SUPER_SPEED_DV = 25.0        # u/s added along the camera yaw, projected onto the contact plane (24.1 from rest on floor)
SUPER_SPEED_YAW = lambda dx, dy: math.atan2(dx, dy)   # camera yaw facing world direction (dx, dy): forward = (sin, cos)

SUPER_BOUNCE_RESTITUTION = 0.9      # measured 0.89 (plain 0.44-0.47)
SHOCK_RESTITUTION = 0.01            # measured 0.00
HELI_GRAVITY_MULT = 0.25            # drop fall speed x0.54 measured = sqrt(0.29)
HELI_AIR_ACCEL_MULT = 2.0
EFFECT_S = {3: 5.0, 4: 5.0, 5: 5.0}  # Super Bounce, Shock Absorber, Helicopter active time
MEGA_S = 10.0                        # 5.0 in competitive Hunt ($MPPref::Server::CompetitiveMode)

MEGA_RADIUS = 0.6666                 # vs 0.18975 (MarbleData megaScale / scale)
MEGA_SPEED_FACTOR = 0.81             # floor speed after 2.3 s: 14.8 vs 18.2 u/s
MEGA_BRAKE = 17.0                    # u/s^2 (plain 14.4)
MEGA_JUMP_APEX = 1.79                # u (plain 1.33)
MEGA_POP_VZ = 6.0                    # activation within 1 u of floor: gravityImpulse "6 6 6" (pops the marble up)
MEGA_AIR_BOOST_VZ = {1: 24.7, 2: 24.7, 3: 17.4, 4: 17.4}   # +vz when activated k decisions after the jump key
MEGA_LAUNCH_SAT = 37.0               # u/s: spin launch on a high-friction floor saturates here (250+ rad/s)
MEGA_LAUNCH = {90: 8.4, 150: 18.9, 250: 37.0, 400: 37.6}    # peak speed by spin (rad/s), tarmac; plain 5.3/8.1/11.4/19.3
SPIN_UP_AIR = 44.0                   # rad/s per second of a held direction in the air; rolling spin = speed / R (98 at 17.8)

BLAST_CHARGE_S = 25.0                # the meter fills 0 -> 1 linearly
BLAST_REQUIRED = 0.2                 # usable from here
BLAST_POWER = 10.0                   # impulse = BLAST_POWER x sqrt(meter) along -gravity
BLAST_SPECIAL_POWER = 1.03           # a picked-up Blast arms the meter at 1 and fires at BLAST_POWER x this
BLAST_AFTER = 0.03                   # meter right after a blast
RESPAWN_S = 7.0                      # items reappear this long after a pickup; the same type is refused while held


def blast_vz(meter, special=False):
    """Vertical speed a blast adds now (0 if the meter is under the threshold)."""
    if special:
        return BLAST_POWER * BLAST_SPECIAL_POWER
    return BLAST_POWER * math.sqrt(meter) if meter >= BLAST_REQUIRED else 0.0


def blast_apex(meter, special=False, g=20.0):
    v = blast_vz(meter, special)
    return v * v / (2.0 * g)
