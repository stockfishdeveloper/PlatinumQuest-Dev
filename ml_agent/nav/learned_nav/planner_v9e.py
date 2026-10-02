"""Ablation v9e (stage 5b): v9 (stopping room, air-brake proposals) with jumps in approach mode only as an escape
(when every program without a jump falls). Sets the base module's switches, then exposes it."""
import nav.learned_nav.planner as _base
_base.AIR_BRAKE = True
_base.APPROACH_JUMPS = 'escape'
_base.STOP_CHECK = True
from nav.learned_nav.planner import *                                             # noqa: E402,F401,F403
SEED_VERSION = 'v9'
