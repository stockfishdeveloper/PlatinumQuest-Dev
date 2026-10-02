"""Ablation v9s (stage 5b): v9 with only the stopping-room check (no air-brake proposals, no flight check for jumps
without a pickup plan, approach jumps as in v8). Sets the base module's switches, then exposes it."""
import nav.learned_nav.planner as _base
_base.AIR_BRAKE = False
_base.FS_APPROACH = 0.0
_base.APPROACH_JUMPS = 'robust'
_base.STOP_CHECK = True
from nav.learned_nav.planner import *                                             # noqa: E402,F401,F403
SEED_VERSION = 'v9'
