"""Apply HANDOFF 28.25 to nav/waypoints.py: EDGE_K 0, FALL_AFTER_JUMP 25, CONTINUOUS group chaining
(synthetic mode) with SPAWN_BLOCK_U, history record factored into _record(). Idempotent-ish: asserts on
the pre-change text, so it refuses to run twice."""
import os, shutil, ast
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nav', 'waypoints.py')
shutil.copy(p, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waypoints_backup_pre_continuous.py'))
s = open(p, encoding='utf-8').read()

def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)

rep("EDGE_K = 1.0                   # RESTORED 09:05 on 2026-09-21.",
    "EDGE_K = 0.0                   # 1.0 -> 0.0 on 2026-09-23 16:20 (HANDOFF 28.25): inside KOTM's 6.5 u centre block every\n"
    "                               # direction has a lip within 3 u, so at human speed the stopping-distance charge fired\n"
    "                               # on most decisions there and priced being FAST near the hole rather than falling.\n"
    "                               # FALL prices the fall; the stuck-breaker handles standoffs. Code and metric kept.\n"
    "                               # (superseded) RESTORED 09:05 on 2026-09-21.")
rep("FALL_AFTER_JUMP = 8.0          # 2026-09-23 (HANDOFF 28.17): price of a fall within JUMP_FALL_WINDOW decisions of",
    "FALL_AFTER_JUMP = 25.0         # 8 -> 25 on 2026-09-23 16:20 (28.25): the practice discount had done its job (takeoffs\n"
    "                               # 1.0-1.5/min, human 0.87) and falls doubled at next2 (21 per 8 rounds). A jump fall\n"
    "                               # now costs the full price again.\n"
    "                               # (superseded) 2026-09-23 (HANDOFF 28.17): price of a fall within JUMP_FALL_WINDOW decisions of")
rep("GEM_GROUP_MIN, GEM_GROUP_MAX = 4, 8   # real hunt gems spawn in groups of this size",
    "CONTINUOUS = True              # 2026-09-23 16:20 (HANDOFF 28.25), synthetic-goal mode: when a group is completed, DO NOT\n"
    "                               # end the segment and DO NOT teleport; sample the next group SPAWN_BLOCK_U..2*SPAWN_BLOCK_U\n"
    "                               # away (the game's rule) and keep position, momentum and recurrent state. Before this the\n"
    "                               # last gem of every group was a TERMINAL event followed by a teleport to rest.\n"
    "SPAWN_BLOCK_U = 30.0           # huntGems.cs: spawnBlock = 2 * $Hunt::RadiusFromGem (15)\n"
    "GEM_GROUP_MIN, GEM_GROUP_MAX = 4, 8   # real hunt gems spawn in groups of this size")
rep("                 'next_field', 'prev_dn')", "                 'next_field', 'prev_dn', 'chain_from_prev')")
rep("        self.next_field = None; self.prev_dn = None   # Dijkstra field of the NEXT gem + last distance (PROGRESS_NEXT)\n",
    "        self.next_field = None; self.prev_dn = None   # Dijkstra field of the NEXT gem + last distance (PROGRESS_NEXT)\n"
    "        self.chain_from_prev = False          # this group was entered rolling from the previous one (CONTINUOUS)\n")
rep("    def _sample_group(self, x, y):\n", "    def _sample_group(self, x, y, dmin=None, dmax=None):\n")
rep("        first = self.terrain.sample_goal(x, y, self.rng, GOAL_DMIN, GOAL_DMAX)",
    "        first = self.terrain.sample_goal(x, y, self.rng, GOAL_DMIN if dmin is None else dmin, GOAL_DMAX if dmax is None else dmax)")
old_rec = s[s.index("            self.history.append({'outcome': outcome, 'decisions': s.decisions,"):]
old_rec = old_rec[:old_rec.index("            if len(self.history) > 2000:")]
rep(old_rec, "            self._record(s, outcome)\n")
rep("    def stats(self, last=300):",
    "    def _record(self, s, outcome):\n"
    "        self.history.append({'outcome': outcome, 'decisions': s.decisions, 'path_len': s.path_len,\n"
    "                             'travelled': s.travelled, 'speed': s.travelled / (max(s.decisions, 1) * 0.064),\n"
    "                             'collected': s.collected, 'group_size': s.group_size, 'falls': s.falls,\n"
    "                             'pickup_speed': (sum(s.pickup_speeds) / len(s.pickup_speeds)) if s.pickup_speeds else 0.0,\n"
    "                             'carry_speed': (sum(s.carry_vals) / len(s.carry_vals)) if s.carry_vals else 0.0,\n"
    "                             'turn_deg': (sum(s.turn_degs) / len(s.turn_degs)) if s.turn_degs else 0.0,\n"
    "                             'chain_dec': s.chain_dec, 'chain_n': s.chain_n})\n"
    "        if len(self.history) > 2000:\n"
    "            self.history = self.history[-2000:]\n\n"
    "    def _chain_group(self, x, y, z):\n"
    "        \"\"\"CONTINUOUS (synthetic mode): the group is done; roll straight into a new one spawned the way the\n"
    "        game does it. Records the finished group, rebases the live Segment and returns True; False if no\n"
    "        group can be placed (the caller then ends the segment as before).\"\"\"\n"
    "        s = self.seg\n"
    "        g = self._sample_group(x, y, dmin=SPAWN_BLOCK_U, dmax=2.0 * SPAWN_BLOCK_U)\n"
    "        if not g:\n"
    "            return False\n"
    "        self._record(s, 'arrived')\n"
    "        gx, gy, gz, path_len = g[0]\n"
    "        s.goal = (gx, gy, gz); s.path_len = path_len; s.remaining = list(g[1:])\n"
    "        s.field = self.terrain.goal_field(gx, gy); s.prev_d = self.terrain.dist_at(s.field, x, y, (gx, gy))\n"
    "        s.start = (x, y, z); s.decisions = 0; s.travelled = 0.0; s.collected = 0; s.group_size = len(g)\n"
    "        s.falls = 0; s.fall_mark = None; s.offmap = 0; s.grace = PICKUP_GRACE\n"
    "        s.pickup_speeds = []; s.carry_vals = []; s.turn_degs = []; s.chain_dec = 0; s.chain_n = 0\n"
    "        s.chain_from_prev = True\n"
    "        self._set_next_field(x, y)\n"
    "        self.pending_mark = (gx, gy, gz)\n"
    "        return True\n\n"
    "    def stats(self, last=300):")
rep("            else:\n                done, outcome = True, 'arrived'          # whole group collected",
    "            elif CONTINUOUS and not round_ended and self._chain_group(x, y, z):\n"
    "                pass                                     # rolled into the next group, no episode end\n"
    "            else:\n                done, outcome = True, 'arrived'          # whole group collected")
rep("            if s.collected > 0:                   # a pickup-to-pickup interval, not the from-rest first gem",
    "            if s.collected > 0 or s.chain_from_prev:   # a pickup-to-pickup interval (chained groups count from gem 1)")
ast.parse(s)
open(p, 'w', encoding='utf-8').write(s)
print('continuous patch applied to', p)
