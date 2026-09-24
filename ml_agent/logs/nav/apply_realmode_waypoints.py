"""HANDOFF 28.25: real-gem training mode in nav/waypoints.py (SegmentManager.real_mode).
begin(real_goal=..., real_next=...) starts a segment on the game's gem with no teleport; retarget()/set_next()
follow the chooser; step(picked=gem_delta) treats the game's pickup as the arrival; no chaining, no done."""
import os, ast
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nav', 'waypoints.py')
s = open(p, encoding='utf-8').read()

def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)

rep("                 'next_field', 'prev_dn', 'chain_from_prev')", "                 'next_field', 'prev_dn', 'chain_from_prev', 'real_next')")
rep("        self.chain_from_prev = False          # this group was entered rolling from the previous one (CONTINUOUS)\n",
    "        self.chain_from_prev = False          # this group was entered rolling from the previous one (CONTINUOUS)\n"
    "        self.real_next = None                 # real-gem mode: the chooser's next gem (x, y, z) or None\n")
rep("        self._prev_cmd = None          # last commanded direction, for TURN_COST\n\n    def take_respawn(self):",
    "        self._prev_cmd = None          # last commanded direction, for TURN_COST\n"
    "        self.real_mode = False         # HANDOFF 28.25: goals are the GAME's gems (worker passes them in); the\n"
    "                                       # game's gem_delta is the arrival; no teleports, no chaining, no done at pickups\n\n"
    "    def retarget(self, x, y, goal, next_goal):\n"
    "        \"\"\"Real-gem mode: the chooser switched target (a pickup or a better gem). Rebase on the new gem.\"\"\"\n"
    "        s = self.seg\n"
    "        gx, gy, gz = goal\n"
    "        s.goal = (gx, gy, gz)\n"
    "        s.field = self.terrain.goal_field(gx, gy)\n"
    "        s.prev_d = self.terrain.dist_at(s.field, x, y, (gx, gy)); s.path_len = s.prev_d\n"
    "        s.fall_mark = None\n"
    "        self.set_next(x, y, next_goal)\n"
    "        self.pending_mark = (gx, gy, gz)\n\n"
    "    def set_next(self, x, y, next_goal):\n"
    "        s = self.seg\n"
    "        s.real_next = (float(next_goal[0]), float(next_goal[1]), float(next_goal[2])) if next_goal is not None else None\n"
    "        self._set_next_field(x, y)\n\n"
    "    def take_respawn(self):")
# next_goal(): real mode returns the chooser's next gem
rep("        s = self.seg\n        if s is None or not s.remaining:\n            return None\n        gx, gy, _ = s.goal",
    "        s = self.seg\n        if s is None:\n            return None\n        if self.real_mode:\n            return s.real_next\n        if not s.remaining:\n            return None\n        gx, gy, _ = s.goal")
# begin(): real goal path
rep("    def begin(self, env, teleport=None):", "    def begin(self, env, teleport=None, real_goal=None, real_next=None):")
rep("        if self.last_outcome == 'arrived':\n            do_tp = True",
    "        if self.last_outcome == 'arrived':\n            do_tp = True\n        if real_goal is not None:\n            do_tp = False                 # real-gem mode: the game placed the marble; never teleport")
rep("            g = self._sample_group(x, y)\n            if g is not None:\n                break",
    "            g = [(float(real_goal[0]), float(real_goal[1]), float(real_goal[2]), 0.0)] if real_goal is not None else self._sample_group(x, y)\n            if g is not None:\n                break")
rep("        self.seg.last_pos = np.array([x, y, z])\n        self._set_next_field(x, y)\n        return self.seg.goal",
    "        self.seg.last_pos = np.array([x, y, z])\n"
    "        if real_goal is not None:\n"
    "            self.seg.path_len = self.seg.prev_d; self.seg.chain_from_prev = True\n"
    "            self.seg.real_next = (float(real_next[0]), float(real_next[1]), float(real_next[2])) if real_next is not None else None\n"
    "        self._set_next_field(x, y)\n        return self.seg.goal")
# step(): picked + real-mode arrival/timeout
rep("             jumped=False, vel=(0.0, 0.0), cmd_dir=None):", "             jumped=False, vel=(0.0, 0.0), cmd_dir=None, picked=0.0):")
rep("        elif math.hypot(gx - x, gy - y) < self.arrive_r and abs(gz - z) < self.arrive_dz:",
    "        elif picked > 0 or (not self.real_mode and math.hypot(gx - x, gy - y) < self.arrive_r and abs(gz - z) < self.arrive_dz):")
rep("            if s.remaining:\n                # keep going: no teleport, no reset, momentum carries into the next gem",
    "            if self.real_mode:\n"
    "                s.grace = PICKUP_GRACE               # the worker retargets onto the game's next gem next decision\n"
    "            elif s.remaining:\n                # keep going: no teleport, no reset, momentum carries into the next gem")
rep("        elif s.decisions >= TIMEOUT_PER_GEM * (1 + len(s.remaining)):",
    "        elif (s.gem_decisions >= TIMEOUT_PER_GEM) if self.real_mode else (s.decisions >= TIMEOUT_PER_GEM * (1 + len(s.remaining))):")
ast.parse(s)
open(p, 'w', encoding='utf-8').write(s)
print('real-mode waypoints patch applied')
