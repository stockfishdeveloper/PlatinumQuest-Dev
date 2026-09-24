"""HANDOFF 28.25: real-gem training mode in nav/vec_worker.py (NAV_REAL_GEMS=1, default ON).
Goals are the game's own gems (nav/gems.choose, same as real rounds), the game's gem_delta is the
arrival, no teleports, and every finished round logs 'GAME map=.. points=.. gems=.. falls=..'."""
import os, ast
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nav', 'vec_worker.py')
s = open(p, encoding='utf-8').read()

def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)

rep("from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402\n",
    "from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402\n"
    "from nav.gems import visible_gems, choose, STICKY_TOL                               # noqa: E402\n")
rep("GAME_SPEED = int(os.environ.get('NAV_SPEED', '3'))\n",
    "GAME_SPEED = int(os.environ.get('NAV_SPEED', '3'))\n"
    "REAL_GEMS = os.environ.get('NAV_REAL_GEMS', '1') == '1'   # 2026-09-23 (HANDOFF 28.25): train on the GAME's gems.\n"
    "                              # Goals come from nav/gems.choose (the same chooser real rounds use), the game's\n"
    "                              # gem_delta is the arrival, there are no teleports and no episode end at a pickup,\n"
    "                              # and every finished round logs its real score: training IS the real game now.\n")
rep("        self.smooth_dir = None            # EMA state for the commanded direction\n        self.t0 = time.perf_counter()",
    "        self.smooth_dir = None            # EMA state for the commanded direction\n"
    "        self.real = False; self.target = None            # real-gem mode state\n"
    "        self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0\n"
    "        self.t0 = time.perf_counter()")
# load_map: enable real mode when the map has spawn points
i = s.index("self.segs = SegmentManager(")
j = s.index("\n", i)
line = s[s.rfind("\n", 0, i) + 1:j]
indent = line[:len(line) - len(line.lstrip())]
s = s[:j + 1] + indent + "self.real = REAL_GEMS and bool(getattr(t, 'gem_spawns', None)); self.segs.real_mode = self.real\n" + indent + "self.log(f'real-gem training mode: {self.real}')\n" + s[j + 1:]
# start_segment: real branch
rep("    def start_segment(self):\n        tb = time.perf_counter()\n        while True:\n            try:\n                self.goal = self.segs.begin(self.env)\n                break",
    "    def start_segment(self):\n        tb = time.perf_counter()\n        while True:\n            try:\n"
    "                if self.real:\n"
    "                    vis = []\n"
    "                    for _ in range(60):               # a gem is normally visible at once; wait through a spawn gap\n"
    "                        vis = visible_gems(self.env.msg.obs)\n"
    "                        if vis:\n"
    "                            break\n"
    "                        self.segs._step_checked(self.env, 1)\n"
    "                    if vis:\n"
    "                        o = self.env.msg.obs\n"
    "                        tgt, nxt = choose(vis, None, o[0:2], o[3:5])\n"
    "                        self.goal = self.segs.begin(self.env, real_goal=tgt[:3], real_next=(nxt[:3] if nxt else None))\n"
    "                        self.target = tgt\n"
    "                        break\n"
    "                self.goal = self.segs.begin(self.env)\n                break")
# step: score accumulation right after the game step
rep("        msg, info = self.env.step(js)\n        self.prof['game'] += time.perf_counter() - tg\n",
    "        msg, info = self.env.step(js)\n        self.prof['game'] += time.perf_counter() - tg\n"
    "        if self.real:\n"
    "            self.round_points += float(info['gem_delta'])\n"
    "            if info['gem_delta'] > 0:\n"
    "                self.round_gems += 1\n"
    "            if info['fell']:\n"
    "                self.round_falls += 1\n")
# round end: log the real score
rep("        if info['round_ended'] or info['reconnected']:\n            self.segs.abandon()\n            if info['round_ended']:\n                self.env.wait_new_round()",
    "        if info['round_ended'] or info['reconnected']:\n            self.segs.abandon()\n"
    "            if self.real and info['round_ended']:\n"
    "                self.log(f'GAME map={self.mission} points={self.round_points:.0f} gems={self.round_gems} falls={self.round_falls} rtf={self.env.rtf():.1f}')\n"
    "            self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0\n"
    "            if info['round_ended']:\n                self.env.wait_new_round()")
# segs.step gets the game's pickup
rep("                                          cmd_dir=(a[5], a[6]) if len(a) > 6 else (a[0], a[1]))",
    "                                          cmd_dir=(a[5], a[6]) if len(a) > 6 else (a[0], a[1]),\n"
    "                                          picked=float(info['gem_delta']) if self.real else 0.0)")
# retarget onto the game's gems before building the next observation
rep("        else:\n            tg = time.perf_counter()\n            self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal, self.segs.next_goal())",
    "        else:\n"
    "            if self.real:\n"
    "                vis = visible_gems(msg.obs)\n"
    "                if vis:\n"
    "                    tgt, nxt = choose(vis, self.target, msg.obs[0:2], msg.obs[3:5])\n"
    "                    g = self.segs.seg.goal\n"
    "                    want = (nxt[:3] if nxt else None)\n"
    "                    if tgt is not None and math.hypot(tgt[0] - g[0], tgt[1] - g[1]) > STICKY_TOL:\n"
    "                        self.segs.retarget(float(msg.obs[0]), float(msg.obs[1]), tgt[:3], want)\n"
    "                        self.goal = self.segs.seg.goal\n"
    "                        self.env.mark(self.goal[0], self.goal[1], self.goal[2])\n"
    "                    elif tgt is not None:\n"
    "                        nn = self.segs.seg.real_next\n"
    "                        if (nn is None) != (want is None) or (want is not None and math.hypot(want[0] - nn[0], want[1] - nn[1]) > STICKY_TOL):\n"
    "                            self.segs.set_next(float(msg.obs[0]), float(msg.obs[1]), want)\n"
    "                    self.target = tgt\n"
    "            tg = time.perf_counter()\n"
    "            self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal, self.segs.next_goal())")
ast.parse(s)
open(p, 'w', encoding='utf-8').write(s)
print('real-mode worker patch applied')
