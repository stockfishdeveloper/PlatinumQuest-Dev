"""Candidate v38: relaxed terminal support gate, behind --hover-gate while it is being judged (default = pre-v38 gate,
so the offline benchmark's base runs and the source hashes of running pools are unaffected); bake in on adoption.

A plan was rejected ('no_supported_continuation') unless its last 4 forecast states touch the floor (support raycast
r + 0.10 u) with a bounce speed under 2 u/s. On tile seams and bumps the marble rolls with small hops (0.2 u): after a
KOTM pickup (v23 s2 decision 129) 30 of 38 candidates were rejected although they end on the floor, and the planner took
whichever plan happened to end touching it (often a jump plan). With --hover-gate the floor may be up to 0.6 u below the
marble and the bounce speed up to 4 u/s; a plan over a void still fails (no floor within 0.6 u), and the braking backup
still checks every useful plan for falls.
usage: python apply_hover_flag.py [ml_agent dir]
"""
import sys, os
ml = sys.argv[1] if len(sys.argv) > 1 else 'ml_agent'


def patch(path, pairs):
    s = open(path, encoding='utf-8').read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (path, n, old[:80])
        s = s.replace(old, new)
    open(path + '.new', 'w', encoding='utf-8').write(s)
    os.replace(path + '.new', path)
    print('patched', path)


patch(ml + '/nav/marble_dynamics/search.py', [
("""    speeds=[]
    for state in tail:
        p=np.asarray(state['position']);r=state['radius']
        hit=mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:return False,False,None
        n=np.asarray(hit['normal']);n=n/np.linalg.norm(n)
        speeds.append(abs(float(np.asarray(state['velocity'])@n)))
    supported=len(tail)>=4
    stable=supported and all(s<2. for s in speeds) and all(np.linalg.norm(s['velocity'])<18. for s in tail)
    return supported,stable,max(speeds,default=None)""",
"""    speeds=[]
    for state in tail:
        p=np.asarray(state['position']);r=state['radius']
        # v38 (--hover-gate): a small hop off a tile seam or bump (floor within HOVER_U below the
        # marble, bounce speed under HOVER_SPEED) counts as supported; over a void there is no floor
        hit=mesh.raycast(p+[0,0,.05],p-[0,0,r+HOVER_U],cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:return False,False,None
        n=np.asarray(hit['normal']);n=n/np.linalg.norm(n)
        speeds.append(abs(float(np.asarray(state['velocity'])@n)))
    supported=len(tail)>=4
    stable=supported and all(s<HOVER_SPEED for s in speeds) and all(np.linalg.norm(s['velocity'])<18. for s in tail)
    return supported,stable,max(speeds,default=None)"""),
("""THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
""",
"""THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
HOVER_U=.10              # terminal gate: floor within r + HOVER_U below each of the last 4 states (v38 --hover-gate: 0.6)
HOVER_SPEED=2.           # terminal gate: bounce speed along the floor normal under this (v38 --hover-gate: 4 u/s)
"""),
])
patch(ml + '/nav/marble_dynamics/reliability_eval.py', [
("""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
""",
"""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
        _search.HOVER_U,_search.HOVER_SPEED=(.6,4.) if getattr(a,'hover_gate',False) else (.10,2.)
"""),
("""    p.add_argument('--no-through-gem',action='store_true',""",
"""    p.add_argument('--hover-gate',action='store_true',help='Planner v38 (under test): terminal gate accepts a small hop (floor within r + 0.6 u, bounce speed < 4 u/s)')
    p.add_argument('--no-through-gem',action='store_true',"""),
])
print('hover gate (flag) applied')
