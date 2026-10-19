"""Candidate v38: corner-speed through proposals (behind --corner-proposals while under test; bake in on adoption).

v23 arrives at every gem at >= 6 u/s (the pickup floor) or at full speed (through proposals), whatever the turn to the
next gem: a reversal then takes 3.3 s for a 7 u leg (1.05 s straight on), ~27 s a round in all. Two more through
proposals brake to the turn's speed at the gem, cap * cos^2(turn / 2) (floor 1.5 u/s; turns under 30 degrees skipped),
then follow the route to the next gem; the existing pickup scoring decides.
usage: python apply_corner.py [ml_agent dir]
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
("""THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
""",
"""THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
CORNER_PROPOSALS=False   # v38 (--corner-proposals, under test): through proposals that reach the gem at the turn's speed
"""),
("""    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False,blast=-1,through=None):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps""",
"""    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False,blast=-1,through=None,arrive=None):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps
        self.arrive=arrive   # v38: speed at the gem for a corner (through proposals only)"""),
("""        if self.through is not None:
            # the gem needs no stop: brake only for what follows it
            base=speed_distance if speed_distance is not None else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
            speed_distance=base+self.through[2]""",
"""        min_speed=PICKUP_MIN_SPEED if not (self.brake_after is not None and index>=self.brake_after) else 0.
        if self.through is not None:
            # the gem needs no stop: brake only for what follows it
            base=speed_distance if speed_distance is not None else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
            if self.arrive is not None:
                # v38 corner: the stopping law brought down to self.arrive at the gem (not to 0)
                speed_distance=base+self.arrive*self.arrive/(2*_gem_demo.BRAKE_DECEL);min_speed=0.
            else:speed_distance=base+self.through[2]"""),
("""                          speed_distance=speed_distance,min_speed=PICKUP_MIN_SPEED if not (self.brake_after is not None and index>=self.brake_after) else 0.)""",
"""                          speed_distance=speed_distance,min_speed=min_speed)"""),
("""                    ('through_0.7',SequenceProposal(route,gem,steps,scale=.7,through=through))]""",
"""                    ('through_0.7',SequenceProposal(route,gem,steps,scale=.7,through=through))]
            arrive=corner_speed(route,dynamics['native']['state']['position'],gem,through) if CORNER_PROPOSALS else None
            if arrive is not None:
                specs+=[('through_corner',SequenceProposal(route,gem,steps,through=through,arrive=arrive)),
                        ('direct_through_corner',SequenceProposal(route,gem,steps,direct=True,through=through,arrive=arrive))]"""),
("""class SequenceProposal:""",
"""def corner_speed(route,position,gem,through):
    \"\"\"v38: speed at the gem for the turn toward the next gem, cap * cos^2(turn / 2) (full speed straight on, the
    1.5 u/s floor for a reversal); None for turns under 30 degrees (the plain through proposals cover them).
    Approach = marble -> gem, exit = gem -> the next gem's route waypoint.\"\"\"
    nroute,ngem,leg=through;g=np.asarray(gem[:3],dtype=float);p=np.asarray(position,dtype=float)
    a=g[:2]-p[:2]
    try:wp,_=path_waypoint(route.terrain,float(g[0]),float(g[1]),nroute.field,ngem,mz=float(g[2]))
    except Exception:wp=ngem
    b=np.asarray(wp[:2],dtype=float)-g[:2]
    na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    if na<.5 or nb<.5:return None
    c=float(a@b)/(na*nb)
    if c>math.cos(math.radians(30)):return None
    return max(1.5,_gem_demo.TARGET_SPEED_CAP*(1+c)/2)


class SequenceProposal:"""),
("""from nav.routing import snap_to_walkable,RoutePlanner,goal_key
""",
"""from nav.routing import snap_to_walkable,RoutePlanner,goal_key,path_waypoint
import nav.marble_dynamics.gem_demo as _gem_demo
"""),
])
patch(ml + '/nav/marble_dynamics/reliability_eval.py', [
("""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
""",
"""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
        _search.CORNER_PROPOSALS=bool(getattr(a,'corner_proposals',False))
"""),
("""    p.add_argument('--no-through-gem',action='store_true',""",
"""    p.add_argument('--corner-proposals',action='store_true',help='Planner v38 (under test): through proposals that slow to the turn speed at the gem')
    p.add_argument('--no-through-gem',action='store_true',"""),
])
print('corner proposals applied')
