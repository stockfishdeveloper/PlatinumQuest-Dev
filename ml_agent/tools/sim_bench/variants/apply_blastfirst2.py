"""Candidate v39: Blast climb families first below a nearby raised gem (behind --raised-blast-first while under test).

v37 Concentric s0 spent 69 s within 4 u (horizontally) of a gem 4.4 u above the floor, the Blast meter full for the
last 30 s of it: an unbudgeted search finds 'blast_7_jump_3_landing' (pickup at reply 13), but the Blast families come
last in the search order and the 2 s deadline cut them off every time (24 of 53 candidates judged). The detour rule
(Blast families first) only fires when the walk route is 1.5x the straight distance, which is never true right below a
gem (straight < 4 u counts as ratio 1). Now a gem >= RAISED_BLAST_RISE_U above the marble and within RAISED_BLAST_NEAR_U
horizontally also puts the Blast families first (same meter condition as the detour rule), like v21 did for a held
Super Jump / Helicopter.
usage: python apply_blastfirst.py [ml_agent dir]
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
("""    elif detour_ratio(route,dynamics['native']['state']['position'],gem)>=float(getattr(engine,'detour_blast_ratio',1.5) or 1.5)             and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False):""",
"""    elif (detour_ratio(route,dynamics['native']['state']['position'],gem)>=float(getattr(engine,'detour_blast_ratio',1.5) or 1.5)
            or (RAISED_BLAST_FIRST and raised_near(dynamics['native']['state']['position'],gem)))             and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False):"""),
("""    climbing=bool(getattr(engine,'crossing_phase',False) and (getattr(engine,'crossing',None) or {}).get('climb'))""",
"""    climbing=bool(getattr(engine,'crossing_phase',False) and (getattr(engine,'crossing',None) or {}).get('climb'))
    if (RAISED_BLAST_FIRST and not climbing and raised_near(dynamics['native']['state']['position'],gem)
            and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False)):
        # v39: below a nearby raised gem with the meter charged, the climb families (a jump every 2 replies, the Blast 2
        # replies after it while rising, direct aim, half-throttle variants) go first, as for a climb crossing
        climbing=True"""),
("""def detour_ratio(route,position,gem):""",
"""RAISED_BLAST_FIRST=False  # v39 (--raised-blast-first, under test): Blast families first below a nearby raised gem
RAISED_BLAST_RISE_U=1.5   # the gem at least this far above the marble ...
RAISED_BLAST_NEAR_U=8.    # ... and within this horizontal distance


def raised_near(position,gem):
    \"\"\"v39: a gem the Blast (+ jump) can reach from here: raised and nearly overhead.\"\"\"
    p=np.asarray(position[:3],dtype=float);g=np.asarray(gem[:3],dtype=float)
    return float(g[2]-p[2])>=RAISED_BLAST_RISE_U and float(np.linalg.norm(g[:2]-p[:2]))<=RAISED_BLAST_NEAR_U


def detour_ratio(route,position,gem):"""),
])
patch(ml + '/nav/marble_dynamics/reliability_eval.py', [
("""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
""",
"""        _search.THROUGH_GEM=not getattr(a,'no_through_gem',False)
        _search.RAISED_BLAST_FIRST=bool(getattr(a,'raised_blast_first',False))
"""),
("""    p.add_argument('--no-through-gem',action='store_true',""",
"""    p.add_argument('--raised-blast-first',action='store_true',help='Planner v39 (under test): Blast families first below a nearby raised gem')
    p.add_argument('--no-through-gem',action='store_true',"""),
])
print('raised blast first applied')
