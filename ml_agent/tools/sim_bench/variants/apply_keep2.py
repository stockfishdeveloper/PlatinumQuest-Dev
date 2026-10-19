"""Candidate: continue the last checked plan instead of the A2 fallback (behind --keep-plan while under test).

Every in-game fall of v23 (32 of 32) and v26 (49 of 51) was A2-led. Typical case (Cube s2 decision 1322, reproduced by the
offline benchmark): a checked plan collects an edge gem as the marble starts down the side; the pickup invalidates the
plan; the next 2 s searches find nothing useful (the marble is unsettled: 'no_supported_continuation') and A2 drives off
the edge. Now the remainder of the plan that was running at the pickup (checked when chosen: it ends supported, with a
braking backup) is kept; while the searches find nothing useful it is revalidated from the current state (same gates) and
executed instead of the A2 fallback, until a search succeeds or it no longer passes. Likewise when any search finds
nothing useful: the remainder of the last accepted plan (still in memory) is revalidated and continued before A2.
usage: python apply_keep.py [ml_agent dir]
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


patch(ml + '/nav/marble_dynamics/reliability_eval.py', [
("""        elif events or (picked and not self.memory.landing_mode):
            self.invalidate('observed_context_or_pickup')""",
"""        elif events or (picked and not self.memory.landing_mode):
            # keep-plan: the remainder of the checked plan that was running at the pickup becomes the fallback for
            # the next decisions whose searches find nothing useful (instead of the A2 actor)
            keep=list(self.memory.remaining) if (picked and not events and getattr(self.args,'keep_plan',False)) else None
            self.invalidate('observed_context_or_pickup')
            if keep and len(keep)>=4:self.continue_plan=keep"""),
("""                hold=getattr(self.engine,'safe_hold',None)
                if (not chosen and hold is not None""",
"""                kept=(getattr(self,'continue_plan',None) or (list(self.memory.remaining) if self.memory.actions else None)) if getattr(self.args,'keep_plan',False) else None
                if not chosen and kept and len(kept)>=2 and self.resource is None and not self.memory.landing_mode:
                    # keep-plan: revalidate the kept remainder from the current state (same gates as any plan)
                    try:
                        saved_deadline=self.engine.planning_deadline;self.engine.planning_deadline=None
                        try:kroll=predict_replies(self.engine,dynamics,kept,stop_on_ambiguity=True)
                        finally:self.engine.planning_deadline=saved_deadline
                        kinfo=assess(self.engine,self.route,dynamics,np.asarray(target[:3],dtype=float),kroll,'continue_plan',None)
                        if kinfo['rejected'] is None:
                            kinfo=dict(kinfo,label='continue_plan',primary_fallback_reason=reason)
                            chosen=(kinfo,kroll);reason=None;self.counts['continue_plan']+=1
                            self.memory.accept(kroll,'continue_plan',list(target[:3]),kinfo)
                        else:self.counts['continue_plan_rejected']+=1
                    except ValueError:self.counts['continue_plan_error']+=1
                    self.continue_plan=None
                elif chosen:self.continue_plan=None
                hold=getattr(self.engine,'safe_hold',None)
                if (not chosen and hold is not None"""),
("""    p.add_argument('--no-through-gem',action='store_true',""",
"""    p.add_argument('--keep-plan',action='store_true',help='Planner (under test): after a pickup, the running plan remainder replaces the A2 fallback while searches find nothing useful')
    p.add_argument('--no-through-gem',action='store_true',"""),
])
print('keep plan applied')
