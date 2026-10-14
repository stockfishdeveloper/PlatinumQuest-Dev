"""General collision-surface recovery proposals before committing to a gem.

Actual full-map support points are destinations, not gems/rewards. Every proposal
must end supported and retain a verified48-reply ordinary stopping backup. A
selected forecast still needs independent native execution; this is action search.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from nav.collision.mesh import atomic_json
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.landing import landing_points
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.search import SequenceProposal,assess
from nav.marble_dynamics.viability import braking_viability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--map',required=True);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-frame-ms',type=int,choices=[16,64],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);x=json.loads(a.source.read_text())
    engine=GeneralPhysics(CachedCompiledSweepMap(ML/'collision_maps'/(a.map+'.npz'),a.dll));engine.bridge_physics_ms=a.physics_frame_ms
    points=landing_points(engine.mesh,x['dynamics']['native']['state'],12)
    atomic_json(a.output/'prepared.json',dict(source=str(a.source),source_sha256=sha(a.source),map=a.map,
        source_hashes={str(path):sha(path) for path in list(Path(__file__).parent.glob('*.py'))+[a.dll]},
        support_points=points,horizons=[12,20,32],jump_replies=[-1,0,3],offsets=[0.,-.7,.7],scales=[1.,.7,.4],braking_backup=48,
        training=False,scored_navigation=False,contract='No state/inventory injection. Full actual collision surfaces propose interim destinations; all original support/ambiguity checks and stopping backup retained. Stop at first checked recovery; no native success claim.'))
    # Direct controls deliberately allow a destination away from the gem. The
    # walking tour's progress filter must not veto a physically checked retreat.
    results=[];selected=None;start=time.perf_counter()
    for point in points:
        for horizon in [12,20,32]:
            for jump in [-1,0,3]:
                for offset in [0.,-.7,.7]:
                    for scale in [1.,.7,.4]:
                        proposal=SequenceProposal(None,point['position'],horizon,jump=jump,offset=offset,scale=scale,direct=True)
                        forecast=predict_replies(engine,x['dynamics'],proposal,stop_on_ambiguity=True)
                        info=assess(engine,None,x['dynamics'],point['position'],forecast,'geometry_recovery',routed=False)
                        info['possible_pickup_interval']=None
                        info.update(support_point=point,horizon=horizon,jump=jump,offset=offset,scale=scale)
                        if info['rejected'] is None:
                            check,continued=braking_viability(engine,x['dynamics'],forecast,point['position'],48)
                            info['stopping_check']=check
                            if check['valid']:
                                forecast=continued;selected=info
                            else:info['rejected']='stopping:'+check['rejected']
                        results.append(info)
                        if selected:break
                    if selected:break
                if selected:break
            if selected:break
        atomic_json(a.output/'progress.json',dict(candidates=results,selected=selected,training=False))
        if selected:break
    report=dict(selected=selected,candidates=results,rejections=dict(Counter(str(r['rejected']) for r in results)),
        elapsed_s=time.perf_counter()-start,native_success=False,training=False)
    atomic_json(a.output/'report.json',report)
    if selected:atomic_json(a.output/'selected_forecast.json',forecast)
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'}),flush=True)


if __name__=='__main__':main()
