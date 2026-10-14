"""Profile a general predicted-state proposal from each archived initial state.

This measures the mechanics/route hot loop, not a full candidate search or a
performance benchmark. No future recorded states/controls, native game or model
training are used. Setup/import/geometry load is outside the profile.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats

from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.record import ML,sha
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.search import SequenceProposal
from nav.routing import RoutePlanner
from nav.terrain import TerrainGrid,TerrainMap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path)
    a=p.parse_args();summary=json.loads((a.data/'summary.json').read_text());seen=set();results=[]
    for report in summary['branches']:
        name=report['map']
        if name in seen or report.get('status')=='invalid':continue
        seen.add(name)
        source=a.data/name/str(report['seed'])/report['branch']/'decisions.json'
        first=json.loads(source.read_text())[0]
        engine=GeneralPhysics(CollisionMap(ML/'collision_maps'/(name+'.npz')))
        route=RoutePlanner(TerrainGrid(TerrainMap.resolve(name)))
        goal=first['gem'][:3];route.navigate(first['dynamics']['native']['state']['position'],goal)
        proposal=SequenceProposal(route,goal,40)
        profiler=cProfile.Profile();profiler.enable()
        rollout=predict_replies(engine,first['dynamics'],proposal)
        profiler.disable();stats=pstats.Stats(profiler)
        ordered=sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)
        hot=[dict(file=key[0],line=key[1],function=key[2],calls=value[1],
                  self_s=value[2],cumulative_s=value[3]) for key,value in ordered[:20]]
        results.append(dict(map=name,source_sha256=sha(source),replies=len(rollout['proposed']),
                            simulated_s=(len(rollout['states'])-1)*.064,failure=rollout['failure'],
                            profiled_total_s=stats.total_tt,hot_functions=hot))
        print(json.dumps(dict(map=name,profiled_total_s=stats.total_tt,hot_functions=hot[:5])))
    atomic_json(a.data/'profiling.json',dict(cases=results,profile_source_sha256=sha(Path(__file__)),
        contract='One initial synchronized state + newly proposed controls; cProfile overhead included; no full-search or real-time latency claim.',policy_training_started=False))


if __name__=='__main__':main()
