"""Execute an explicitly archived proposal from an exact earlier native prefix."""
import argparse
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.forecast_replay_probe import execute
from nav.marble_dynamics.record import ML,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--decision',type=int,required=True)
    p.add_argument('--forecast',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=9055)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=json.loads((a.source/'decisions.json').read_text());initial=records[a.decision]
    report=json.loads((a.source/'report.json').read_text())
    source=a.output/'archived_world'/'physics_plan';source.mkdir(parents=True)
    atomic_json(source.parent/'fixture.json',json.loads((a.source.parent/'fixture.json').read_text()))
    atomic_json(source/'report.json',dict(map=report['map'],seed=report['seed']))
    atomic_json(source/'decisions.json',records[:a.decision])
    atomic_json(source/'terminal_search.json',dict(dynamics=initial['dynamics'],target=initial['gem']))
    actions=json.loads(a.forecast.read_text())['proposed']
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    atomic_json(a.output/'prepared.json',dict(original_decisions=str(a.source/'decisions.json'),
        original_decisions_sha256=sha(a.source/'decisions.json'),forecast_path=str(a.forecast),
        forecast_sha256=sha(a.forecast),source_hashes={str(path):sha(path) for path in sources},
        prefix_decisions=a.decision,continuation_replies=len(actions),training=False,scored_navigation=False,
        declaration='Exact earlier archived prefix followed by fully saved ordinary controls; no resets/grants/rescue during execution.'))
    for path in sources:
        saved=a.output/'source_snapshot'/path.resolve().relative_to(ML.parent.resolve())
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(path.read_bytes())
    execute(source,actions,a.output/'continuation',a.port)


if __name__=='__main__':main()
