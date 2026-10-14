"""Reproduce an accepted ledge approach followed by legal braking in native play.

Archive a clearly labeled prefix slice of the existing world before replaying.
The fixture tests mechanics only; it is not a gem-collection comparison.
"""
import argparse
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.forecast_replay_probe import execute
from nav.marble_dynamics.record import ML,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=9054)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    original=ML/'dynamics_reports/reliability_landing_validation_20261012/CubeIsle_Hunt/151113/physics_plan'
    forecast_path=ML/'dynamics_reports/stopping_viability_20261012/original_blast_braking_forecast.json'
    records=json.loads((original/'decisions.json').read_text());initial=records[503]
    source=a.output/'archived_world'/'physics_plan';source.mkdir(parents=True)
    atomic_json(source.parent/'fixture.json',json.loads((original.parent/'fixture.json').read_text()))
    atomic_json(source/'report.json',dict(map='CubeIsle_Hunt',seed=151113))
    atomic_json(source/'decisions.json',records[:503])
    atomic_json(source/'terminal_search.json',dict(dynamics=initial['dynamics'],target=initial['gem']))
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    atomic_json(a.output/'prepared.json',dict(original_decisions=str(original/'decisions.json'),
        original_decisions_sha256=sha(original/'decisions.json'),forecast_sha256=sha(forecast_path),
        source_hashes={str(path):sha(path) for path in sources},prefix_decisions=503,
        continuation_replies=52,training=False,scored_navigation=False,
        declaration='Exact archived prefix, original40-reply approach plus12 braking replies; original prediction finds no supported stopping continuation.'))
    for path in sources:
        saved=a.output/'source_snapshot'/path.resolve().relative_to(ML.parent.resolve())
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(path.read_bytes())
    actions=json.loads(forecast_path.read_text())['proposed'][:52]
    execute(source,actions,a.output/'cube_ledge_braking',a.port)


if __name__=='__main__':main()
