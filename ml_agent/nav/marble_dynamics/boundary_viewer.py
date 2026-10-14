"""Inspect independently replayed episode endpoints, including native failures."""
import argparse
import html
import json
from pathlib import Path

from plotly.offline import get_plotlyjs
from nav.collision.mesh import CollisionMap, atomic_json
from nav.collision.viewer import build_map, write_atomic
from nav.marble_dynamics.evaluate import errors, load_rows
from nav.marble_dynamics.record import ML, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    folder = a.output / 'trajectories'
    folder.mkdir(exist_ok=True)
    traces, provenance, reports = {}, [], []
    for root in a.data:
        report = json.loads((root / 'report.json').read_text())
        initial = json.loads((root / 'initial.json').read_text())
        logs = json.loads((root / 'decisions.json').read_text())
        forecast = json.loads((root / 'forecast.json').read_text())
        prepared = json.loads((root / 'prepared.json').read_text())
        first = initial['dynamics']['native']['frame_clock_ms']
        rows = [r for r in load_rows(root / 'native.jsonl.gz')
                if first <= r['sim_ms'] < first + 64 * len(logs)]
        if len(rows) != 4 * len(logs):
            raise ValueError('Incomplete native endpoint recording: ' + str(root))
        actual = [initial['dynamics']['native']['state']] + [r['after'] for r in rows]
        predicted = forecast['native_states'][:len(actual)]
        if len(predicted) != len(actual):
            raise ValueError('Incomplete proposed-input forecast: ' + str(root))
        measures = [errors(p, n) for p, n in zip(predicted, actual)]
        prefix = json.loads((Path(prepared['source']) / 'decisions.json').read_text())
        goal = prefix[-1]['gem'][:3]
        score, meter = [0.] * len(actual), [None] * len(actual)
        gained = 0.
        for i, log in enumerate(logs):
            meter[4*i] = log['dynamics']['blast']['value']
            score[4*i:4*i+4] = [gained] * 4
            gained += log['score_delta']
        score[-1] = gained
        values = lambda states, key: [s[key] for s in states]
        name = report['map']
        trace = dict(id=report['seed'], source_dataset=root.name,
            family='whole_native16ms_endpoint', powerup='No new use; feedback braking',
            split='archived_prefix_development', time_s=[i*.016 for i in range(len(actual))],
            actual_position=values(actual, 'position'), predicted_position=values(predicted, 'position'),
            actual_velocity=values(actual, 'velocity'), predicted_velocity=values(predicted, 'velocity'),
            actual_spin=values(actual, 'spin'), predicted_spin=values(predicted, 'spin'),
            position_error=[e[0] for e in measures], max_position_error_u=max(e[0] for e in measures),
            final_errors=measures[-1], prediction_available=[True]*len(actual),
            truncated_external_samples=0, rollout_failure=forecast['failure'],
            native_use_events=[], native_powerup_activations=[],
            ambiguous_contact_intervals=[4*i for i in forecast['ambiguous_intervals'] if 4*i < len(rows)],
            held_powerup=[str(s['held']) if s['held'] else 'None' for s in actual],
            score=score, pickup_decisions=[], goal_position=[goal]*len(actual),
            goal_kind=['retained gem context; braking continuation']*len(actual), blast_meter=meter,
            control_basis='Exact archived control prefix replayed first. Shown continuation is one open-loop forecast from its independently observed endpoint and proposed feedback-brake replies. No future native input or state enters prediction. Only the actually played interval is drawn; unused forecast tail is omitted. Score is gained during this continuation. A native fall is shown, not suppressed or labeled safe.',
            outcomes=report)
        traces.setdefault(name, []).append(trace)
        reports.append(report)
        provenance.append(dict(path=str(root), report_sha256=sha(root/'report.json'),
            native_sha256=sha(root/'native.jsonl.gz'), forecast_sha256=sha(root/'forecast.json')))
    names = list(traces)
    for name in names:
        mesh = CollisionMap(ML/'collision_maps'/(name+'.npz'))
        atomic_json(folder/(name+'.trajectories.json'), dict(map=name,
            geometry_sha256=mesh.info['mesh_sha256'], trajectories=traces[name]))
    write_atomic(a.output/'plotly.min.js', get_plotlyjs())
    maps = [build_map(ML/'collision_maps'/(name+'.npz'), a.output, names, [folder]) for name in names]
    links = ''.join('<li><a href="'+name+'.html">'+html.escape(name)+'</a></li>' for name in names)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in
        ('map', 'seed', 'prefix_replies', 'replies', 'actual_points', 'falls'))+'</tr>' for r in reports)
    write_atomic(a.output/'index.html', '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Native episode endpoint checks</title><style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:32px;line-height:1.6}a{color:#79c8ff}p{max-width:1000px}td,th{padding:8px;text-align:left}</style>
<h1>What happens after the episode ends?</h1>
<p>Choose Recording, Focus trajectory, and scrub Time on the full reviewed collision map. Cyan is measured16ms native motion; coral is one complete forecast from the observed endpoint and proposed64ms braking replies. Only the executed interval appears. Spin and velocity are measured and predicted; intermediate Blast charge is unknown and left blank.</p>
<p>The archived episode prefix is independently replayed before recording this continuation; its measured physical endpoint must match. Braking is a general control probe, not evidence that every possible escape fails. A fall disproves the claim that this endpoint can safely stop with this braking control. No reset, rescue, item grant, training or checkpoint promotion occurs during the continuation.</p><ul>'''+links+'''</ul>
<table><thead><tr><th>Map</th><th>Seed</th><th>Replayed prefix</th><th>Brake replies</th><th>New points</th><th>Native falls</th></tr></thead><tbody>'''+rows+'''</tbody></table>
<p><a href="../planning_progress_20261012/index.html">Planning results and limitations</a></p></html>''')
    atomic_json(a.output/'summary.json', dict(maps=maps, sources=provenance,
        training=False, scored_navigation=False))
    print(a.output/'index.html')


if __name__ == '__main__':
    main()
