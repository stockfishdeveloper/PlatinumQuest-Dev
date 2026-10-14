"""Separate review viewer for gem pilots, built from the same collision NPZs."""
import argparse
import html
from pathlib import Path

from plotly.offline import get_plotlyjs
from nav.collision.mesh import atomic_json
from nav.collision.viewer import build_map,write_atomic
from nav.marble_dynamics.record import ML


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectories',type=Path,nargs='+',required=True)
    p.add_argument('--maps',nargs='+',default=['Concentric_Hunt','KingOfTheMarble_Hunt','Sprawl_Hunt'])
    p.add_argument('--output',type=Path,default=ML/'collision_maps/viewer/gem_demo')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    write_atomic(a.output/'plotly.min.js',get_plotlyjs())
    rows=[build_map(ML/'collision_maps'/(name+'.npz'),a.output,a.maps,a.trajectories) for name in a.maps]
    links=''.join('<li><a href="'+name+'.html">'+html.escape(name)+'</a></li>' for name in a.maps)
    write_atomic(a.output/'index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><title>Gem planning demonstration</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:40px;line-height:1.6}a{color:#79c8ff}p{max-width:950px}</style>
<h1>Gem planning demonstration</h1><p>Each viewer renders the exact collision geometry. Recording choices compare native execution with five-step (320 ms) predictions made from one synchronized initial state and the recorded bridge commands. Predictions receive no future physical state or camera observations.</p>
<p>These are short control-window replays from live gem pilots, including confirmed pickups. They do not demonstrate reliable multi-second landings or improved trained-policy performance. Committed comparisons restore the same settled spawn/world before each branch; no restore occurs during branch execution. The comparator is a simple routed controller.</p>
<p>Use Recording, Focus trajectory, the Time slider and Largest error. Prior native mechanics/uncertainty recordings remain in the <a href="../index.html">original ten-map viewer</a>.</p><ul>'''+links+'</ul></html>')
    atomic_json(a.output/'summary.json',dict(maps=rows,training_started=False))
    print('Saved '+str(a.output/'index.html'))


if __name__=='__main__':main()
