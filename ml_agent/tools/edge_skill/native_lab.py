"""Owned native lab collection and exact engine collision export."""
import argparse
import copy
import gzip
import time

from nav.collision.collect import command_file
from nav.collision.mesh import convert_snapshot
from nav.collision.mission import annotations_and_coverage
from nav.marble_dynamics.record import game, command
from nav.protocol import NOOP_ACTION, format_teleport

from .common import ART, NIGHT, ROOT, RUNTIME, ASSET, load_json, sha, write_json
from .maps import transform


def prepare(env):
    command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
    env.control('FIXEDSTEP 16 64')
    env.obs_ms = 64
    for _ in range(8):
        env.step(NOOP_ACTION, repeat=1)


def place(env, position, velocity, spin):
    env.control('RESPAWN')
    env.step(NOOP_ACTION, repeat=16)
    env.control('GIVEPOW none')
    env.step(NOOP_ACTION, repeat=4)
    env.control(format_teleport(*position, *velocity, spin=spin))
    env.step(NOOP_ACTION, repeat=1)
    d = copy.deepcopy(env.msg.dynamics)
    if d is None or d['native']['state']['mode'] != 0 or d['native']['state']['oob']:
        raise RuntimeError('No playable initial snapshot')
    return d


def export_map(entry, port):
    name = entry['name']
    artifact=NIGHT if entry.get('generation','').startswith('diverse') else ART
    folder = artifact/'geometry'
    folder.mkdir(exist_ok=True,parents=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    raw = RUNTIME/(name+'.engine.jsonl')
    with game(name, port, obs_ms=16, seed=116031, process_cap=2) as (env, proc):
        prepare(env)
        command_file(env, 'GEOMETRYEXPORT '+raw.as_posix(), raw)
        out = folder/(name+'.npz')
        metadata = dict(map=name, mission=entry['mission'], mission_sha256=entry['sha256'],
                        source_inventory=entry.get('source_inventory') or [dict(path=str(ASSET.relative_to(ROOT)), sha256=sha(ASSET))],
                        game_info=env.info,
                        probe_executable_sha256=sha(ROOT/'Marble Blast Platinum/marbleblast_geometry_probe.exe'))
        info = convert_snapshot(raw, out, metadata)
        info.update(annotations_and_coverage(ROOT/entry['mission'], info['objects']))
        snap = folder/(name+'.engine.jsonl.gz')
        snap.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
        info['raw_snapshot_file'] = snap.name
        write_json(out.with_suffix('.json'), info)
        # Native initial state templates; all later synthetic variants are explicitly synthetic.
        c = entry['cells'][0]
        d = place(env, transform(c, [-4, 0, .18975]), [0., 0., 0.], [0., 0., 0.])
        write_json(artifact/(name+'.initial.json'), dict(dynamics=d, observation=env.msg.obs.tolist()))
        print(name, 'exported', info['triangles'], 'triangles; radius', d['native']['state']['radius'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9580)
    ap.add_argument('--map', default=None)
    args = ap.parse_args()
    manifest = load_json(ART/'maps.json')
    for m in manifest['maps']:
        if args.map and m['name'] != args.map:
            continue
        export_map(m, args.port)


if __name__ == '__main__':
    main()
