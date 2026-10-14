"""Audited native mechanics coverage; no policy, reward or learned-model fitting.

Ordinary control families plus matched use/no-use power trials from supported
and falling starts. All inventory grants/teleports are setup-only fixtures.
"""
from nav.source_archive import snapshot_sources
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from nav.collision.mesh import CollisionMap,atomic_json
from nav.marble_dynamics.evaluate import load_rows
from nav.marble_dynamics.record import ML,RUNTIME,DEV,TEST,POWERUPS,Sampler,game,run_trial,archive,sha

ORDINARY=('roll','switch','brake','jump','drop_spin','wall_impact','edge_timing','ceiling_impact')


def specifications(sampler,group):
    for family in ORDINARY:
        yield dict(family=family,variant='ordinary',pair_group=None,
            spec=sampler.sample(family,group))
    for power in range(5):
        for start in ('supported','falling'):
            base=sampler.sample('powerup' if start=='supported' else 'drop_spin',group*5+power)
            if base is not None:
                base.update(family='powerup',powerup=POWERUPS[power],frames=375)
            pair='group_'+str(group)+'/'+start+'/'+POWERUPS[power]
            # Held inventory stays the SAME for the control; only use is varied.
            for use in (False,True):
                spec=copy.deepcopy(base)
                if spec is not None:spec['use_frame']=12 if use else -1
                yield dict(family='powerup',variant=start,pair_group=pair,use_requested=use,spec=spec)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',nargs='+',required=True);p.add_argument('--groups',type=int,default=1)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--port',type=int,default=9053)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.groups<1:raise ValueError('Positive bounded groups required')
    a.output.mkdir(parents=True,exist_ok=False)
    sources=list(Path(__file__).parent.glob('*.py'))+[ML/'nav/collision/mesh.py',
        ML.parent/'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
        ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe']
    fingerprints={str(path):sha(path) for path in sources}
    atomic_json(a.output/'prepared.json',dict(maps=a.maps,groups=a.groups,seed=a.seed,
        source_hashes=fingerprints,training=False,scored_navigation=False,
        contract='Native16ms audited mechanics. Eight ordinary control families and five supported power types from supported/falling starts, matched held-inventory use/no-use comparisons. Setup-only grants/respawns/teleports; no intervention during recorded continuation. Every failure/fall retained. Actual activated body/support state reported, not inferred from requested setup.'))
    snapshot_sources(a.output,[path for path in sources if path.suffix.lower() in ('.py','.cs')])
    summary=[]
    for mi,name in enumerate(a.maps):
        if any(sha(Path(path))!=digest for path,digest in fingerprints.items()):
            raise ValueError('Source changed during collection; use a new directory')
        mesh=CollisionMap(ML/'collision_maps'/(name+'.npz'));sampler=Sampler(mesh,a.seed+mi)
        folder=a.output/name;folder.mkdir()
        manifest=dict(schema='marble_trajectories_v1',map=name,seed=a.seed+mi,
            geometry_sha256=mesh.info['mesh_sha256'],split='dev' if name in DEV else 'test' if name in TEST else 'train',
            trials=[],replay_checks=[],prepared=str(a.output/'prepared.json'),collection_complete=False,
            probe_executable_sha256=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe'))
        with game(name,a.port,seed=a.seed+mi) as (env,proc):
            manifest['owned_game_pid']=proc.pid
            for group in range(a.groups):
                for request in specifications(sampler,group):
                    ident=len(manifest['trials']);spec=request['spec']
                    row=dict(id=ident,**request)
                    if spec is None:row.update(valid=False,reason='no_clear_sampled_start')
                    else:
                        raw=RUNTIME/(name+'.rich_mechanics_%04d.jsonl'%ident)
                        row.update(run_trial(env,spec,raw))
                        if row['valid']:
                            dest=folder/('trial_%04d.jsonl.gz'%ident);archive(raw,dest)
                            rows=load_rows(dest)
                            if any('control_events' not in r for r in rows):
                                raise ValueError('Complete native camera audit required')
                            row.update(file=dest.name,raw_sha256=sha(raw),actual_initial_state=rows[0]['before'])
                            row['actual_activation_samples']=[dict(index=i,position=r['before']['position'],
                                velocity=r['before']['velocity'],spin=r['before']['spin'],held=r['before']['held'],
                                native_contact_steps=r['telemetry']['contact_steps'],native_support_steps=r['telemetry']['support_steps'])
                                for i,r in enumerate(rows) if r['move']['use'] and r['before']['held'] and not r['after']['held']]
                    manifest['trials'].append(row);atomic_json(folder/'manifest.json',manifest)
                    print(json.dumps(dict(map=name,trial=ident,variant=row['variant'],family=row['family'],
                        valid=row['valid'],samples=row.get('samples'),activations=len(row.get('actual_activation_samples',[])))),flush=True)
        pairs={}
        for trial in manifest['trials']:
            if trial['valid'] and trial['pair_group']:pairs.setdefault(trial['pair_group'],[]).append(trial)
        checks=[]
        for key,trials in pairs.items():
            first=trials[0]['actual_initial_state']
            checks.append(dict(group=key,trials=len(trials),
                maximum_initial_position_difference_u=max(float(np.linalg.norm(np.asarray(t['actual_initial_state']['position'])-first['position'])) for t in trials),
                maximum_initial_velocity_difference_u_per_s=max(float(np.linalg.norm(np.asarray(t['actual_initial_state']['velocity'])-first['velocity'])) for t in trials),
                maximum_initial_spin_difference_rad_per_s=max(float(np.linalg.norm(np.asarray(t['actual_initial_state']['spin'])-first['spin'])) for t in trials),
                native_held_types=[t['actual_initial_state']['held'] for t in trials],
                actual_activations=[len(t['actual_activation_samples']) for t in trials]))
        manifest.update(collection_complete=True,pair_checks=checks);atomic_json(folder/'manifest.json',manifest)
        summary.append(dict(map=name,trials=len(manifest['trials']),valid=sum(t['valid'] for t in manifest['trials']),
            samples=sum(t.get('samples',0) for t in manifest['trials']),
            falls=sum(t.get('fell',False) for t in manifest['trials']),
            activations=sum(len(t.get('actual_activation_samples',[])) for t in manifest['trials'])))
        atomic_json(a.output/'collection.json',dict(maps=summary,training=False,scored_navigation=False,
            collection_complete=len(summary)==len(a.maps)))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
