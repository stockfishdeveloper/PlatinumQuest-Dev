"""Replay saved fixture specs with native per-substep force/contact diagnostics."""
import argparse
import copy
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.marble_dynamics.record import ML,RUNTIME,game,command,run_trial,archive,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ML/'datasets/marble_dynamics/pilot_20261011')
    p.add_argument('--map',default='Concentric_Hunt')
    p.add_argument('--trials',nargs='+',type=int,default=[0,18,35])
    p.add_argument('--output',type=Path,default=ML/'datasets/marble_dynamics/contact_diagnostics_20261012')
    p.add_argument('--port',type=int,default=9053)
    args=p.parse_args();source=json.loads((args.data/args.map/'manifest.json').read_text())
    result=copy.deepcopy(source);result['trials']=[];result['replay_checks']=[]
    result['diagnostic_source']=str(args.data);result['native_substep_detail']=True
    result['probe_executable_sha256']=sha(ML.parent/'Marble Blast Platinum/marbleblast_geometry_probe.exe')
    result['native_instrumentation_sha256']=sha(Path(__file__).with_name('engine_dynamics.inc'))
    dest=args.output/args.map;dest.mkdir(parents=True,exist_ok=True)
    with game(args.map,args.port) as (env,proc):
        command(env,'DYNAMICSDETAIL 1','dynamics_detail')
        result['owned_game_pid']=proc.pid
        for trial in source['trials']:
            if trial['id'] not in args.trials:continue
            path=RUNTIME/(args.map+'.diagnostic_%04d.jsonl'%trial['id'])
            captured=run_trial(env,trial['spec'],path)
            row=dict(id=trial['id'],family=trial['family'],spec=trial['spec'],**captured)
            if captured['valid']:
                output=dest/('trial_%04d.jsonl.gz'%trial['id']);archive(path,output)
                row['file']=output.name;row['raw_sha256']=sha(path)
            result['trials'].append(row);atomic_json(dest/'manifest.json',result)
            print(json.dumps(dict(trial=trial['id'],valid=row['valid'],samples=row.get('samples'))),flush=True)
    atomic_json(dest/'manifest.json',result)


if __name__=='__main__':main()
