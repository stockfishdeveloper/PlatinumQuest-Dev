"""Authorized continuous bounded training/evaluation; STOP cooperatively ends the loop."""
import argparse
from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'logs/nav/powerup_handling_20261010/baseline_A2'
HANDOFF = ROOT / 'docs/HANDOFF_POWERUP_LOOP_2026-10-10.md'


def metrics(path):
    d = json.loads(Path(path).read_text())
    rows = d['results']
    expected = set(json.loads((ROOT / 'configs/real_maps_20261009.json').read_text())['maps'])
    if len(rows) != len(expected) or {r['map'] for r in rows} != expected:
        raise ValueError('incomplete or duplicate map evaluation')
    if not d.get('checkpoints_unchanged') or any(r['status'] != 'ok' or r['n'] < 2 for r in rows):
        raise ValueError('invalid evaluation or modified weights')
    return dict(points=statistics.mean(r['mean']['points'] for r in rows),
                falls=statistics.mean(r['mean']['falls'] for r in rows),
                kotm=next(r['mean']['points'] for r in rows if r['map'] == 'KingOfTheMarble_Hunt'))


def accept(candidate, best, reference):
    # Practical screening thresholds, never a claim of statistical significance.
    return (candidate['points'] >= 1.02 * best['points'] and
            candidate['kotm'] >= .95 * reference['kotm'] and
            candidate['falls'] <= 1.2 * reference['falls'])


def practice_choice(candidates, screen, holdout):
    """Choose judgment practice only with several exploratory use/save cases."""
    by_id = defaultdict(dict)
    for r in screen['rows']:
        if r['status'] == 'ok':
            by_id[r['id']][r['branch']] = r
    buckets = defaultdict(lambda: {'use': [], 'save': []})
    for s in candidates:
        pair = by_id[s['id']]
        if not {'handling','no_use'} <= pair.keys():
            continue
        used, saved = pair['handling'], pair['no_use']
        delta = used['points'] + used.get('reset_points', 0) - saved['points']
        bucket = buckets[s['mission'], int(s['raw'][38])]
        if delta >= 2 and used['points'] >= 2 and used['falls'] <= saved['falls']:
            bucket['use'].append(s)
        elif delta <= 0 and saved['points'] >= 2 and saved['falls'] <= 1:
            bucket['save'].append(s)
    choices = []
    for (mission, resource), bucket in buckets.items():
        use, save = bucket['use'], bucket['save']
        held = [s for s in holdout if s['mission'] == mission and int(s['raw'][38]) == resource]
        if (len(use) >= 3 and len(save) >= 3 and
            len({s['split_group'] for s in use}) >= 3 and
            len({s['split_group'] for s in held}) >= 2):
            n = min(len(use), len(save), 6)
            choices.append(dict(mission=mission, resource=resource, dev=use[:n]+save[:n], holdout=held))
    return max(choices, key=lambda x: len(x['dev'])) if choices else None


def pid_alive(pid):
    # os.kill(pid, 0) is unsafe on Windows: use a process handle and a zero-time wait.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


class Loop:
    def __init__(self, directory, resume=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stopfile = self.directory / 'STOP'
        previous = json.loads(Path(resume).read_text()) if resume else {}
        self.state = {k: previous[k] for k in ('iteration','best','best_checkpoint','source','reference','pool','best_rounds','reference_rounds') if k in previous}
        self.state.update(pid=os.getpid(), phase='starting')
        self.state.setdefault('iteration', 0)
        self.active_stop = None
        self.write_state()

    def write_state(self):
        (self.directory / 'state.json').write_text(json.dumps(self.state, indent=2))

    def log(self, message):
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{stamp}] {message}'
        print(line, flush=True)
        with (self.directory / 'events.log').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
        with HANDOFF.open('a', encoding='utf-8') as f:
            f.write(f'\n- Autonomous loop {line}\n')

    def check_stop(self):
        if self.stopfile.exists():
            if self.active_stop:
                self.active_stop.touch()
            raise InterruptedError('operator STOP requested')

    def run(self, command, stem, timeout, phase):
        from nav.phase_r import hidden_process
        self.check_stop()
        self.state.update(phase=phase, command=command)
        self.log(f'{phase} started; output {stem}.stdout/stderr.txt')
        started = time.monotonic()
        with Path(str(stem)+'.stdout.txt').open('w') as out, Path(str(stem)+'.stderr.txt').open('w') as err:
            proc = hidden_process(command, cwd=ROOT, stdout=out, stderr=err)
            self.state['child_pid'] = proc.pid; self.write_state()
            try:
                while proc.poll() is None:
                    self.check_stop()
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f'{phase} exceeded bounded timeout')
                    time.sleep(2)
                if proc.returncode:
                    raise RuntimeError(f'{phase} exited {proc.returncode}; inspect {stem}.stderr.txt')
            except BaseException:
                if self.active_stop:
                    self.active_stop.touch()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                if proc.poll() is None:
                    # Only this phase's descendant tree, never global game/trainer names.
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'], capture_output=True)
                raise
        self.state.pop('child_pid', None); self.write_state()

    def prepare_pool(self, run_dir):
        pool = run_dir / 'review/next_curriculum'
        args = [sys.executable,'-u','-m','nav.prepare_powerup_curriculum',
                '--dev',str(BASE/'escape_practice/starts_dev.json'),
                '--holdout',str(BASE/'escape_practice/starts_eval.json'),
                '--recordings',str(run_dir/'held_starts'),'--out',str(pool)]
        self.run(args, run_dir/'curriculum_prepare', 180, 'fresh curriculum preparation')
        self.run([sys.executable,'-u','-m','nav.powerup_practice_eval',
                  '--ckpt',self.state['best_checkpoint'],'--starts',str(pool/'dev_candidates.json'),
                  '--n-per-map','0','--branches','no_use,handling','--resources','1,2,5',
                  '--port','9234','--out',str(pool/'dev_screen.json')],
                 run_dir/'curriculum_screen', 900, 'development opportunity screen')
        return pool

    def confirm(self, checkpoint, name):
        output = self.directory / name
        self.run([sys.executable,'-u','-m','nav.phase_r','--baseline','--rounds','4','--instances','8',
                  '--profiles','assisted','--checkpoint-file',str(checkpoint),'--output',str(output)],
                 self.directory/name,1800,'four-round confirmation '+name)
        result = metrics(output/'summary.json')
        self.log(f'Four-round confirmation {name}: {result}; forty valid rounds required by runner. Reused maps remain validation.')
        return result

    def review_ordinary(self, run_tag):
        run_dir = ROOT/'logs/nav'/run_tag
        self.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'review_powerup_pilot.ps1'),
                  '-RunTag',run_tag,'-LauncherPID','0','-BaselineDir',str(BASE),
                  '-HoldoutStarts',str(BASE/'sj_easy_eval.json'),'-BaselinePractice',
                  str(ROOT/'logs/nav/powerup_handling_20261010/sj_holdout_after_oob_fix_A2.json')],
                 run_dir/'review_driver',1200,'adopted ordinary block review')

    def main(self, wait_pid, adopt_run=None, adopt_pid=None, confirm_reference=False, stop_after_evaluations=False):
        self.state.update(phase='waiting for current control follow-up',stop_after_evaluations=stop_after_evaluations)
        self.write_state()
        started = time.monotonic()
        while pid_alive(wait_pid):
            self.check_stop()
            if time.monotonic()-started > 3300:
                raise TimeoutError('current control follow-up exceeded55 minutes')
            time.sleep(2)
        control = ROOT/'logs/nav/real_maps_20261010_powerup_control1'
        reference = metrics(BASE/'summary.json')
        initial = metrics(control/'review/full_maps/summary.json')
        if not (control/'review/next_curriculum/dev_screen_A2.json').exists():
            raise RuntimeError('initial follow-up failed; inspect before running more training')
        base_checkpoint = ROOT/'models/nav/real_maps_20261010_A2/nav_latest.pth'
        chosen = ROOT/'models/nav/real_maps_20261010_powerup_control1/review_checkpoint.pth'
        if 'best' not in self.state:
            best = initial if accept(initial, reference, reference) else reference
            if best == reference:
                chosen = base_checkpoint
            self.state.update(best=best, best_checkpoint=str(chosen), source=str(chosen), reference=reference)
        if adopt_run:
            if not adopt_pid or not adopt_run.replace('_','').isalnum():
                raise ValueError('adoption needs a specific valid run and launcher PID')
            run_dir = ROOT/'logs/nav'/adopt_run
            self.active_stop = run_dir/'STOP'
            self.state.update(phase='adopted training still running',run=adopt_run,child_pid=adopt_pid)
            self.write_state(); self.log(f'Adopted active ordinary run {adopt_run}, launcher{adopt_pid}; training uninterrupted.')
            deadline = time.monotonic()+3000
            while pid_alive(adopt_pid):
                self.check_stop()
                if time.monotonic()>deadline:
                    self.active_stop.touch()
                    raise TimeoutError('adopted training exceeded remaining bound')
                time.sleep(2)
            self.check_stop(); self.active_stop = None
            self.review_ordinary(adopt_run)
        if confirm_reference:
            confirmed_base = self.confirm(base_checkpoint,'reference_A2_four_rounds')
            confirmed_best = self.confirm(self.state['best_checkpoint'],'working_best_four_rounds')
            if accept(confirmed_best,confirmed_base,confirmed_base):
                self.state.update(best=confirmed_best,source=self.state['best_checkpoint'])
            else:
                self.state.update(best=confirmed_base,best_checkpoint=str(base_checkpoint),source=str(base_checkpoint))
            self.state.update(reference=confirmed_base,best_rounds=4,reference_rounds=4)
            self.log(f'Recalibrated continuation anchor: {self.state["best"]}, {self.state["best_checkpoint"]}; historic two-round40.4 is no longer the automatic reset target.')
            if adopt_run:
                result = metrics(ROOT/'logs/nav'/adopt_run/'review/full_maps/summary.json')
                retained = result['points'] >= .9*self.state['best']['points'] and result['kotm'] >= .9*confirmed_base['kotm'] and result['falls'] <= 1.4*confirmed_base['falls']
                if retained:
                    self.state['source'] = str(ROOT/'models/nav'/adopt_run/'review_checkpoint.pth')
                    self.log(f'Continue cumulative adopted candidate {result}; working best remains separate and confirmed.')
            self.write_state()
        if stop_after_evaluations:
            self.stopfile.touch()
            self.state.pop('child_pid', None)
            self.state.update(phase='stopped after requested evaluations',stop_reason='operator requested evaluation completion then stop')
            self.write_state()
            self.log('Requested evaluations complete; loop stopped before any next training block. All checkpoints and evaluation artifacts retained.')
            return
        reference = self.state.get('reference',reference)
        self.log(f'Working continuation source selected: {self.state["source"]}; best metrics {self.state["best"]}. A2 remains protected; screening is provisional.')
        pool = Path(self.state.get('pool', control/'review/next_curriculum'))
        stale = 0
        while True:
            self.check_stop()
            self.state['iteration'] += 1
            i = self.state['iteration']
            run_tag = f'real_maps_20261010_auto_{i:03d}'
            run_dir = ROOT/'logs/nav'/run_tag
            run_dir.mkdir(exist_ok=False)
            candidates = json.loads((pool/'dev_candidates.json').read_text())
            holdout = json.loads((pool/'holdout_candidates.json').read_text())
            screen_file = pool/('dev_screen.json' if (pool/'dev_screen.json').exists() else 'dev_screen_A2.json')
            screen = json.loads(screen_file.read_text())
            if screen['summary']['failed']:
                self.log(f'{len(screen["summary"]["failed"])} failed development windows retained; excluded by choice logic.')
            plan = practice_choice(candidates, screen, holdout) if i % 2 == 0 else None
            self.state.update(run=run_tag, practice=bool(plan))
            steps = 70 if stale >= 3 else 35
            import torch
            ck = torch.load(self.state['source'], map_location='cpu', weights_only=False)
            warm = int(ck.get('critic_warmup_remaining',0))
            if ck.get('reward_configuration',{}).get('practice_reward') != 'shared_discounted_potential_bootstrapped_v2':
                warm = 20
            target = ck['update'] + warm + steps
            self.log(f'Block{i}: {"judgment practice" if plan else "ordinary real maps"}, seed{100+i}, {steps} actor updates; source{ck["update"]}, target{target}. No activation bonus or exploration-floor increase.')
            holdout_file = BASE/'sj_easy_eval.json'
            practice_reference = ROOT/'logs/nav/powerup_handling_20261010/sj_holdout_after_oob_fix_A2.json'
            resources = '1,0'
            extra = []
            if plan:
                dev_file, holdout_file = run_dir/'practice_dev.json', run_dir/'practice_holdout.json'
                dev_file.write_text(json.dumps(plan['dev'],indent=1))
                holdout_file.write_text(json.dumps(plan['holdout'],indent=1))
                resources = str(plan['resource'])
                practice_reference = run_dir/'practice_reference.json'
                self.run([sys.executable,'-u','-m','nav.powerup_practice_eval','--ckpt',self.state['source'],
                          '--starts',str(holdout_file),'--n-per-map','0','--resources',resources,'--port','9232',
                          '--out',str(practice_reference)],run_dir/'practice_reference',900,'locked practice reference')
                if json.loads(practice_reference.read_text())['summary']['failed']:
                    raise RuntimeError('practice reference fixture failure; do not train this mixture')
                extra = ['-DrillPlan','0:4','-DrillStarts',str(dev_file),'-PracticeMode','judgment','-PracticeMission',plan['mission']]
                self.log(f'Judgment pool{len(plan["dev"])} balanced exploratory use/save states on {plan["mission"]}; heldout source rounds kept disjoint. Benefits are development estimates, not necessities.')
            else:
                extra = ['-DrillPlan','none']
                self.log('No qualified judgment pilot selected: require>=3 useful cases from3 development rounds,>=3 save cases and>=2 heldout rounds; continue ordinary learning/collection.')
            self.active_stop = run_dir/'STOP'
            self.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'prepare_real_maps.ps1'),
                      '-Checkpoint','A2','-SourceCheckpoint',self.state['source'],'-StartTraining','-RunTag',run_tag,
                      '-ShapingMode','discounted','-StopUpdate',str(target),'-MaxWallMinutes','45','-Seed',str(100+i)]+extra,
                     run_dir/'launcher',3000,'bounded training')
            self.active_stop = None
            # The launcher is already gone; PID0 cannot refer to an unrelated process.
            self.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'review_powerup_pilot.ps1'),
                      '-RunTag',run_tag,'-LauncherPID','0','-BaselineDir',str(BASE),
                      '-HoldoutStarts',str(holdout_file),'-BaselinePractice',str(practice_reference),
                      '-PracticeResources',resources],run_dir/'review_driver',1200,'practice and full-map review')
            result = metrics(run_dir/'review/full_maps/summary.json')
            review = json.loads((run_dir/'review/comparison.json').read_text())
            self.log(f'Review saved update{review["update"]}, requested target{target}; wall cap may stop an experiment early.')
            candidate = str(ROOT/'models/nav'/run_tag/'review_checkpoint.pth')
            use_result = review['practice']['learned_use_sample']
            practice_retained = (not plan or (use_result['candidate_points'] >= use_result['baseline_points']
                                  and use_result['candidate_falls'] <= use_result['baseline_falls']))
            if accept(result,self.state['best'],reference) and practice_retained:
                result = self.confirm(candidate,f'candidate_{i:03d}_four_rounds')
            if accept(result,self.state['best'],reference) and practice_retained:
                self.state.update(best=result,best_checkpoint=candidate,source=candidate)
                self.state['best_rounds'] = 4
                stale = 0
                self.log(f'Provisional working best improved: {result}; frozen checkpoint {candidate}. Reused validation maps do not establish unseen-map generalization.')
            else:
                stale += 1
                retained = result['points'] >= .9*self.state['best']['points'] and result['kotm'] >= .9*reference['kotm'] and result['falls'] <= 1.4*reference['falls']
                self.state['source'] = candidate if retained and not plan else self.state['best_checkpoint']
                self.log(f'Candidate metrics{result}; practice sampling retained={practice_retained}; best retained{self.state["best"]}. Next source {self.state["source"]}; no promotion. Modest ordinary fluctuations may continue; practice regressions return to best.')
            self.write_state()
            pool = self.prepare_pool(run_dir)
            self.state['pool'] = str(pool); self.write_state()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--wait-pid', type=int, required=True)
    p.add_argument('--resume-state', help='Preserve working best/source and advance past an invalidated run')
    p.add_argument('--adopt-run', help='Finish review of an already running ordinary block without restarting it')
    p.add_argument('--adopt-pid', type=int)
    p.add_argument('--confirm-reference', action='store_true')
    p.add_argument('--stop-after-evaluations', action='store_true', help='Drain the adopted review/confirmation sequence; launch no next training block')
    args = p.parse_args()
    loop = Loop(args.directory, args.resume_state)
    try:
        loop.main(args.wait_pid,args.adopt_run,args.adopt_pid,args.confirm_reference,args.stop_after_evaluations)
    except InterruptedError:
        loop.state['phase'] = 'stopped by operator'; loop.write_state()
        loop.log('Operator stop honored; checkpoints and artifacts retained.')
    except Exception as e:
        loop.state.update(phase='error; no further block started',error=str(e)); loop.write_state()
        loop.log(f'Loop error retained: {e}. No automatic retry of an invalid experiment.')
        raise


if __name__ == '__main__':
    main()
