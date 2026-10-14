"""Check goal/score/resource readouts and their rendered 3D markers."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from nav.collision.mesh import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--browser',default='C:/Program Files/Google/Chrome/Application/chrome.exe');a=p.parse_args()
    names=[r['map'] for r in json.loads((a.directory/'summary.json').read_text())['maps']];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.browser,headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        try:
            page=browser.new_page(viewport=dict(width=1600,height=1000))
            for name in names:
                page.goto((a.directory/(name+'.html')).resolve().as_uri())
                page.wait_for_function("document.getElementById('geometry')._fullLayout && document.getElementById('geometry')._fullLayout.scene._scene")
                count=page.evaluate('model.trajectories.length')
                for j in range(count):
                    page.evaluate('async j=>await selectTrajectory(j)',j)
                    indices=page.evaluate("""() => Array.from(new Set([0,activeTrajectory.time_s.length-1,
                        Math.floor(activeTrajectory.time_s.length/2),...(activeTrajectory.native_powerup_activations||[]).map(e=>e.interval)]))""")
                    for i in indices:
                        result=page.evaluate("""async i=>{await scrubTrajectory(i);const c=activeTrajectory,
                            read=JSON.parse(document.getElementById('trajectory-readout').textContent),tr=gd.data[model.trajectory_traces[4]];
                            return {index:i,goal:c.goal_position?c.goal_position[i]:null,
                            rendered_goal:tr.visible?[tr.x[0],tr.y[0],tr.z[0]]:null,
                            goal_readout:read.goal,meter:c.blast_meter?c.blast_meter[i]:null,
                            meter_readout:read.blast_meter,score:c.score?c.score[i]:null,score_readout:read.game_score};}""",i)
                        if result['goal'] is not None:
                            assert result['goal']==result['rendered_goal']==result['goal_readout'],result
                        if result['meter'] is not None:assert result['meter']==result['meter_readout'],result
                        if result['score'] is not None:assert result['score']==result['score_readout'],result
                        checks.append(dict(map=name,trajectory=j,**result))
            atomic_json(a.directory/'goal_readout_checks.json',dict(cases=checks,training=False))
            print(json.dumps(dict(checked_samples=len(checks),maps=len(names),passed=True)),flush=True)
        finally:browser.close()


if __name__=='__main__':main()
