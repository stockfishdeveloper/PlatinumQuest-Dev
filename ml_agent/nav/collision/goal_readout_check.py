"""Verify rendered goals, observed scores/charge and local review links.

Separate headless browser; does not attach to the operator's browser session.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import unquote,urlparse

from playwright.sync_api import sync_playwright

from nav.collision.mesh import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--index',type=Path)
    p.add_argument('--browser',default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    a=p.parse_args();cases=[];links=[];errors=[]
    names=[r['map'] for r in json.loads((a.directory/'summary.json').read_text())['maps']]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.browser,headless=True,
            args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        try:
            page=browser.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
            for name in names:
                page.goto((a.directory/(name+'.html')).resolve().as_uri())
                page.wait_for_function("typeof model !== 'undefined' && gd._fullLayout && gd._fullLayout.scene._scene")
                count=page.evaluate('model.trajectories.length')
                for trajectory in range(count):
                    page.evaluate('async i => await selectTrajectory(String(i))',trajectory)
                    indexes=page.evaluate("""() => {
                        const c=activeTrajectory,n=c.time_s.length;
                        const indexes=new Set([0,n-1,Math.floor((n-1)/2)]);
                        (c.pickup_decisions||[]).forEach(i=>{indexes.add(i);indexes.add(Math.min(n-1,i+1));});
                        (c.goal_kind||[]).forEach((kind,i)=>{if(i && kind!==c.goal_kind[i-1])indexes.add(i);});
                        return [...indexes].sort((a,b)=>a-b);
                    }""")
                    for index in indexes:
                        page.evaluate('async i => await scrubTrajectory(i)',index)
                        values=page.evaluate("""i => {
                            const c=activeTrajectory,r=JSON.parse(document.getElementById('trajectory-readout').textContent);
                            const tr=gd.data[model.trajectory_traces[4]];
                            return {index:i,goal:c.goal_position?c.goal_position[i]:null,
                                rendered_goal:[tr.x[0],tr.y[0],tr.z[0]],
                                goal_readout:r.goal===undefined?null:r.goal,
                                meter:c.blast_meter?c.blast_meter[i]:null,
                                meter_readout:r.blast_meter===undefined?null:r.blast_meter,
                                score:c.score?c.score[i]:null,
                                score_readout:r.game_score===undefined?null:r.game_score};
                        }""",index)
                        if values['goal'] is None:
                            assert values['rendered_goal']==[None,None,None] and values['goal_readout'] is None,(name,trajectory,values)
                        else:
                            assert values['goal']==values['rendered_goal']==values['goal_readout'],(name,trajectory,values)
                        assert values['meter']==values['meter_readout'],(name,trajectory,values)
                        assert values['score']==values['score_readout'],(name,trajectory,values)
                        cases.append(dict(map=name,trajectory=trajectory,**values))
            if a.index:
                page.goto(a.index.resolve().as_uri())
                for href in page.locator('a').evaluate_all("links => links.map(a=>a.getAttribute('href'))"):
                    parsed=urlparse(href)
                    if parsed.scheme or not parsed.path:continue
                    path=(a.index.parent/unquote(parsed.path)).resolve()
                    assert path.is_file(),('missing review link',href,str(path))
                    links.append(dict(href=href,path=str(path)))
            assert not errors,errors
        finally:browser.close()
    atomic_json(a.directory/'goal_readout_checks.json',dict(cases=cases,javascript_errors=errors))
    if a.index:atomic_json(a.index.parent/'browser_links.json',dict(links=links,javascript_errors=errors))
    print(json.dumps(dict(goal_samples=len(cases),local_links=len(links),javascript_errors=errors)),flush=True)


if __name__=='__main__':main()
