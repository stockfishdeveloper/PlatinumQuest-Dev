"""Optional headless-browser review check; Playwright is a verification dependency.

Use a separate browser; never attach to the user's browsing session.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from nav.collision.mesh import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('ml_agent/collision_maps/viewer'))
    parser.add_argument('--geometry-directory',type=Path,help='Geometry/validation root for a separate pilot viewer')
    parser.add_argument('--browser',default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    args=parser.parse_args()
    names=[r['map'] for r in json.loads((args.directory/'summary.json').read_text())['maps']]
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.browser,headless=True,
                    args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        try:
            page=browser.new_page(viewport=dict(width=1600,height=1000))
            for name in names:
                errors=[]
                page.on('pageerror',lambda e: errors.append(str(e)))
                page.goto((args.directory/(name+'.html')).resolve().as_uri(),wait_until='load')
                for link in page.locator('header a').evaluate_all("links => links.map(a=>a.getAttribute('href'))"):
                    assert (args.directory/link).is_file(),(name,'missing local link',link)
                page.wait_for_function("document.getElementById('geometry')._fullLayout && document.getElementById('geometry')._fullLayout.scene._scene",timeout=30000)
                counts=page.evaluate("""async () => {
                    const vertices=new Float64Array(model.info.triangles*9);let faces=0;
                    model.records.forEach((r,i)=>{const tr=gd._fullData[i];faces+=tr.i.length;
                      r.ids.forEach((id,k)=>{for(let v=0;v<3;v++){
                        vertices[id*9+v*3]=tr.x[k*3+v];vertices[id*9+v*3+1]=tr.y[k*3+v];vertices[id*9+v*3+2]=tr.z[k*3+v];}});});
                    const digest=await crypto.subtle.digest('SHA-256',vertices.buffer);
                    const geomHash=Array.from(new Uint8Array(digest)).map(x=>x.toString(16).padStart(2,'0')).join('');
                    return {triangles:model.info.triangles,rendered_triangles:faces,rendered_geometry_sha256:geomHash,
                      hash:model.info.mesh_sha256,traces:gd.data.length,objects:model.records.length};
                }""")
                page.select_option('#color','orientation')
                page.select_option('#drag','pan')
                page.wait_for_function("gd.layout.scene.dragmode === 'pan'")
                page.select_option('#drag','orbit')
                page.check('#edges')
                page.click('#under')
                page.wait_for_timeout(300)
                page.select_option('#color','material')
                page.uncheck('#edges')
                page.click('#reset')
                page.evaluate("gd.emit('plotly_click',{points:[{curveNumber:0,pointNumber:0,customdata:model.records[0].ids[0]}]})")
                assert 'friction' in page.locator('#selected').inner_text()
                if name in ('Concentric_Hunt','Duplex_Hunt','Cragmire_Hunt','KingOfTheMarble_Hunt'):
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(args.directory/(name+'.preview.png')),full_page=True)
                geometry_directory=args.geometry_directory or args.directory.parent
                cases=json.loads((geometry_directory/(name+'.validation.json')).read_text())['sweeps']['disagreements']
                cases=[r for r in cases if r['reason']=='material_boundary_tie']
                contacts=page.evaluate("""model.ambiguities.map((a,i)=>{
                    const tr=gd._fullData[model.ambiguity_traces[1]];return [tr.x[i],tr.y[i],tr.z[i]];})""")
                assert contacts==[r['engine']['point'] for r in cases],(name,'ambiguity marker location')
                assert page.locator('#ambiguity-list button').count()==len(cases)
                if cases:
                    assert page.locator('#ambiguity-panel').is_visible()
                    for i,case in enumerate(cases):
                        page.locator('#ambiguity-list button').nth(i).click()
                        page.wait_for_function("JSON.parse(document.getElementById('selected').textContent).test_id === "+str(case['id']))
                        camera=page.evaluate('gd.layout.scene.camera')
                        # A bad mutable range calculation previously sent later buttons many map widths away.
                        assert all(abs(camera['center'][axis])<2 for axis in ('x','y','z')),(name,case['id'],camera)
                        if i==2:
                            page.wait_for_timeout(500)
                            page.screenshot(path=str(args.directory/(name+'.ambiguity_focus.png')),full_page=True)
                    page.uncheck('#ambiguities')
                    page.wait_for_function('model.ambiguity_traces.every(i=>gd.data[i].visible===false)')
                    page.locator('#ambiguity-list button').first.click()
                    page.wait_for_function('model.ambiguity_traces.every(i=>gd.data[i].visible===true)')
                    page.wait_for_function("JSON.parse(document.getElementById('selected').textContent).test_id === "+str(cases[0]['id']))
                    assert all(abs(x)<2 for x in page.evaluate('Object.values(gd.layout.scene.camera.center)'))
                    page.click('#reset')
                trajectory_checks=[]
                trajectory_count=page.evaluate('model.trajectories.length')
                if trajectory_count:
                    candidates=page.evaluate("""Array.from(new Set([0,model.trajectories.length-1,
                        model.trajectories.reduce((best,c,i)=>c.max_position_error_u>model.trajectories[best].max_position_error_u?i:best,0),
                        model.trajectories.findIndex(c=>c.family==='edge_timing'),model.trajectories.findIndex(c=>c.family==='drop_spin'),
                        ...model.trajectories.map((c,i)=>['complete_native_hunt_round','incomplete_native_hunt_attempt','native_script_response'].includes(c.family)?i:-1)])).filter(i=>i>=0)""")
                    for index in candidates:
                        page.select_option('#trajectory',str(index))
                        page.wait_for_function("""index => activeTrajectory===model.trajectories[index] &&
                            gd.data[model.trajectory_traces[2]].visible===true &&
                            ['actual','predicted'].every((basis,j)=>{
                              const tr=gd.data[model.trajectory_traces[j]];
                              const points=typeof trajectoryLine==='function'?trajectoryLine(activeTrajectory,basis).points:activeTrajectory[basis+'_position'];
                              return JSON.stringify(points)===JSON.stringify(tr.x.map((x,i)=>[x,tr.y[i],tr.z[i]]));
                            })""",arg=index)
                        values=page.evaluate("""() => {
                            const c=activeTrajectory,checks=['actual','predicted'].map((basis,j)=>{
                              const tr=gd.data[model.trajectory_traces[j]];
                              const points=typeof trajectoryLine==='function'?trajectoryLine(c,basis).points:c[basis+'_position'];
                              return JSON.stringify(points)===JSON.stringify(tr.x.map((x,i)=>[x,tr.y[i],tr.z[i]]));});
                            const gaps=c.path_break_before?c.path_break_before.filter(Boolean).length:0;
                            const actual=gd.data[model.trajectory_traces[0]],blank=actual.x.filter(x=>x===null).length;
                            return {trial:c.id,source:c.source,samples:c.time_s.length,exact_coordinates:checks,
                              path_breaks:gaps,actual_line_blank_points:blank};
                        }""")
                        assert all(values['exact_coordinates']),(name,index,'trajectory coordinate mismatch')
                        if page.evaluate('Boolean(activeTrajectory.path_break_before)'):
                            assert values['path_breaks']==values['actual_line_blank_points'],(name,index,'respawn/control path gaps')
                        page.click('#trajectory-worst')
                        if page.evaluate('activeTrajectory.position_error.some(Number.isFinite)'):
                            page.wait_for_function("JSON.parse(document.getElementById('trajectory-readout').textContent).position_error_u === activeTrajectory.max_position_error_u")
                        else:
                            assert page.evaluate("JSON.parse(document.getElementById('trajectory-readout').textContent).position_error_u === null"),(name,index,'unknown prediction must remain unknown')
                        page.click('#trajectory-focus')
                        page.wait_for_timeout(150)
                        assert all(abs(x)<2 for x in page.evaluate('Object.values(gd.layout.scene.camera.center)'))
                        marker=page.evaluate("""() => {const i=Number(timeSlider.value),tr=gd.data[model.trajectory_traces[2]];
                            return {actual:activeTrajectory.actual_position[i],shown:[tr.x[0],tr.y[0],tr.z[0]]};}""")
                        assert marker['actual']==marker['shown'],(name,index,'selected measured point was lost at a path gap')
                        if page.evaluate('Boolean(activeTrajectory.actual_radius)'):
                            radius_checks=page.evaluate("""async () => {
                                const c=activeTrajectory,points=Array.from(new Set([0,c.time_s.length-1,
                                  ...c.native_use_events,...c.outcomes.observed_body_boundary_samples]));const checks=[];
                                for(const sample of points){await scrubTrajectory(sample);const tr=gd.data[model.radius_trace],
                                  center=c.actual_position[sample],r=c.actual_radius[sample],shown=[tr.x,tr.y,tr.z],
                                  readout=JSON.parse(document.getElementById('trajectory-readout').textContent);
                                  checks.push({sample,radius:r,shown_radius:readout.actual_radius,
                                    stage:readout.sample_stage,expected_stage:c.sample_stage[sample],visible:tr.visible,
                                    error:Math.max(...shown.map((axis,i)=>Math.max(
                                      Math.abs(Math.min(...axis)-(center[i]-r)),Math.abs(Math.max(...axis)-(center[i]+r)))))});}
                                return checks;}""")
                            assert all(c['visible'] and c['radius']==c['shown_radius'] and c['stage']==c['expected_stage'] and c['error']<1e-8 for c in radius_checks),(name,index,'native radius or event phase display differs')
                            values['radius_checks']=radius_checks
                        if page.evaluate('Boolean(activeTrajectory.task_goal_position)'):
                            task=page.evaluate("""() => {
                                const i=Number(timeSlider.value),expected=activeTrajectory.task_goal_position[i],
                                  tr=gd.data[model.trajectory_traces[5]],readout=JSON.parse(document.getElementById('trajectory-readout').textContent);
                                return {expected,shown:[tr.x[0],tr.y[0],tr.z[0]],visible:tr.visible,
                                  readout:readout.task_goal,actor:readout.actor_input_goal,
                                  expectedActor:activeTrajectory.actor_goal_position[i]};}""")
                            assert task['expected']==task['shown']==task['readout'],(name,index,'task destination mismatch')
                            assert task['actor']==task['expectedActor'],(name,index,'actor input destination mismatch')
                            assert task['visible']==all(v is not None for v in task['expected']),(name,index,'unknown task visibility')
                            values['task_and_actor_destination_check']=True
                        trajectory_checks.append(values)
                    page.screenshot(path=str(args.directory/(name+'.dynamics_preview.png')),full_page=True)
                    page.select_option('#trajectory','')
                    page.wait_for_function('model.trajectory_traces.every(i=>gd.data[i].visible===false)')
                    assert page.evaluate("typeof model.radius_trace !== 'number' || gd.data[model.radius_trace].visible===false")
                    page.click('#reset')
                first=page.locator('#objects input').first
                first.uncheck()
                page.wait_for_function('gd.data[0].visible === false')
                first.check()
                page.wait_for_function('gd.data[0].visible === true')
                row=dict(map=name,**counts,javascript_errors=errors,ambiguity_contacts=contacts,
                         ambiguity_focus_buttons_checked=len(cases),trajectory_count=trajectory_count,trajectory_checks=trajectory_checks)
                results.append(row)
                print(json.dumps(row),flush=True)
                if errors:
                    raise RuntimeError('Viewer JavaScript error: '+str(errors))
            page.goto((args.directory/'index.html').resolve().as_uri())
            expected_links={name+'.html' for name in names}
            map_links=page.locator('a').evaluate_all("links => links.map(a=>a.getAttribute('href')).filter(h=>h && !h.includes('/') && h.endsWith('.html') && h!=='index.html')")
            assert len(map_links)==len(names) and set(map_links)==expected_links,('gallery map links',map_links)
            index_rows=page.locator('tbody tr').filter(has=page.locator('a[href$=".html"]')).count()
            if index_rows:
                assert index_rows==len(names),('gallery rows',index_rows,len(names))
            else:
                assert page.locator('li').count()==len(names),('gallery list entries',len(names))
        finally:
            browser.close()
    atomic_json(args.directory/'browser_checks.json',dict(maps=results,index_rows=index_rows,index_map_links=len(map_links)))


if __name__=='__main__':
    main()
