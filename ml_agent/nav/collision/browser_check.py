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
                cases=json.loads((args.directory.parent/(name+'.validation.json')).read_text())['sweeps']['disagreements']
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
                first=page.locator('#objects input').first
                first.uncheck()
                page.wait_for_function('gd.data[0].visible === false')
                first.check()
                page.wait_for_function('gd.data[0].visible === true')
                row=dict(map=name,**counts,javascript_errors=errors,ambiguity_contacts=contacts,
                         ambiguity_focus_buttons_checked=len(cases))
                results.append(row)
                print(json.dumps(row),flush=True)
                if errors:
                    raise RuntimeError('Viewer JavaScript error: '+str(errors))
            page.goto((args.directory/'index.html').resolve().as_uri())
            assert page.locator('tbody tr').count()==10
        finally:
            browser.close()
    atomic_json(args.directory/'browser_checks.json',dict(maps=results,index_rows=10))


if __name__=='__main__':
    main()
