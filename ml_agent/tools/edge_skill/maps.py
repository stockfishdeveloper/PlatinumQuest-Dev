"""Two procedural lip practice maps, using a real game DIF and native export.

Run as `python -m tools.edge_skill.maps` with PYTHONPATH=ml_agent.
Each isolated cell has a run-up, a short lip, and a raised landing platform.
The manifest stores exact geometry parameters, not a substitute collision mesh.
"""
import math
import subprocess
import hashlib

import numpy as np

from .common import ART, NIGHT, ROOT, MAP_DIR, ASSET, sha, write_json

GENERATED_ASSETS = {}


def baked_box(scale):
    """Bake dimensions into the DIF; nonuniform runtime scale + rotation is not calibrated."""
    import hxDif
    key = hashlib.sha256(words(scale).encode()).hexdigest()[:12]
    path = ROOT/'Marble Blast Platinum/platinum/data/multiplayer/interiors/custom/edge_skill'/('box_'+key+'.dif')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        d = hxDif.Dif.Load(str(ASSET)); it = d.interiors[0]
        s = np.asarray(scale)
        old_normals = np.asarray([[n.x,n.y,n.z] for n in it.normals])
        for p in it.points:
            p.x *= s[0]; p.y *= s[1]; p.z *= s[2]
        for plane in it.planes:
            plane.planeDistance /= float(np.linalg.norm(old_normals[plane.normalIndex]/s))
        for normal, old in zip(it.normals, old_normals):
            n = old/s; n /= np.linalg.norm(n)
            normal.x,normal.y,normal.z = map(float,n)
        for bounds in [it.boundingBox]+list(it.convexHulls):
            for j,axis in enumerate('XYZ'):
                for side in ('min','max'):
                    setattr(bounds,side+axis,float(getattr(bounds,side+axis)*s[j]))
        sphere=it.boundingSphere
        for j,axis in enumerate('XYZ'):
            setattr(sphere,'origin'+axis,float(getattr(sphere,'origin'+axis)*s[j]))
        sphere.radius=float(np.linalg.norm([2*s[0],2*s[1],.25*s[2]]))
        version=hxDif.Version();version.interiorType='mbg'
        hxDif.Dif.Save(d,version,str(path))
        check=hxDif.Dif.Load(str(path)).interiors[0]
        assert np.allclose([[p.x,p.y,p.z] for p in check.points],[[p.x,p.y,p.z] for p in it.points],atol=1e-5)
    GENERATED_ASSETS[path.name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    return '~/data/multiplayer/interiors/custom/edge_skill/'+path.name


def transform(cell, point):
    a = math.radians(cell['yaw_deg'])
    x, y, z = point
    return [cell['origin'][0] + math.cos(a)*x - math.sin(a)*y,
            cell['origin'][1] + math.sin(a)*x + math.cos(a)*y, cell['origin'][2]+z]


def words(values):
    return ' '.join('%.8g' % x for x in values)


def box(cell, lo, hi):
    # The source DIF bounds are [-4,0,0]..[0,4,.5].
    scale = [(hi[0]-lo[0])/4, (hi[1]-lo[1])/4, (hi[2]-lo[2])/.5]
    pos = transform(cell, [hi[0], lo[1], lo[2]])
    asset=baked_box(scale)
    return '''new InteriorInstance() {
position = "%s"; rotation = "0 0 1 %.8g"; scale = "%s";
interiorFile = "%s";
showTerrainInside = "0";
};''' % (words(pos), -cell['yaw_deg'], '1 1 1', asset)


def oriented_box(cell,lo,hi,matrix,pivot):
    from scipy.spatial.transform import Rotation
    from .common import rotation
    lo=np.asarray(lo);hi=np.asarray(hi);pivot=np.asarray(pivot);matrix=np.asarray(matrix)
    asset=baked_box((hi-lo)/[4,4,.5]);offset=np.array([hi[0],lo[1],lo[2]])
    pos=transform(cell,pivot+matrix@(offset-pivot));world=rotation(cell)@matrix
    vector=Rotation.from_matrix(world).as_rotvec();angle=float(np.linalg.norm(vector));axis=vector/max(angle,1e-12)
    if angle<1e-12:axis=np.array([0.,0.,1.])
    text='new InteriorInstance() { position = "%s"; rotation = "%s %.9g"; scale = "1 1 1"; interiorFile = "%s"; showTerrainInside = "0"; };'%(words(pos),words(axis),-math.degrees(angle),asset)
    corners=[transform(cell,pivot+matrix@(np.array([x,y,z])-pivot)) for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
    return text,corners


def sloped_maps(seed=6031126):
    """Tilt the lip line and landing plane independently, keeping a flat run-up."""
    from scipy.spatial.transform import Rotation
    import json
    target=NIGHT/'sloped_maps.json'
    if target.exists():return json.loads(target.read_text())
    template=(MAP_DIR/'edge_skill_runup_v2.mcs').read_text();header=template.split('new InteriorInstance()')[0]
    header=header.replace('area = "-100 -100 600 500"','area = "-200 -200 1000 1000"')
    header=header.replace('position = "-70 300 -20"; scale = "500 400 120"','position = "-150 900 -60"; scale = "1100 1100 200"')
    entries=[]
    for group in range(4):
        rng=np.random.default_rng(seed+group*10000);name='edge_skill_sloped_v6_%02d'%group
        split='train' if group<2 else 'dev' if group==2 else 'test';cells=[];geometry=[];gems=[]
        for i in range(16):
            width=float(rng.uniform(4.,10.));lip=float(rng.uniform(.75,1.25));depth=float(rng.uniform(.7,2.))
            beta=float(rng.uniform(-1,1)*min(math.radians(12),math.atan(.35/(width/2))))
            height=lip+float(rng.uniform(.5,1.2));length=float(rng.uniform(9,16));land_width=float(rng.uniform(3,7))
            theta=float(rng.uniform(-.7,.7));pitch=float(rng.uniform(-.22,.22));roll=float(rng.uniform(-.16,.16))
            r=Rotation.from_euler('ZYX',[theta,pitch,roll]).as_matrix();basis=r[:2,:2];normal=r[:,2]
            slope=(-normal[:2]/normal[2]).tolist();inverse=np.linalg.inv(basis).tolist()
            gap=float(rng.uniform(.5,2.5));offset=float(rng.uniform(-2.5,2.5))
            center=[depth+gap+abs(basis[0,0])*length/2+abs(basis[0,1])*land_width/2,offset+basis[1,0]*length/2]
            goal_xy=np.asarray(center)+basis@np.array([-length/2+rng.uniform(2.5,4.),0.])
            goal_z=height+float((goal_xy-center)@slope)+.189750001/normal[2]
            u=basis[:,0]/np.linalg.norm(basis[:,0]);exit_speed=float(rng.uniform(3.5,7.))
            c=dict(id=name+'_%02d'%i,index=i,map=name,split=split,origin=[(i%4)*160.,(i//4)*160.,0.],
                   yaw_deg=float(rng.uniform(-180,180)),lip_z=lip,lip_slope_y=math.tan(beta),lip_depth=depth,width=width*math.cos(beta),
                   landing_z=height,landing_x=center[0]-length/2,landing_end=center[0]+length/2,
                   landing_center=center,landing_size=[length,land_width],landing_yaw=math.atan2(u[1],u[0]),
                   landing_slope=slope,landing_inverse=inverse,goal_local=[*goal_xy.tolist(),goal_z],exit_speed=exit_speed,
                   family='high',friction=1.,restitution=1.,generation='diverse_sloped_v6',lip_tilt_deg=math.degrees(beta),
                   landing_pitch_deg=math.degrees(pitch),landing_roll_deg=math.degrees(roll))
            parts=[];corners=[]
            text,points=oriented_box(c,[-38,-14,-1],[0,14,0],np.eye(3),[0,0,0]);parts.append(text);corners+=points
            text,points=oriented_box(c,[0,-width/2,-1],[depth,width/2,lip],Rotation.from_euler('X',beta).as_matrix(),[0,0,lip]);parts.append(text);corners+=points
            land=dict(c,origin=transform(c,[*center,0.]))
            text,points=oriented_box(land,[-length/2,-land_width/2,-1],[length/2,land_width/2,height],r,[0,0,height]);parts.append(text);corners+=points
            c['geometry_corners']=corners;cells.append(c);geometry+=parts
            pos=transform(c,[-20,0,.1]);geometry.append('new Trigger() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "SpawnTrigger"; polyhedron = "0 0 0 1 0 0 0 -1 0 0 0 1"; add = "0 0 1"; gravity = "0"; };'%words(pos))
            gp=transform(c,[*goal_xy.tolist(),goal_z+.15]);gems.append('new Item() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "GemItem"; collideable = "0"; static = "1"; rotate = "1"; };'%words(gp))
        prefix=header.replace('edge_skill_runup_v2',name).replace('edgeskillrunupv2',name.replace('_',''))
        text=prefix+'\n'.join(geometry)+'\nnew SimGroup(GemGroups) { new SimGroup() {\n'+'\n'.join(gems)+'\n}; };\n}; }\n'
        path=MAP_DIR/(name+'.mcs')
        if path.exists() and path.read_text()!=text:raise FileExistsError(path)
        path.write_text(text);entries.append(dict(name=name,mission=str(path.relative_to(ROOT)),sha256=sha(path),cells=cells,
                                                  source_inventory=list(GENERATED_ASSETS.values()),generation='diverse_sloped_v6'))
    result=dict(schema='sloped_lip_lab_v6',seed=seed,maps=entries,splits='Groups0-1 training,2 development,3 sealed test')
    write_json(target,result);return result


def generate(name, transfer=False):
    cells = []
    for i in range(16):
        mode = 'low' if i < 8 else 'high'
        j = i % 8
        lip = ([.06, .10, .14, .18, .08, .12, .16, .20][j] if mode == 'low'
               else [.45, .60, .75, .90, 1.05, 1.20, 1.35, 1.50][j])
        if transfer:
            lip += .015 if mode == 'low' else .055
        rise = (.35 + .12*(j % 4)) if mode == 'low' else (.55 + .22*(j % 4))
        cell = dict(id='%s_%02d' % (name, i), index=i, map=name,
                    split='test' if transfer else ('dev' if j in (3, 7) else 'train'),
                    family=mode, origin=[(i % 4)*90., (i // 4)*65., 0.],
                    yaw_deg=([15., 30., 60., 90.][j % 4] if transfer else [0., 90.][j % 2]),
                    lip_z=lip, landing_z=lip+rise, lip_depth=1.5,
                    landing_x=2.4 + .15*(j % 3), landing_end=42.,
                    width=10. if not transfer else 8., goal_local=[7., 0., lip+rise+.19])
        cells.append(cell)
    info = '''function MP_PQ_%s_GetMissionInfo() {
return new ScriptObject() { name = "%s"; type = "Custom"; level = "9999";
desc = "Lip-contact skill laboratory; ordinary controls only";
artist = "Edge skill experiment"; game = "Ultra"; gameMode = "Hunt";
blast = "0"; Time = "3600000"; gemGroups = "1"; gems = "16";
maxGemsPerSpawn = "16"; radiusFromGem = "1000"; maxScore = "16";
interior0 = $usermods @ "/data/interiors_mbp/neutral2_square.dif"; interiors = "1";
}; }
if ($loadingMissionInfo) return;
if (!$Server::Hosting || $Server::_Dedicated) return;
function MP_PQ_%s_LoadMission() { return new SimGroup(MissionGroup) {
new MissionArea(MissionArea) { area = "-100 -100 600 500"; flightCeiling = "300"; flightCeilingRange = "20"; };
new Sky(Sky) { materialList = "~/data/skies_mbu/beginner/sky_beginner.dml"; visibleDistance = "10000";
useSkyTextures = "1"; renderBottomTexture = "1"; fogDistance = "10000"; };
new Sun() { direction = "0.5 0.3 -0.8"; color = "1 1 1"; ambient = "0.5 0.5 0.5"; };
new Trigger(StayHere) { position = "-70 300 -20"; scale = "500 400 120"; rotation = "0 0 1 0";
dataBlock = "InBoundsTrigger"; polyhedron = "0 0 0 1 0 0 0 -1 0 0 0 1"; };
''' % (name.replace('_', ''), name, name.replace('_', ''))
    geometry = []
    gems = []
    for c in cells:
        w = c['width']/2
        geometry += [box(c, [-16, -w, -1.], [0, w, 0.]),
                     box(c, [0, -w, -1.], [c['lip_depth'], w, c['lip_z']]),
                     box(c, [c['landing_x'], -w, -1.], [c['landing_end'], w, c['landing_z']])]
        pos = transform(c, [-10, 0, .1])
        geometry.append('new Trigger() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "SpawnTrigger"; polyhedron = "0 0 0 1 0 0 0 -1 0 0 0 1"; add = "0 0 1"; gravity = "0"; };' % words(pos))
        gp = transform(c, [7, 0, c['landing_z']+.7])
        gems.append('new Item() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "GemItem"; collideable = "0"; static = "1"; rotate = "1"; };' % words(gp))
    text = info + '\n'.join(geometry) + '\nnew SimGroup(GemGroups) { new SimGroup() {\n' + '\n'.join(gems) + '\n}; };\n}; }\n'
    path = MAP_DIR / (name+'.mcs')
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise FileExistsError('Refusing to overwrite a differing drill mission: '+str(path))
    path.write_text(text, encoding='utf-8')
    return dict(name=name, mission=str(path.relative_to(ROOT)), sha256=sha(path), cells=cells,
                source_inventory=list(GENERATED_ASSETS.values()))


def main():
    ART.mkdir(parents=True, exist_ok=True)
    # Inspect the source asset, so incorrect assumed dimensions fail before launch.
    import hxDif
    interior = hxDif.Dif.Load(str(ASSET)).interiors[0]
    p = np.array([[v.x, v.y, v.z] for v in interior.points])
    assert np.array_equal(p.min(0), [-4, 0, 0]) and np.array_equal(p.max(0), [0, 4, .5])
    maps = [generate('edge_skill_runup_v2'), generate('edge_skill_transfer_v2', True)]
    manifest = dict(schema='edge_skill_maps_v1', maps=maps, asset=str(ASSET.relative_to(ROOT)), asset_sha256=sha(ASSET),
                    reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    constraints=dict(powerups=False, blast=False, wall_face_skill=False, integration=False))
    previous = ART/'maps.json'
    if previous.exists() and not (ART/'maps_v0.json').exists():
        (ART/'maps_v0.json').write_bytes(previous.read_bytes())
    if previous.exists() and not (ART/'maps_v1.json').exists():
        (ART/'maps_v1.json').write_bytes(previous.read_bytes())
    write_json(previous, manifest)
    print('Generated', len(maps), 'maps with', sum(len(m['cells']) for m in maps), 'isolated cells', flush=True)


def diverse_maps(seed=301126, groups=8):
    """Independent lip and landing frames; compact maps exported by the real game."""
    NIGHT.mkdir(exist_ok=True)
    entries=[]
    # Reuse the compatible mission header, not its geometry or success distribution.
    template=(MAP_DIR/'edge_skill_runup_v2.mcs').read_text()
    header=template.split('new InteriorInstance()')[0]
    header=header.replace('area = "-100 -100 600 500"','area = "-200 -200 1000 1000"')
    header=header.replace('position = "-70 300 -20"; scale = "500 400 120"','position = "-150 900 -60"; scale = "1100 1100 200"')
    for group in range(groups):
        rng=np.random.default_rng(seed+group*10000)
        name='edge_skill_diverse_v3_%02d'%group
        split='train' if group<4 else ('dev' if group<6 else 'test')
        geometry=[];gems=[];cells=[]
        for i in range(16):
            width=float(rng.uniform(3.5,12.));lip=float(rng.uniform(.42,1.47))
            # The low-lip family remains a separate diagnostic, not the headline benchmark.
            if i==0:lip=float(rng.uniform(.035,.10))
            depth=float(rng.uniform(.5,2.5));rise=float(rng.uniform(.45,1.35));height=lip+rise
            length=float(rng.uniform(8.,17.));land_width=float(rng.uniform(2.2,8.))
            theta=float(rng.uniform(-.8,.8));gap=float(rng.uniform(.35,3.5));offset=float(rng.uniform(-3.5,3.5))
            front=[depth+gap+abs(math.sin(theta))*land_width/2,offset]
            u=np.array([math.cos(theta),math.sin(theta)])
            center=(np.asarray(front)+u*length/2).tolist();goal=(np.asarray(front)+u*float(rng.uniform(2.,4.))).tolist()
            c=dict(id=name+'_%02d'%i,index=i,map=name,split=split,origin=[(i%4)*160.,(i//4)*160.,0.],
                   yaw_deg=float(rng.uniform(-180,180)),lip_z=lip,lip_depth=depth,width=width,
                   landing_z=height,landing_x=front[0],landing_end=front[0]+length,
                   landing_center=center,landing_size=[length,land_width],landing_yaw=theta,
                   goal_local=[*goal,height+.18975],exit_speed=float(rng.uniform(3.,8.)),
                   family='low' if i==0 else 'high',friction=1.,restitution=1.,generation='diverse_v3')
            cells.append(c)
            geometry += [box(c,[-38,-14,-1],[0,14,0]),box(c,[0,-width/2,-1],[depth,width/2,lip])]
            land=dict(c,origin=transform(c,[*center,0.]),yaw_deg=c['yaw_deg']+math.degrees(theta))
            geometry.append(box(land,[-length/2,-land_width/2,-1],[length/2,land_width/2,height]))
            pos=transform(c,[-20,0,.1])
            geometry.append('new Trigger() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "SpawnTrigger"; polyhedron = "0 0 0 1 0 0 0 -1 0 0 0 1"; add = "0 0 1"; gravity = "0"; };'%words(pos))
            gp=transform(c,[*goal,height+.3])
            gems.append('new Item() { position = "%s"; rotation = "0 0 1 0"; scale = "1 1 1"; dataBlock = "GemItem"; collideable = "0"; static = "1"; rotate = "1"; };'%words(gp))
        prefix=header.replace('edge_skill_runup_v2',name).replace('edgeskillrunupv2',name.replace('_',''))
        text=prefix+'\n'.join(geometry)+'\nnew SimGroup(GemGroups) { new SimGroup() {\n'+'\n'.join(gems)+'\n}; };\n}; }\n'
        path=MAP_DIR/(name+'.mcs')
        if path.exists() and path.read_text()!=text:raise FileExistsError(path)
        path.write_text(text)
        entries.append(dict(name=name,mission=str(path.relative_to(ROOT)),sha256=sha(path),cells=cells,
                            source_inventory=list(GENERATED_ASSETS.values()),generation='diverse_v3'))
    manifest=dict(schema='diverse_edge_lab_v3',seed=seed,maps=entries,
                  splits='Geometry groups 0-3 train, 4-5 development, 6-7 sealed final test. Low lips reported separately.',
                  constraints=dict(blast=False,powerups=False,integration=False))
    path=NIGHT/'maps.json'
    if path.exists() and __import__('json').loads(path.read_text())!=manifest:raise FileExistsError(path)
    write_json(path,manifest)
    return manifest


def real_fixtures(seed=730311,per_map=6):
    """Geometry-selected real-map trials, independent of solver success.

    Ground resets and explicitly synthetic airborne encounters are separate.
    The latter validate execution on real collision geometry, not route reachability.
    """
    import copy
    from nav.marble_dynamics.edge_hits import lips
    from .common import ML,engine_for
    output=NIGHT/'real_manifest.json'
    if output.exists():return __import__('json').loads(output.read_text())
    cases=[];inventory=[];radius=.189750001
    for map_index,path in enumerate(sorted((ML/'collision_maps').glob('*_Hunt.npz'))):
        name=path.stem;e=engine_for(name);m=e.mesh;edges,normals=lips(m.triangles,m.normals,m.boundary_edges)
        rng=np.random.default_rng(seed+map_index);candidates=[]
        for edge,normal in zip(edges,normals):
            if normal[2]<.995 or abs(edge[0,2]-edge[1,2])>.01:continue
            delta=edge[1]-edge[0];length=float(np.linalg.norm(delta))
            if length<2.:continue
            middle=edge.mean(0);side=np.array([delta[1],-delta[0],0.])/length
            for sign in (1,-1):
                forward=side*sign;tangent=np.array([-forward[1],forward[0],0.])
                def floor(x,y=0.,above=.4,below=6.):
                    p=middle+forward*x+tangent*y
                    return m.raycast(p+[0,0,above],p-[0,0,below],cull_backfaces=True)
                inside_hit=floor(.45);outside_hit=floor(-.45)
                if inside_hit is None or abs(inside_hit['point'][2]-middle[2])>.025:continue
                if outside_hit is not None and outside_hit['point'][2]>middle[2]-.25:continue
                depth=None
                for possible in (3.5,5.,7.,9.):
                    hits=[floor(x,y) for x in np.linspace(.35,possible-.25,6) for y in (-.6,0.,.6)]
                    if all(h is not None and abs(h['point'][2]-middle[2])<.025 for h in hits):depth=possible
                    else:break
                if depth is None:continue
                width=1.8
                for possible in (2.5,3.5,5.):
                    hits=[floor(x,y) for x in (.4,depth/2,depth-.3) for y in (-possible/2+.1,possible/2-.1)]
                    if all(h is not None and abs(h['point'][2]-middle[2])<.025 for h in hits):width=possible
                    else:break
                ground=[]
                for distance in (3.,6.,10.):
                    h=floor(-distance)
                    if h is None:continue
                    height=middle[2]-h['point'][2]
                    if not .35<height<1.48:continue
                    run=[floor(-x) for x in np.linspace(.3,distance,8)]
                    if all(v is not None and abs(v['point'][2]-h['point'][2])<.025 for v in run):ground.append((distance,float(h['point'][2])))
                base=ground[0][1] if ground else float(middle[2]-1.)
                candidates.append(dict(point=middle.tolist(),forward=forward.tolist(),depth=depth,width=width,
                                       edge=edge.tolist(),edge_length=length,ground=ground,base=base))
        # Prefer opportunities with genuine ground approaches, but reserve air
        # encounters so maps with higher or isolated ledges are still represented.
        rng.shuffle(candidates);candidates.sort(key=lambda c:not bool(c['ground']))
        chosen=[]
        for c in candidates:
            if any(np.linalg.norm(np.asarray(c['point'])-o['point'])<5. for o in chosen):continue
            chosen.append(c)
            if len(chosen)>=per_map:break
        inventory.append(dict(map=name,mesh_sha256=sha(path),edges=len(edges),eligible=len(candidates),selected=len(chosen)))
        for i,source in enumerate(chosen):
            point=source['point'];f=source['forward'];height=point[2]-source['base'];split='real_dev' if i%2==0 else 'real_test'
            speed=3. if source['depth']<5 else 5.;gx=1. if source['depth']<5 else 2.
            cell=dict(id=name+'_v3_edge_%02d'%i,index=i,map=name,split=split,origin=[point[0],point[1],source['base']],
                      yaw_deg=math.degrees(math.atan2(f[1],f[0])),lip_z=height,lip_depth=source['depth'],width=source['edge_length'],
                      landing_z=height,landing_x=0.,landing_end=source['depth'],landing_center=[source['depth']/2,0.],
                      landing_size=[source['depth'],source['width']],landing_yaw=0.,goal_local=[gx,0.,height+radius],
                      exit_speed=speed,family='high',source_edge=source['edge'],friction=1.,restitution=1.)
            starts=[]
            for j,(distance,z) in enumerate(source['ground']):
                for vx in (7.,13.):starts.append(dict(position=[-distance,0.,z-source['base']+radius],velocity=[vx,0.,0.],
                        airborne=False,start_kind='ground',start_floor=z-source['base'],initial_state_kind='ground_reset'))
            for contact_vz,t,vx,nz in ((2.,.20,12.,.8),(-2.,.12,12.,.9),(-3.,.16,16.,.85),(3.,.28,10.,.75)):
                z=height+radius*nz-contact_vz*t-10*t*t;vz=contact_vz+20*t
                p=[-radius*math.sqrt(1-nz*nz)-vx*t,0.,z]
                world=transform(cell,p);contacts=e.native_sim.contacts(world,radius)
                if len(contacts) and np.min(contacts[:,8])<radius-.025:continue
                starts.append(dict(position=p,velocity=[vx,0.,vz],airborne=True,start_kind='air_rising' if vz>.6 else 'air_descending',
                                   start_floor=0.,initial_state_kind='synthetic_air_encounter'))
            for j,start in enumerate(starts):
                c=dict(id=cell['id']+'_'+start['start_kind']+'_%03d'%j,cell=copy.deepcopy(cell),goal=cell['goal_local'],
                       exit_velocity=[speed,0.],goal_radius=.7,seed=seed+map_index*10000+i*100+j,split=split,required_contact_mode=None,**start)
                vx,vy,_=c['velocity'];c['spin']=[-vy/radius,vx/radius,0.];cases.append(c)
        print(name,'real geometry candidates',len(candidates),'selected',len(chosen),flush=True)
    for split in ('real_dev','real_test'):write_json(NIGHT/(split+'_cases.json'),[c for c in cases if c['split']==split])
    result=dict(seed=seed,inventory=inventory,cases=len(cases),
                selection='Geometry only; alternating selected edges per map are development and sealed real test.',
                limitations='Synthetic reset states, not collected live route opportunities. Horizontal lips and landings only.')
    write_json(output,result);return result


if __name__ == '__main__':
    main()
