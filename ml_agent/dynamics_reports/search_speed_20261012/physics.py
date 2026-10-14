"""Independent general contact-dynamics baseline over the verified collision mesh.

Engine mechanics supply an interpretable starting model; native trajectories validate
the port. This is NOT the native engine or a certification of future landings.
"""
import copy
import gzip
import hashlib
import json
import math
import numpy as np


def unit(v):
    v=np.asarray(v,dtype=float);n=np.linalg.norm(v)
    return v/n if n>1e-12 else np.zeros(3)


def vector_state(s):
    return np.concatenate([s['position'],s['velocity'],s['spin']])


class GeneralPhysics:
    def __init__(self,mesh):
        self.mesh=mesh
        # Source keys/planes can be shared by distinct polygons. Recover exact
        # fan membership from the archived export, without altering reviewed NPZs.
        raw=gzip.decompress((mesh.path.parent/mesh.info['raw_snapshot_file']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=mesh.info['raw_snapshot_sha256']:
            raise ValueError('Native polygon snapshot hash mismatch')
        seen=set();membership=[];self.polygons=[]
        for polygon in map(json.loads,raw.decode().splitlines()):
            if polygon['kind']!='polygon':continue
            key=(polygon['object_id'],polygon['surface_key'],polygon['material_index'],
                 tuple(sorted(tuple(v) for v in polygon['vertices'])),tuple(polygon['normal']))
            if key in seen:continue
            seen.add(key)
            # Exported nine-digit values round-trip native F32 coefficients and
            # vertices. Restore their F32 bits before F64 contact arithmetic.
            n=np.asarray(polygon['normal'],np.float32).astype(float)
            vertices=np.asarray(polygon['vertices'],np.float32).astype(float)
            previous=np.roll(vertices,1,axis=0)
            # Torque's three-point PlaneD cross product is opposite the usual
            # convention, matching the clockwise native polygon winding.
            edge_normals=-np.cross(n,vertices-previous)
            lengths=np.linalg.norm(edge_normals,axis=1)
            edge_normals=np.divide(edge_normals,lengths[:,None],out=np.zeros_like(edge_normals),where=lengths[:,None]>0)
            self.polygons.append(dict(vertices=vertices,normal=n,d=float(np.float32(polygon['plane_d'])),edges=edge_normals,
                                      edge_valid=lengths>0,first_triangle=len(membership),native=polygon))
            membership.extend([len(seen)-1]*(len(polygon['vertices'])-2))
        if len(membership)!=len(mesh.triangles):raise ValueError('Native fan membership mismatch')
        self.polygon_ids=np.asarray(membership)
        from nav.marble_dynamics.contact_order import NativeContactOrder
        self.contact_order=NativeContactOrder(mesh,self.polygons)

    def contacts(self,p,radius,ambiguities=None):
        # Group fan triangles by native polygon before selecting its closest point.
        # This removes artificial triangulation-edge contacts inside a polygon.
        ids=self.mesh.query_aabb(p-radius-.0001,p+radius+.0001)
        if not len(ids):return []
        result=[]
        for pid in np.unique(self.polygon_ids[ids]):
            polygon=self.polygons[pid];n=polygon['normal'];vertices=polygon['vertices'];edges=polygon['edges']
            distance=float(n[1]*p[1]+n[2]*p[2]+n[0]*p[0]+polygon['d'])
            if abs(np.float32(distance))>radius+.0001:continue
            projected_distance=float(n[0]*p[0]+n[1]*p[1]+n[2]*p[2]+polygon['d'])
            contact=p-n*projected_distance;point=contact.copy();rejected=False
            # Native contact generation uses original vertices and the stored
            # plane, whose F32 rounding need not put each vertex on that plane.
            separation=float(np.float32(math.sqrt(max(0.,radius*radius-distance*distance)))) if abs(distance)<=radius else float('nan')
            # At the sqrt boundary, one last-bit difference can turn separation
            # into NaN in native code. Its edge rejection then compares against
            # NaN (false). Preserve the nominal set, but expose a valid alternate
            # contact rather than pretending this branch is predictable.
            boundary=abs(abs(distance)-radius)<=1e-12
            for j,vertex in enumerate(vertices):
                if not polygon['edge_valid'][j]:continue
                edge=edges[j];vert_distance=float(edge@(contact-vertex))
                if vert_distance<0:
                    if vert_distance<-(separation+.0001):
                        rejected=True
                        if ambiguities is None or not boundary:break
                    before=vertices[j-1];along=unit(np.cross(edge,n))
                    if along@(contact-vertex)>=0:
                        if (-along)@(contact-before)>=0:
                            point=contact-edge*vert_distance;break
                        point=before
                    else:point=vertex
            if rejected and (ambiguities is None or not boundary):continue
            delta=p-point;distance=float(np.linalg.norm(delta))
            if distance>radius+.0001:continue
            idx=polygon['first_triangle'];normal=delta/distance if distance else n
            mat=self.mesh.info['materials'][int(self.mesh.material_ids[idx])]
            if rejected:
                ambiguities.append(dict(kind='ambiguous_contact',position=p.tolist(),
                                        candidate_normal=normal.tolist(),candidate_point=point.tolist(),
                                        triangle_id=idx,plane_distance=projected_distance,radius=radius,
                                        reason='Roundoff at sphere/plane tangency can change native NaN edge rejection'))
                continue
            result.append(dict(normal=normal,point=point,distance=distance,triangle_id=idx,polygon_id=int(pid),material=mat))
        return self.contact_order.reorder(result,p,radius)

    @staticmethod
    def cancel(v,w,contacts,centered,no_bounce,s,params):
        slide=centered;r=s['radius']
        for iteration in range(7):
            changed=False
            for c in contacts:
                n=c['normal'];dot=float(v[2]*n[2]+v[1]*n[1]+v[0]*n[0])
                if not ((iteration==0 and dot<0) or dot<-.0001):continue
                speed=np.linalg.norm(v);normal_v=n*dot
                if no_bounce:
                    v-=normal_v
                elif not slide and dot>-params['max_dot_slide']*speed:
                    v-=normal_v;v=unit(v)*speed;slide=True
                elif dot>-params['min_bounce_velocity']:
                    v-=normal_v
                else:
                    base=.01 if s['effects'][4][0] else .9 if s['effects'][3][0] else params['bounce_restitution']
                    restitution=c['material']['restitution']*base
                    tangent=np.cross(w,-n*r)+v-dot*n
                    mag=np.linalg.norm(tangent)
                    if mag>1e-12:
                        friction=float(np.float32(params['bounce_kinetic_friction']*c['material']['friction']))
                        delta_mag=min(friction*2.5*(-dot)/r,mag/r)
                        dw=np.cross(n,tangent/mag)*delta_mag
                        w+=dw;v-=np.cross(dw,n*r)
                    v-=(1+restitution)*normal_v
                changed=True
            if not changed:break
        return v,w

    @staticmethod
    def axes(s,move,dt_ms):
        # The pilot contract uses ordinary downward gravity. Arbitrary gravity
        # rotations need their native matrix, not an invented yaw convention.
        gravity=np.asarray(s['gravity_direction'],dtype=float)
        if np.linalg.norm(gravity-[0,0,-1])>1e-6:
            raise ValueError('Gravity rotation matrix is not yet modeled')
        yaw=s['yaw']
        if s.get('smooth_camera',False):
            next_yaw=s.get('next_yaw',yaw)+move.get('yaw',0)
            diff=(next_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            yaw+=diff*(1-.5**(dt_ms/16))
            s['next_yaw']=next_yaw
        else:yaw+=move.get('yaw',0)
        s['yaw']=yaw%(2*math.pi)
        # Torque's clockwise z Euler matrix maps forward to (sin(yaw),cos(yaw)).
        side=np.array([math.cos(yaw),-math.sin(yaw),0.])
        forward=np.array([math.sin(yaw),math.cos(yaw),0.])
        s['camera_offset']=(forward*math.cos(s.get('pitch',.45))-np.array([0,0,1])*math.sin(s.get('pitch',.45))).tolist()
        return side,forward

    def step(self,state,move,dt_ms=16,diagnostic=False):
        s=copy.deepcopy(state);params=s['parameters'];r=float(np.float32(s['radius']))
        s['radius']=r
        for key,value in params.items():
            if key!='effect_duration_ms':params[key]=float(np.float32(value))
        p=np.asarray(s['position'],float);v=np.asarray(s['velocity'],float);w=np.asarray(s['spin'],float)
        if s['mode']!=0:raise ValueError('Only ordinary playable marble mode is modeled')
        side,forward=self.axes(s,move,dt_ms)
        gravity=np.asarray(s['gravity_direction'],float)
        direction=side*move['x']+forward*move['y']
        clock=int(s.get('clock_ms',0))
        deadlines=s.get('effect_end_clock_ms',[clock+int(e[1]) for e in s['effects']])
        held=int(s['held'])
        if move.get('use') and held:
            if held==1:v-=gravity*20.
            elif held==2:
                if 'best_contact_normal' not in s:raise ValueError('Super Speed requires recorded previous best-contact normal')
                previous=np.asarray(s['best_contact_normal'],float)
                aim=forward.astype(np.float32)
                projection=float(np.float32(previous)@aim)
                boost=(aim-previous*projection).astype(np.float32)
                length=np.sqrt(boost@boost)
                boost=boost/length if length else np.asarray([1,0,0],np.float32)
                v+=(boost*np.float32(25.)).astype(float)
            elif 3<=held<=5:
                s['effects'][held]=[True,params.get('effect_duration_ms',[5000]*6)[held]]
            else:raise ValueError('Script-driven activation requires an external-event model')
            s['held']=0
            duration=params.get('effect_duration_ms',[5000,1000,1000,5000,5000,5000])[held]
            s['effects'][held]=[True,duration];deadlines[held]=clock+duration
        total=dt_ms/1000.;events=[];iterations=0
        # Same <=8 ms semi-implicit substep structure as native physics.
        while total>1e-10:
            chunk=min(.008,total);left=chunk
            while left>1e-10:
                contacts=self.contacts(p,r,ambiguities=events)
                control=np.zeros(3);desired=np.zeros(3)
                centered=move['x']==0 and move['y']==0
                radius_up=-gravity*r
                if not centered:
                    rolling=np.cross(w,radius_up)
                    current=np.array([rolling@side,rolling@forward])
                    # JSON renders exactly integral F32 values without a decimal
                    # point. Never let NumPy infer an integer target and truncate
                    # the engine's preserved overspeed rolling component.
                    target=np.asarray([move['x'],move['y']],dtype=float)*params['max_roll_velocity']
                    for j in range(2):
                        if target[j]>0 and current[j]>target[j] or target[j]<0 and current[j]<target[j]:target[j]=current[j]
                    desired=np.cross(radius_up,side*target[0]+forward*target[1])/(r*r)
                    control=desired-w
                    control*=min(1.,params['angular_acceleration']/max(np.linalg.norm(control),1e-12))
                if not centered and len(contacts)>1:
                    dots=[float(v[2]*c['normal'][2]+v[1]*c['normal'][1]+v[0]*c['normal'][0]) for c in contacts]
                    fragile=[i for i,dot in enumerate(dots[:-1]) if abs(dot)<1e-6 and min(dots[i+1:])<-params['min_bounce_velocity']]
                    if fragile:events.append(dict(kind='ambiguous_contact',position=p.tolist(),
                                                  grazing_contacts=fragile,normal_velocities=dots,
                                                  reason='Near-zero first contact can change slide/bounce branch before later impact'))
                v,w=self.cancel(v,w,contacts,centered,False,s,params)
                A=gravity*params['gravity']*(.25 if s['effects'][5][0] else 1.)
                if not contacts:A+=direction*params['air_acceleration']*(2. if s['effects'][5][0] else 1.)
                force_contacts=[c for c in contacts if c['material']['force']!=0]
                if force_contacts:
                    n=unit(sum((c['normal'] for c in force_contacts),np.zeros(3)))
                    impulse=force_contacts[-1]['material']['force']/params.get('mass',1.)
                    dot=n@v
                    if impulse>dot:A+=n*(impulse-max(0.,dot))/left
                nf=[-float(c['normal']@A) for c in contacts]
                best=int(np.argmax(nf)) if nf and max(nf)>0 else None
                angular=np.zeros(3)
                if best is not None:
                    c=contacts[best];n=c['normal'];normal_force=nf[best]
                    s['best_contact_normal']=n.tolist()
                    events.append(dict(kind='support',normal=n.tolist(),triangle_id=c['triangle_id']))
                    if move.get('jump'):
                        impulse=params['jump_impulse']-max(0.,float(np.float32(v@n)))
                        if impulse>0:v+=n*impulse;events.append(dict(kind='jump',normal=n.tolist()))
                for c in contacts:
                    # Native constraints update acceleration sequentially. A
                    # second surface must see what the first already canceled.
                    force=-float(c['normal']@A)
                    if force>0 and v@c['normal']<=.0001:A+=c['normal']*force
                if best is not None:
                    c=contacts[best];n=c['normal']
                    at_contact=np.cross(w,-n*r)+v
                    mag=np.linalg.norm(at_contact)
                    slipping=False;angular_friction=np.zeros(3);linear_friction=np.zeros(3)
                    if mag>1e-12:
                        mu=float(np.float32(params['kinetic_friction']*c['material']['friction']))
                        if mu<0:raise ValueError('Random-force material is not modeled')
                        alpha=mu*2.5*normal_force/r;accel=normal_force*mu
                        delta=(r*alpha+accel)*left
                        slipping=mag>=delta
                        if mag<delta and delta>0:
                            alpha*=mag/delta;accel*=mag/delta
                        contact_dir=at_contact/mag
                        angular_friction=np.cross(n,contact_dir)*alpha
                        linear_friction=-accel*contact_dir
                    if not slipping:
                        add=np.cross(radius_up,A)/(r*r)
                        if centered:
                            control=desired-(angular*left+w)
                            control*=min(1.,params['braking_acceleration']/max(np.linalg.norm(control),1e-12))
                        linear_add=-np.cross(control,-n*r)
                        at_mag=np.linalg.norm(linear_add+np.cross(add,-n*r))
                        static=float(np.float32(params['static_friction']*c['material']['friction']))
                        kinetic=float(np.float32(params['kinetic_friction']*c['material']['friction']))
                        if static*normal_force<float(np.float32(at_mag)):
                            linear_add*=kinetic*normal_force/max(float(np.float32(at_mag)),1e-12)
                        A+=linear_add;angular+=add
                    A+=linear_friction;angular+=angular_friction
                angular+=control
                if diagnostic:
                    events.append(dict(kind='diagnostic',dt_s=left,position=p.tolist(),velocity=v.tolist(),spin=w.tolist(),
                                       linear_accel=A.tolist(),angular_accel=angular.tolist(),
                                       contacts=[dict(normal=c['normal'].tolist(),point=c['point'].tolist(),
                                                      triangle_id=c['triangle_id'],material=c['material']) for c in contacts]))
                v+=A*left;w+=angular*left
                v,w=self.cancel(v,w,contacts,centered,True,s,params)
                hit=self.mesh.sweep_sphere(p,p+v*left,r,native_dt_s=left)
                if hit:
                    elapsed=left*hit['t'];p=np.asarray(hit['center'])
                    v-=A*(left-elapsed);w-=angular*(left-elapsed)
                    events.append(dict(kind='collision',normal=hit['normal'],point=hit['point'],triangle_id=hit['triangle_id']))
                    left-=elapsed
                    if elapsed<1e-9:
                        # Rounding at a previously reached edge can give a tiny
                        # positive sweep root. Resolve outside its boundary and
                        # retry all remaining time; never silently discard it.
                        p+=np.asarray(hit['normal'])*1e-8
                else:
                    p+=v*left;left=0.
                if not np.isfinite(p).all() or np.linalg.norm(v)>1000:raise ValueError('Prediction diverged')
                iterations+=1
                # Native's frame-wide counter breaks only this slice after the
                # eleventh integration. A colliding slice can lose its remainder;
                # later 8ms slices still execute one integration apiece. Advancing
                # the discarded time changes the observable position at edges.
                if iterations>10:
                    if left>1e-10:
                        events.append(dict(kind='native_solver_limit',discarded_time_s=left,
                                           position=p.tolist(),iterations=iterations))
                    break
            total-=chunk
        if move.get('trigger1'):v*=.8;w*=.8
        for i,effect in enumerate(s['effects']):
            if effect[0]:
                effect[1]=max(0,deadlines[i]-clock)
                if clock>deadlines[i]:effect[0]=False
        s['clock_ms']=clock;s['effect_end_clock_ms']=deadlines
        s['position']=p.tolist();s['velocity']=v.tolist();s['spin']=w.tolist()
        return s,events
