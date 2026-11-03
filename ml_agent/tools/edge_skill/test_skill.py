"""Mechanism, terminal-state and native-controller conformance checks."""
import copy
import unittest

import numpy as np

from nav.native.forecast import predict_many
from .audit import contact_kind,audit_native_contract,audit_forecast_contract
from .common import NIGHT,load_json,engine_for
from .maps import transform
from .execution import Policy,initial,local,inside,assess,make_case,seeds,resume_static
from .compiled import predict
from nav.joystick import action_to_joystick
from nav.marble_dynamics.bridge import reply,translated


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.cell=dict(origin=[0,0,0],yaw_deg=0,lip_z=.9,width=8,landing_z=1.5,
                       landing_center=[10,0],landing_size=[20,6],landing_yaw=0)
        self.case=dict(cell=self.cell,goal=[5,0,1.689750001],exit_velocity=[6,0],goal_radius=.9,
                       start_floor=0,airborne=False,start_kind='ground')

    def test_contact_mechanism_and_wall_assist(self):
        c=self.case;z=self.cell['lip_z']
        self.assertEqual(contact_kind(c,[0,0,z],[-.6,0,.8]),'lip')
        self.assertEqual(contact_kind(c,[0,0,z],[0,0,1]),'other')
        self.assertEqual(contact_kind(c,[0,0,z],[-.6,0,-.8]),'other')
        self.assertEqual(contact_kind(c,[0,20,z],[-.6,0,.8]),'other')
        def transition(point,normal,entry,after):
            return dict(substep_trace=[dict(contact_entry_velocity=entry,pre_integration=dict(velocity=after),
                                            contacts=[dict(position=point,normal=normal)])])
        lip=transition([0,0,z],[-.6,0,.8],[10,0,0],[8,0,5])
        wall=transition([4,0,z],[-1,0,0],[10,0,0],[-4,0,0])
        self.assertTrue(audit_native_contract(c,[lip])['valid'])
        self.assertFalse(audit_native_contract(c,[wall,lip])['valid'])
        self.assertFalse(audit_native_contract(c,[lip,wall])['valid'])
        no_kick=transition([0,0,z],[-.6,0,.8],[10,0,0],[10,0,0])
        self.assertFalse(audit_native_contract(c,[no_kick])['valid'])

    def test_terminal_contract(self):
        states=np.zeros((100,10));states[:,0]=5+6*np.arange(100)*.016;states[:,2]=self.case['goal'][2]
        states[:,3]=6;states[:,9]=.189750001;contact=dict(valid=True,tick=1)
        self.assertTrue(assess(self.case,states,contact=contact)['success'])
        self.assertFalse(assess(self.case,states)['success'])
        self.assertFalse(assess(self.case,states[:50],contact=contact)['success'])
        wrong_speed=states.copy();wrong_speed[:,3]=15
        self.assertFalse(assess(self.case,wrong_speed,contact=contact)['success'])
        falling=states.copy();falling[20:,2]-=.4
        self.assertFalse(assess(self.case,falling,contact=contact)['success'])
        outside=states.copy();outside[:,1]=6
        self.assertFalse(assess(self.case,outside,contact=contact)['success'])
        mega=states.copy();mega[:,9]*=2
        self.assertFalse(assess(self.case,mega,contact=contact)['success'])
        powered=np.pad(states,((0,0),(0,15)));powered[20:30,19]=1
        self.assertFalse(assess(self.case,powered,contact=contact)['success'])

    def test_rotation_and_platform_bounds(self):
        point=[-2,1,.7];c=copy.deepcopy(self.cell);c['origin']=[100,-40,8];c['yaw_deg']=37
        self.assertTrue(np.allclose(local(c,transform(c,point)),point))
        c['landing_yaw']=.71;theta=c['landing_yaw'];u=np.array([np.cos(theta),np.sin(theta)])
        self.assertTrue(inside(c,[*(np.array(c['landing_center'])+u*9.),1.5]))
        self.assertFalse(inside(c,[*(np.array(c['landing_center'])+u*10.),1.5]))

    def test_stable_input_encoding(self):
        import math
        for angle in np.linspace(-math.pi,math.pi,1001):
            for throttle in (0.,.03,.12,.5,.74,.99,1.):
                encoded=min(1.0002,(math.floor(throttle*16)+.5)/16)
                action=reply(action_to_joystick(math.cos(angle),math.sin(angle),encoded))
                move,_,_,_=translated(action,0,0.)
                expected=math.floor(throttle*16)/16
                self.assertEqual(move['x'],expected);self.assertEqual(move['y'],expected)

    def test_sloped_contact_and_supported_exit(self):
        c=copy.deepcopy(self.case);c['cell']['lip_slope_y']=.1;c['cell']['landing_slope']=[.15,.05]
        self.assertEqual(contact_kind(c,[0,2,1.1],[-.6,-.08,.8]),'lip')
        self.assertEqual(contact_kind(c,[0,2,.9],[-.6,-.08,.8]),'other')
        states=np.zeros((100,10));states[:,0]=5+6*np.arange(100)*.016;states[:,2]=c['goal'][2]+.15*(states[:,0]-5)
        states[:,3]=6;states[:,5]=.9;states[:,9]=.189750001
        self.assertTrue(assess(c,states,contact=dict(valid=True,tick=1))['success'])
        states[:,5]=0
        self.assertFalse(assess(c,states,contact=dict(valid=True,tick=1))['success'])

    def test_compiled_policy_and_queued_continuation(self):
        manifest=load_json(NIGHT/'maps.json');rng=np.random.default_rng(3321)
        # Only training geometry; final test maps remain sealed.
        entries=manifest['maps'][:2]
        if (NIGHT/'sloped_geometry_conformance.json').exists():entries+=load_json(NIGHT/'sloped_maps.json')['maps'][:2]
        for entry in entries:
            e=engine_for(entry['name'])
            for index in (1,5,9,13):
                c=make_case(entry['cells'][index],index,3312+index,start_kind='ground' if index%3 else 'air_rising')
                c['stable_inputs']=index%2==1
                d=initial(c);pp=[seeds(c)[i] for i in rng.choice(len(seeds(c)),6,replace=False)]
                fast,_=predict(e,c,d,pp,steps=96);slow=predict_many(e,d,[Policy(c,p,steps=96,engine=e,d=d) for p in pp])
                for a,b in zip(fast,slow):
                    self.assertTrue(np.array_equal(a['proposed'],b['proposed']))
                    self.assertTrue(np.array_equal(a['native_states'].rows,b['native_states'].rows))
                    self.assertTrue(all(not x[6] and not x[8] for x in a['proposed']))
                for a in fast[:2]:
                    self.assertEqual(audit_forecast_contract(e,c,d,a),audit_forecast_contract(e,c,d,a,compiled=False))
                dd=resume_static(d,fast[0],5)
                continuation=predict_many(e,dd,[fast[0]['proposed'][5:]])[0]
                self.assertTrue(np.array_equal(continuation['native_states'].rows,fast[0]['native_states'].rows[20:]))


if __name__=='__main__':unittest.main()
