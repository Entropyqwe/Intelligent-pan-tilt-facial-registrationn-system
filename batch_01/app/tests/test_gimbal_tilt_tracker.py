import copy
import json
from pathlib import Path
import tempfile
import unittest

from core import Observation, Snapshot
from gimbal_control import GimbalSteering
from gimbal_tilt_tracker import TiltFramer, framing_target, validate_framing


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'gimbal_config.json').read_text(encoding='utf-8'))


class TiltTests(unittest.TestCase):
    def framer(self):
        return TiltFramer(CFG['tilt']['framing'])

    def update(self, tracker, stamp=10., target=1, error_px=60., error_deg=6., applied=1500):
        return tracker.update(captured=stamp, target_id=target,
            error_deg=error_deg, error_px=error_px, capture_tilt_deg=(1500-applied)/10.,
            applied_us=applied, to_angle=lambda p:(1500-p)/10.,
            to_pulse=lambda a:1500-10*a, minimum=1300, maximum=1600)

    def test_large_box_reserves_headroom_instead_of_centering_whole_body(self):
        observed, desired = framing_target((200, 0, 180, 480), 480, CFG['tilt']['framing'])
        self.assertEqual(observed, 0)
        self.assertAlmostEqual(desired, 48.)
        # Crop at the top requires the camera to point upward.
        self.assertLess(observed-desired, 0)

    def test_small_person_places_estimated_upper_body_near_center(self):
        c=CFG['tilt']['framing']
        _, desired=framing_target((200, 80, 100, 240), 480, c)
        self.assertAlmostEqual(desired+.5*.6*240, .42*480)

    def test_framing_is_resolution_independent_and_continuous(self):
        c=CFG['tilt']['framing']
        first=framing_target((20, 40, 150, 300), 480, c)
        second=framing_target((40, 80, 300, 600), 960, c)
        for a,b in zip(first,second):self.assertAlmostEqual(a*2,b)
        values=[framing_target((20,20,100,h),480,c)[1] for h in range(100,460)]
        self.assertLess(max(abs(b-a) for a,b in zip(values,values[1:])), 2.)

    def test_bottom_truncated_box_uses_visible_extent_without_inventing_waist(self):
        c=CFG['tilt']['framing']
        self.assertEqual(framing_target((10, 80, 100, 700),480,c),
                         framing_target((10, 80, 100, 400),480,c))

    def test_direction_and_first_step_bounded(self):
        for error in (-6.,6.):
            pulse,info=self.update(self.framer(),error_px=error*10,error_deg=error)
            self.assertGreater((pulse-1500)*error,0) # installed tilt direction -1
            self.assertLessEqual(abs(pulse-1500),3)
            self.assertEqual(info['tilt_state'],'TRACK')

    def test_hysteresis_holds_centered_jitter(self):
        f=self.framer()
        pulse,info=self.update(f,error_px=4,error_deg=.4)
        self.assertEqual((pulse,info['tilt_state']),(1500,'HOLD'))
        pulse,info=self.update(f,stamp=10.1,error_px=8,error_deg=.8)
        self.assertEqual((pulse,info['tilt_state']),(1500,'HOLD'))
        pulse,info=self.update(f,stamp=10.2,error_px=12,error_deg=1.2)
        self.assertEqual(info['tilt_state'],'TRACK')
        _,info=self.update(f,stamp=10.3,error_px=4,error_deg=.4,applied=pulse)
        self.assertEqual(info['tilt_state'],'HOLD')

    def test_duplicate_capture_does_not_accumulate(self):
        f=self.framer();first=self.update(f)
        for _ in range(5):self.assertEqual(self.update(f,applied=first[0]),first)

    def test_rate_and_travel_limits_are_bounded(self):
        f=self.framer();pulse=1500
        for i in range(120):
            old=pulse;pulse,info=self.update(f,stamp=10+i*.1,error_px=100,error_deg=25,applied=pulse)
            self.assertTrue(1300<=pulse<=1600)
            self.assertLessEqual(abs(pulse-old),20)
        self.assertEqual(pulse,1600);self.assertTrue(info['tilt_at_limit'])
        self.assertEqual(f.command_rate,0.)

    def test_target_change_gap_and_loss_clear_state(self):
        f=self.framer();self.update(f);self.update(f,stamp=10.1)
        pulse,_=self.update(f,stamp=10.2,target=2,error_px=0,error_deg=0,applied=1490)
        self.assertEqual(pulse,1490)
        pulse,_=self.update(f,stamp=11.,target=2,error_px=0,error_deg=0,applied=1480)
        self.assertEqual(pulse,1480)
        f.reset();self.assertIsNone(f.stamp);self.assertEqual(f.command_rate,0)

    def test_invalid_configuration_rejected(self):
        for key,value in (('kp',float('nan')),('hold_px',True),('resume_px',5),
                          ('large_box_full_fraction',.5),('max_rate_deg_s',80)):
            c=copy.deepcopy(CFG['tilt']['framing']);c[key]=value
            with self.assertRaises(ValueError):validate_framing(c)

    def steering(self,framing=True):
        c=copy.deepcopy(CFG)
        if not framing:c['tilt'].pop('framing')
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'g.json';path.write_text(json.dumps(c),encoding='utf-8')
        return GimbalSteering(ROOT/'steering_gimbal_config.json',path,hfov=60)

    def test_s1_bearing_front_pan_and_motion_are_identical_for_same_observation(self):
        active,legacy=self.steering(),self.steering(False)
        for g in (active,legacy):
            g.accept_pose(1,1,1490,1490,1,0,1,0,9.8)
            g.accept_pose(2,1,1490,1490,1,0,1,0,9.95)
        s=Snapshot(1,True,10.,Observation(1,360,30,120,400,.9,.9),640,480,2)
        for g in (active,legacy):g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)
        self.assertEqual(active.fields(s,now=10.01),legacy.fields(s,now=10.01))
        self.assertEqual(active.plan['pan_us'],legacy.plan['pan_us'])
        self.assertEqual(active.plan['motion'],0)
        self.assertEqual(active.plan['motion'],legacy.plan['motion'])

    def test_invalid_visual_or_stale_pose_resets_composition(self):
        g=self.steering();g.accept_pose(1,1,1500,1500,1,0,1,0,9.95)
        s=Snapshot(1,True,10.,Observation(1,280,0,80,480,.9,.9),640,480,2)
        g.publish_frame(s,{},1);g.fields(s,now=10.01)
        self.assertIsNotNone(g.tilt_framer.stamp)
        g.fields(Snapshot(target_id=1),now=10.1)
        self.assertIsNone(g.tilt_framer.stamp)
        self.assertEqual(g.plan['valid'],0)
        s.captured=10.1
        g.fields(s,now=10.11);self.assertIsNotNone(g.tilt_framer.stamp)
        g.fields(s,now=10.31);self.assertIsNone(g.tilt_framer.stamp)


if __name__=='__main__':unittest.main()
