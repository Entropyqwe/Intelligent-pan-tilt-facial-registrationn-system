import json
import math
from pathlib import Path
import unittest
from gimbal_pan_tracker import PanTracker,validate_tracking

ROOT=Path(__file__).resolve().parents[1]


class PanControlTests(unittest.TestCase):
    def make(self):
        cfg=json.loads((ROOT/'gimbal_config.json').read_text())['pan']['tracking']
        return PanTracker(cfg)

    def update(self,controller,error_px,*,t=10.,pulse=1500,target=1,minimum=1350,maximum=1650):
        error=math.degrees(math.atan(error_px/554.2562584))
        return controller.update(captured=t,target_id=target,now=t+.04,
            error_deg=error,error_px=error_px,capture_pan_deg=(1500-pulse)/10,
            applied_us=pulse,to_angle=lambda p:(1500-p)/10,
            to_pulse=lambda a:1500-10*a,minimum=minimum,maximum=maximum)

    def test_new_range_continues_past_old_stops_on_both_sides_and_exits_limits(self):
        histories=[]
        for sign in (-1,1):
            c=self.make();pulse=1500;history=[]
            for i,x in enumerate([120]*15+[-120]*25+[0,1,-2,3,0]):
                pulse,info=self.update(c,sign*x,t=10+i*.1,pulse=pulse,minimum=1300,maximum=1700)
                history.append((pulse,info))
            self.assertEqual(history[14][0],1700 if sign<0 else 1300)
            self.assertGreater((history[15][0]-history[14][0])*sign,0)
            histories.append(history)
        for (a,ai),(b,bi) in zip(*histories):
            self.assertEqual(a+b,3000)
            self.assertEqual(ai['pan_state'],bi['pan_state'])
            self.assertEqual(ai['pan_at_limit'],bi['pan_at_limit'])
            self.assertEqual(ai['pan_command_us'],a)
            self.assertEqual((ai['pan_min_us'],ai['pan_max_us']),(1300,1700))
        self.assertEqual(histories[0][14][1]['pan_limit_side'],'MAX')
        self.assertEqual(histories[1][14][1]['pan_limit_side'],'MIN')

    def test_supported_pan_ranges_are_symmetric_and_release_both_limits(self):
        # Mathematical behavior only; this does not approve physical travel.
        for minimum,maximum in ((1250,1750),(1200,1800),(1100,1900),(1000,2000)):
            with self.subTest(minimum=minimum,maximum=maximum):
                histories=[]
                for sign in (-1,1):
                    c=self.make();pulse=1500;history=[]
                    for i,x in enumerate([120]*20+[-120]*40):
                        pulse,info=self.update(c,sign*x,t=10+i*.1,pulse=pulse,
                                               minimum=minimum,maximum=maximum)
                        self.assertTrue(minimum<=pulse<=maximum)
                        history.append(pulse)
                    self.assertEqual(history[19],maximum if sign<0 else minimum)
                    self.assertGreater((history[20]-history[19])*sign,0)
                    self.assertEqual(history[-1],minimum if sign<0 else maximum)
                    histories.append(history)
                self.assertTrue(all(a+b==3000 for a,b in zip(*histories)))

    def test_static_center_jitter_never_becomes_pan_motion(self):
        c=self.make()
        for i,x in enumerate([0,1,-2,3,-3,1,0]*15):
            pulse,info=self.update(c,x,t=10+i*.125)
            self.assertEqual(pulse,1500)
            self.assertEqual(info['pan_state'],'HOLD')

    def test_hysteresis_resumes_without_old_large_dead_zone(self):
        c=self.make()
        self.update(c,0)
        pulse,info=self.update(c,7,t=10.125)
        self.assertLess(pulse,1500)
        self.assertEqual(info['pan_state'],'TRACK')
        pulse,info=self.update(c,4,t=10.25,pulse=pulse)
        self.assertEqual(info['pan_state'],'TRACK')
        held,info=self.update(c,2,t=10.375,pulse=pulse)
        self.assertEqual(held,pulse)
        self.assertEqual(info['pan_state'],'HOLD')

    def test_fast_motion_obeys_rate_and_travel_limits_without_windup(self):
        c=self.make();pulse=1500
        for i in range(20):
            prev=pulse
            pulse,info=self.update(c,160,t=10+i*.125,pulse=pulse)
            self.assertLessEqual(abs(pulse-prev),51)
            self.assertGreaterEqual(pulse,1350)
        self.assertEqual(pulse,1350)
        self.assertTrue(info['pan_at_limit'])
        returned,info=self.update(c,-100,t=12.5,pulse=pulse)
        self.assertGreater(returned,pulse)

    def test_duplicate_frame_cannot_integrate_or_predict_twice(self):
        c=self.make();pulse,_=self.update(c,120)
        repeated,_=self.update(c,120,pulse=pulse)
        self.assertEqual(repeated,pulse)

    def test_target_switch_gap_and_loss_reset_prediction(self):
        c=self.make()
        self.update(c,30)
        self.update(c,60,t=10.1)
        self.assertGreater(c.target_rate,0)
        _,info=self.update(c,0,t=10.2,target=2)
        self.assertEqual(info['pan_state'],'HOLD')
        self.assertEqual(info['pan_target_rate_deg_s'],0)
        self.update(c,60,t=10.3,target=2)
        _,info=self.update(c,0,t=11,target=2)
        self.assertEqual(info['pan_target_rate_deg_s'],0)
        c.reset()
        _,info=self.update(c,-30,t=12,target=2)
        self.assertEqual(info['pan_lead_deg'],0)

    def test_tracking_config_rejects_nan_bool_and_inverted_hysteresis(self):
        cfg=self.make().c
        for key,value in (('kp_near',True),('filter_tau_s',float('nan')),
                          ('max_rate_deg_s',100),('resume_px',0)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_tracking(dict(cfg,**{key:value}))
        with self.assertRaises(ValueError):validate_tracking(dict(cfg,hold_px=6,resume_px=5))

    def test_closed_loop_step_converges_inside_center_band(self):
        c=self.make();pulse=1500
        for i in range(50):
            target=8.
            pose=(1500-pulse)/10
            error=target-pose
            px=554.2562584*math.tan(math.radians(error))
            pulse,info=self.update(c,px,t=10+i*.125,pulse=pulse)
        self.assertLess(abs(error),.8)
        self.assertEqual(info['pan_state'],'HOLD')

    def test_prediction_is_bounded_and_never_used_while_holding(self):
        c=self.make()
        for i in range(20):
            _,info=self.update(c,20+i*20,t=10+i*.05)
            self.assertLessEqual(abs(info['pan_lead_deg']),1.5)
        _,info=self.update(c,0,t=11)
        self.assertEqual(info['pan_lead_deg'],0)


if __name__=='__main__':unittest.main()
