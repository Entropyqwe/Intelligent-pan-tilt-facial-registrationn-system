import json
import math
from pathlib import Path
import tempfile
import unittest
from core import Snapshot, Observation
from steering import Steering, encode_s1

ROOT=Path(__file__).resolve().parents[1]

class SteeringTests(unittest.TestCase):
    def make(self, **overrides):
        c=json.loads((ROOT/'steering_config.json').read_text(encoding='utf-8'))
        c.update(horizontal_fov_deg=60,filter_tau_seconds=0,max_slew_us_per_second=100000)
        c.update(overrides)
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        p=Path(t.name)/'cfg.json'; p.write_text(json.dumps(c),encoding='utf-8')
        return Steering(p)

    def snap(self,u=320,stamp=10,source=1,target=1):
        return Snapshot(target,True,stamp,Observation(target,u-20,100,40,160,.9,.8),640,480,source,stamp)

    def test_geometric_bearing_center_edges_and_nonlinear(self):
        c=self.make()
        for u,wanted in ((0,-30),(320,0),(640,30)):
            self.assertAlmostEqual(c.bearing(self.snap(u)),wanted,places=7)
        self.assertAlmostEqual(c.bearing(self.snap(480)),math.degrees(math.atan(.5*math.tan(math.radians(30)))))

    def test_direction_clamp_and_deadband(self):
        c=self.make()
        self.assertEqual(c.fields(self.snap(),now=10)['servo_us'],1500)
        self.assertEqual(c.fields(self.snap(640),now=10.1)['servo_us'],1700)
        self.assertEqual(c.fields(self.snap(0),now=10.2)['servo_us'],1300)
        c=self.make(servo_direction=-1)
        self.assertEqual(c.fields(self.snap(640),now=10)['servo_us'],1300)

    def test_stale_loss_future_timestamp_and_source(self):
        c=self.make()
        for snap,now in ((self.snap(),10.401),(Snapshot(target_id=1),10),(self.snap(stamp=11),10)):
            f=c.fields(snap,now=now)
            self.assertEqual((f['valid'],f['source'],f['servo_us'],f['bearing_mdeg']),(0,0,1500,0))
        self.assertEqual(c.fields(self.snap(source=2),now=10)['source'],2)

    def test_rate_limit_and_reset_on_loss(self):
        c=self.make(max_slew_us_per_second=100)
        self.assertEqual(c.fields(self.snap(640),now=10)['servo_us'],1510)
        self.assertEqual(c.fields(self.snap(640),now=10.1)['servo_us'],1520)
        self.assertEqual(c.fields(Snapshot(),now=10.2)['servo_us'],1500)
        self.assertEqual(c.fields(self.snap(0,stamp=10.2),now=10.2)['servo_us'],1490)

    def test_fail_closed_without_fov_or_with_wrong_neutral(self):
        for args in (dict(horizontal_fov_deg=None),dict(servo_center_us=1490),dict(horizontal_fov_deg=float('nan'))):
            with self.assertRaises(ValueError): self.make(**args)

    def test_same_slot_short_loss_retains_filter_memory_but_wire_invalid(self):
        c=self.make(max_slew_us_per_second=500)
        first=c.fields(self.snap(640),now=10)['servo_us']
        self.assertGreater(first,1500)
        loss=c.fields(Snapshot(target_id=1),now=10.1)
        self.assertEqual((loss['valid'],loss['servo_us'],loss['source']),(0,1500,0))
        self.assertEqual(c.current,first)
        recovered=c.fields(self.snap(640,stamp=10.2),now=10.2)
        self.assertGreater(recovered['servo_us'],first)
        c.fields(Snapshot(target_id=1),now=13.2)
        self.assertEqual(c.current,1500)

    def test_calibrated_ray_distortion_and_aspect(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        p=Path(t.name)
        calib=dict(camera_matrix=[[500,0,310],[0,500,240],[0,0,1]],
                   dist_coeffs=[0,0,0,0,0],image_width=640,image_height=480)
        (p/'cal.json').write_text(json.dumps(calib),encoding='utf-8')
        c=self.make(camera_calibration_file=str(p/'cal.json'))
        self.assertAlmostEqual(c.bearing(self.snap(310)),0)
        self.assertAlmostEqual(c.bearing(self.snap(560)),math.degrees(math.atan(.5)))
        s=self.snap(); s.image_width=800
        with self.assertRaises(ValueError): c.bearing(s)

    def test_reacquisition_opposite_side_uses_current_ray_without_filter_lag(self):
        c=self.make(filter_tau_seconds=2,max_slew_us_per_second=100)
        c.fields(self.snap(640),now=10)
        c.fields(Snapshot(target_id=1),now=10.1)
        current=self.snap(160,stamp=10.3)
        recovered=c.fields(current,now=10.3)
        expected=round(max(c.minimum,min(c.maximum,c.center+c.sign*c.gain*c.bearing(current))))
        self.assertEqual(recovered['servo_us'],expected)
        self.assertLess(expected,1500)
        self.assertAlmostEqual(c.filtered,c.bearing(current))
        # Ordinary subsequent tracking still respects the configured rate.
        next_fields=c.fields(self.snap(640,stamp=10.4),now=10.4)
        self.assertLessEqual(abs(next_fields['servo_us']-expected),10)

    def test_reacquisition_center_and_clamps_preserve_invalid_neutral(self):
        for u,expected in ((0,1300),(320,1500),(640,1700)):
            c=self.make(filter_tau_seconds=2,max_slew_us_per_second=100)
            c.fields(self.snap(640),now=10)
            invalid=c.fields(Snapshot(target_id=1),now=10.1)
            self.assertEqual((invalid['valid'],invalid['source'],invalid['servo_us']),(0,0,1500))
            self.assertEqual(c.fields(self.snap(u,stamp=10.3),now=10.3)['servo_us'],expected)

    def test_missed_invalid_publication_detected_from_capture_gap(self):
        c=self.make(filter_tau_seconds=2,max_slew_us_per_second=100)
        c.fields(self.snap(640),now=10)
        self.assertEqual(c.fields(self.snap(0,stamp=10.6),now=10.6)['servo_us'],1300)

    def test_long_loss_or_changed_target_still_starts_with_rate_limit(self):
        for target,stamp in ((1,13.1),(2,10.3)):
            c=self.make(max_slew_us_per_second=100)
            c.fields(self.snap(640),now=10)
            c.fields(Snapshot(target_id=target),now=stamp-.01)
            fields=c.fields(self.snap(0,stamp=stamp,target=target),now=stamp)
            self.assertEqual(fields['servo_us'],1490)

    def test_single_reentry_to_serial_fields_end_to_end(self):
        from single_person import SinglePersonPolicy
        from body_identity import BodySelection
        policy=SinglePersonPolicy();selector=BodySelection()
        c=self.make(filter_tau_seconds=.5,max_slew_us_per_second=100)
        def process(x,stamp):
            bodies=[] if x is None else [dict(box=(x,60,100,330),score=.9)]
            tracks=policy.apply([],bodies,stamp,640,480)
            policy.auto_select(selector)
            return c.fields(selector.update(tracks,stamp,640,480),now=stamp)
        for i in range(6):first=process(430,10+i*.1)
        self.assertEqual(first['valid'],1)
        self.assertEqual(process(None,10.6)['valid'],0)
        self.assertEqual(process(80,11.)['valid'],0)
        recovered=process(80,11.1)
        self.assertEqual((recovered['valid'],recovered['target_id'],recovered['source']),(1,1,2))
        self.assertEqual(recovered['servo_us'],1300)
        self.assertIn(b',1,1,',encode_s1(10,recovered))

    def test_protocol_checksum_known_vector(self):
        f=self.make().fields(self.snap(640),now=10)
        packet=encode_s1(1,f)
        self.assertTrue(packet.startswith(b'@S1,1,1,1,30000,1700,0,1*'))
        body,check=packet[1:].strip().split(b'*')
        xor=0
        for b in body: xor^=b
        self.assertEqual(check.decode(),f'{xor:02X}')

if __name__=='__main__': unittest.main()
