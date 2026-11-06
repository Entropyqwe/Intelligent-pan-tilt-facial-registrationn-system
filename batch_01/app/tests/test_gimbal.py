import json

import math

from pathlib import Path

import tempfile

import time

import unittest

from unittest.mock import patch

import numpy as np

from core import Snapshot, Observation

from gimbal_control import GimbalSteering, checked_frame, ray_to_chassis

from main import Output



ROOT=Path(__file__).resolve().parents[1]





class GimbalTests(unittest.TestCase):

    def make(self,calibrated=False):

        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)

        cfg=json.loads((ROOT/'gimbal_config.json').read_text())

        cfg.update(calibration_confirmed=calibrated,wheelbase_m=.2,front_us_per_wheel_degree=10.)

        # Mathematical cases use an explicit nominal fixture; the installed

        # hardware direction and verified travel may differ after calibration.

        cfg['front_direction']=1

        for name in ('pan','tilt'):

            cfg[name].update(direction=1,min_us=1300,max_us=1700,us_per_degree=10.)

            cfg[name].pop('deadband_deg',None)

            cfg[name].pop('tracking',None)

            cfg[name].pop('framing',None)

        path=Path(temp.name)/'g.json';path.write_text(json.dumps(cfg))

        return GimbalSteering(ROOT/'steering_config.json',path,hfov=60)



    @staticmethod

    def snap(stamp,x=280,y=80,w=80,h=400):

        return Snapshot(1,True,stamp,Observation(1,x,y,w,h,.9,.9),640,480,2)



    def feed(self,g,t,pan=1500,tilt=1500,settled=1):

        for dt in (-.25,-.05):g.accept_pose(1,1,pan,tilt,settled,0,1,20,stamp=t+dt)



    def test_centered_camera_does_not_erase_car_bearing(self):

        g=self.make();t=time.monotonic();self.feed(g,t,1700)

        g.publish_frame(self.snap(t),dict(valid=True,mm=1000,age_ms=0),1)

        f=g.fields(self.snap(t),now=t+.01)

        self.assertEqual(f['valid'],1)

        self.assertAlmostEqual(f['bearing_mdeg'],20000,delta=1)

        self.assertGreater(f['servo_us'],1500)

        self.assertFalse(g.plan['motion']) # no measured linkage yet



    def test_pan_tilt_signs_new_frames_only_and_limits(self):

        g=self.make();t=time.monotonic();self.feed(g,t)

        s=self.snap(t,x=470,y=10,h=220)

        g.publish_frame(s,{},1);g.fields(s,now=t+.01)

        first=g.plan.copy()

        self.assertGreater(first['pan_us'],1500)

        self.assertGreater(first['tilt_us'],1500) # target above => tilt upward

        for i in range(4):g.fields(s,now=t+.02+i*.04)

        self.assertEqual((g.plan['pan_us'],g.plan['tilt_us']),(first['pan_us'],first['tilt_us']))

        self.assertTrue(1300<=g.plan['pan_us']<=1700)



    def test_observed_hardware_directions_and_bench_limits(self):

        g=GimbalSteering(ROOT/'steering_gimbal_config.json',ROOT/'gimbal_config.json',hfov=60)

        self.assertFalse(g.g['calibration_confirmed'])

        self.assertEqual((g.g['pan']['direction'],g.g['tilt']['direction'],

                          g.g['front_direction'],g.sign),(-1,-1,-1,-1))

        self.assertAlmostEqual(g._angle('pan',1550),-5.)

        self.assertAlmostEqual(g._angle('tilt',1550),-5.)

        for x,y,expected in ((470,10,(-1,-1,-1)),(80,250,(1,1,1))):

            g=GimbalSteering(ROOT/'steering_gimbal_config.json',ROOT/'gimbal_config.json',hfov=60)

            t=time.monotonic();self.feed(g,t)

            s=self.snap(t,x=x,y=y,h=220)

            g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

            f=g.fields(s,now=t+.01)

            pulses=(g.plan['pan_us'],g.plan['tilt_us'],f['servo_us'])

            for value,direction,limits in zip(pulses,expected,((1100,1900),(1300,1600),(1450,1550))):

                self.assertGreater((value-1500)*direction,0)

                self.assertTrue(limits[0]<=value<=limits[1])

            self.assertFalse(g.plan['motion'])

            self.assertEqual(g.status()['reason'],'待标定 / 电机禁止')



    def test_gimbal_launcher_uses_separate_steering_profile(self):

        launcher=(ROOT/'run_gimbal_tracking.sh').read_text(encoding='utf-8')

        self.assertIn('--steering-config steering_gimbal_config.json',launcher)



    def test_current_range_and_other_axes_preserve_bench_constraints(self):

        g=GimbalSteering(ROOT/'steering_gimbal_config.json',ROOT/'gimbal_config.json')

        self.assertEqual((g.g['pan']['min_us'],g.g['pan']['max_us']),(1100,1900))

        self.assertEqual((g.g['tilt']['min_us'],g.g['tilt']['max_us']),(1300,1600))

        self.assertEqual((g.minimum,g.maximum),(1450,1550))

        self.assertFalse(g.g['calibration_confirmed'])

        self.assertIsNone(g.g['wheelbase_m'])

        self.assertIsNone(g.g['front_us_per_wheel_degree'])



    def test_pan_noise_and_small_offsets_preserve_new_tilt_composition_hold(self):

        for pixels in (-10,-1,0,1,10):

            with self.subTest(pixels=pixels):

                g=GimbalSteering(ROOT/'steering_gimbal_config.json',ROOT/'gimbal_config.json',hfov=60)

                self.assertEqual(g.g['pan']['deadband_deg'],0.)

                self.assertEqual(g.g['tilt'].get('deadband_deg',g.g['deadband_deg']),1.5)

                t=time.monotonic();self.feed(g,t)

                # Keep observed box top within the composition hold band.

                from gimbal_tilt_tracker import framing_target

                _,head_y=framing_target((0,0,80,300),480,g.g['tilt']['framing'])

                s=self.snap(t,x=280+pixels,y=head_y+pixels*.4,w=80,h=300)

                g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

                f=g.fields(s,now=t+.01)

                self.assertEqual(f['valid'],1)

                if abs(pixels)>=5:

                    self.assertGreater((g.plan['pan_us']-1500)*pixels*g.g['pan']['direction'],0)

                else:self.assertEqual(g.plan['pan_us'],1500)

                self.assertEqual(g.plan['tilt_us'],1500)

                self.assertFalse(g.plan['motion'])

                first=g.plan.copy()

                g.fields(s,now=t+.06)

                for key in ('pan_us','tilt_us','motion','valid'):

                    self.assertEqual(g.plan[key],first[key])

                self.assertGreater(g.plan['age_ms'],first['age_ms'])



    def test_per_axis_deadband_rejects_invalid_values_and_keeps_legacy_fallback(self):

        for value in (-1,float('nan'),float('inf'),True,'0',None):

            with self.subTest(value=value):

                g=self.make()

                g.g['pan']['deadband_deg']=value

                with self.assertRaises(ValueError):g._validate()

        g=self.make();t=time.monotonic();self.feed(g,t)

        s=self.snap(t,x=281,y=120,w=80,h=300)

        g.publish_frame(s,{},1);g.fields(s,now=t+.01)

        self.assertEqual(g.plan['pan_us'],1500) # old configs retain 1.5deg threshold



    def test_no_pose_or_stale_or_lost_emits_invalid_center(self):

        g=self.make();t=time.monotonic();s=self.snap(t)

        self.assertFalse(g.fields(s,now=t)['valid'])

        self.feed(g,t)

        self.assertFalse(g.fields(s,now=t+.41)['valid'])

        self.assertFalse(g.plan['motion'])

        f=g.fields(Snapshot(target_id=1),now=t+.2)

        self.assertEqual((f['valid'],f['servo_us'],g.plan['pan_us']),(0,1500,1500))



    def test_calibrated_geometry_and_motion_gates(self):

        for pan,count,mm,settled,expected in ((1500,1,1000,1,True),

                (1500,2,1000,1,False),(1500,1,1000,0,False),

                (1500,1,0,1,False),(1700,1,1000,1,True)):

            g=self.make(True);t=time.monotonic();self.feed(g,t,pan=pan,settled=settled)

            s=self.snap(t);g.publish_frame(s,dict(valid=bool(mm),mm=mm,age_ms=0),count)

            f=g.fields(s,now=t+.01)

            self.assertEqual(bool(g.plan['motion']),expected,(pan,count,mm,settled,g.info))

            if pan==1700:

                target=math.degrees(math.atan(2*.2*math.sin(math.radians(20))/1.))

                self.assertAlmostEqual(g.info['wheel_deg'],target)

                self.assertEqual(f['servo_us'],round(1500+10*target))



    def test_range_staleness_cannot_be_renewed_by_uart_heartbeat(self):

        g=self.make(True);t=time.monotonic();self.feed(g,t)

        s=self.snap(t);g.publish_frame(s,dict(valid=True,mm=1000,age_ms=299),1)

        g.fields(s,now=t+.02)

        self.assertFalse(g.plan['motion'])



    def test_misalignment_or_camera_moving_inhibits_motor(self):

        g=self.make(True);t=time.monotonic();self.feed(g,t)

        s=self.snap(t,x=480);g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

        g.fields(s,now=t+.01);self.assertFalse(g.plan['motion'])

        self.assertIn('未对准',g.info['reason'])

        g=self.make(True);g.accept_pose(1,1,1450,1500,1,0,0,0,t-.25)

        g.accept_pose(2,1,1500,1500,1,0,0,0,t-.05)

        s=self.snap(t);g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

        g.fields(s,now=t+.01);self.assertFalse(g.plan['motion'])



    def test_pose_is_not_taken_from_future(self):

        g=self.make();g.accept_pose(1,1,1500,1500,1,0,0,0,10.)

        g.accept_pose(2,1,1600,1500,1,0,0,0,10.2)

        self.assertIsNone(g.pose_at(9.9))

        self.assertAlmostEqual(g.pose_at(10.1)[0],5.)

        self.assertIsNone(g.pose_at(11.))



    def test_geometric_projection_combines_vertical_and_horizontal(self):

        r,f,u=ray_to_chassis(0,0,20,10)

        self.assertAlmostEqual(math.degrees(math.atan2(r,f)),20)

        self.assertGreater(u,0)



    def test_invalid_telemetry_cannot_create_pose(self):

        g=self.make()

        for args in ((0,1,999,1500,1,0,0,0),(0,1,1500,1500,3,0,0,0)):

            self.assertFalse(g.accept_pose(*args))

        self.assertIsNone(g.feedback)



    def test_telemetry_and_paired_serial_writer(self):

        g=self.make();t=time.monotonic();self.feed(g,t)

        s=self.snap(t);g.publish_frame(s,{},1)

        with tempfile.TemporaryDirectory() as temp:

            out=Output('',Path(temp)/'run.csv',.4,120,steering=g,print_packets=False)

            try:

                out.stop.set();out.thread.join()

                class Serial:

                    def write(self,packet):self.packet=packet;return len(packet)

                    def close(self):pass

                ser=Serial();out.serial=ser

                out._send(s)

                rows=ser.packet.splitlines()

                self.assertEqual(len(rows),2)

                self.assertTrue(rows[0].startswith(b'@G1,'))

                self.assertTrue(rows[1].startswith(b'@S1,'))

                self.assertEqual(rows[0].split(b',')[1],rows[1].split(b',')[1])

                before=len(g.poses)

                out._parse_telemetry('#G2,2,1,1520,1510,1,0,0,25')

                self.assertEqual(len(g.poses),before+1)

                out._parse_telemetry('#M1,2,1500,1500,0,0,0,STOP_AIM')

                self.assertEqual(out.motor['mode'],'STOP_AIM')

            finally:out.close()



    def test_expanded_pan_feedback_keeps_tilt_and_other_gates_bounded(self):

        for pulse in (1000,1100,1200,1250,1500,1750,1800,1900,2000):

            g=self.make()

            self.assertTrue(g.accept_pose(1,1,pulse,1500,1,0,0,0,10.))

        for pan,tilt in ((999,1500),(2001,1500),(1500,1299),(1500,1701)):

            self.assertFalse(self.make().accept_pose(1,1,pan,tilt,1,0,0,0,10.))

        g=self.make()

        g.g['pan'].update(min_us=1000,max_us=2000)

        g._validate()

        g.g['tilt']['min_us']=1299

        with self.assertRaises(ValueError):g._validate()



    def test_camera_rotation_compensation_keeps_observation_status(self):

        from types import SimpleNamespace

        from tracking_motion import BoxMotion

        g=self.make();g.accept_pose(1,1,1500,1500,1,0,0,0,10.)

        track=SimpleNamespace(box=(280,80,80,300),motion=BoxMotion((280,80,80,300)),control_ready=False)

        ids=SimpleNamespace(tracks={1:track});single=SimpleNamespace(last_box=track.box)

        g.compensate_tracks(ids,single,Snapshot(captured=10.,image_width=640,image_height=480))

        g.accept_pose(2,1,1600,1500,1,0,0,0,10.2)

        g.compensate_tracks(ids,single,Snapshot(captured=10.2,image_width=640,image_height=480))

        self.assertLess(track.box[0],280) # pan right => stationary target shifts left

        self.assertFalse(track.control_ready)



    def test_installed_reverse_pan_has_consistent_bearing_and_association(self):

        from types import SimpleNamespace

        from tracking_motion import BoxMotion

        g=GimbalSteering(ROOT/'steering_gimbal_config.json',ROOT/'gimbal_config.json',hfov=60)

        t=time.monotonic();self.feed(g,t,pan=1450)

        s=self.snap(t)

        g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

        f=g.fields(s,now=t+.01)

        self.assertEqual(f['valid'],1)

        self.assertAlmostEqual(f['bearing_mdeg'],5000,delta=1)

        self.assertEqual(g.plan['pan_us'],1450) # centered image preserves pan pose

        self.assertGreater(g.plan['tilt_us'],1500) # composition independent of bearing

        self.assertLess(f['servo_us'],1500) # unchanged front mapping: chassis right

        self.assertFalse(g.plan['motion'])

        track=SimpleNamespace(box=(280,80,80,300),motion=BoxMotion((280,80,80,300)),control_ready=False)

        ids=SimpleNamespace(tracks={1:track});single=SimpleNamespace(last_box=track.box)

        g.compensate_tracks(ids,single,Snapshot(captured=t,image_width=640,image_height=480))

        g.accept_pose(3,1,1550,1500,1,0,0,0,t+.2)

        g.compensate_tracks(ids,single,Snapshot(captured=t+.2,image_width=640,image_height=480))

        self.assertGreater(track.box[0],280) # pan left => stationary target shifts right

        self.assertFalse(track.control_ready)



    def test_asymmetric_slopes_round_trip_preserves_physical_bearing(self):

        g=self.make()

        g.g['pan'].update(us_per_degree_negative=6.,us_per_degree_positive=12.)

        for direction in (-1,1):

            g.g['pan']['direction']=direction

            for angle in (-12.,-1.,0.,1.,12.):

                self.assertAlmostEqual(g._angle('pan',g._pulse('pan',angle)),angle)

        g.g['pan'].pop('us_per_degree_positive')

        with self.assertRaises(ValueError):g._validate()



    def test_desktop_gimbal_entries_and_full_command(self):

        import desktop_control as d

        self.assertIn('gimbal_full',d.LABELS)

        self.assertEqual(d.command_for('gimbal_full'),['bash',str(d.ROOT/'run_gimbal_tracking.sh')])

        with patch('desktop_control.pick_port',return_value='/dev/ttyUSB0'):

            for axis in ('pan','tilt','front'):

                cmd=d.command_for('gimbal_'+axis)

                self.assertEqual(cmd[-1],axis)

                self.assertIn('tools/gimbal_check.py',cmd)



    def test_new_range_after_settling_is_required(self):

        g=self.make(True);t=time.monotonic();self.feed(g,t)

        g.feedback['range_ready']=False

        s=self.snap(t);g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

        g.fields(s,now=t+.01)

        self.assertFalse(g.plan['motion'])

        self.assertIn('新距离',g.info['reason'])



    def test_high_tilt_uses_horizontal_ranger_azimuth(self):

        g=self.make(True);t=time.monotonic()

        self.feed(g,t,tilt=1700)

        # Raw horizontal image error ~3.9deg, but horizontal chassis bearing

        # is larger when looking upward. A pan-only ranger must stay inhibited.

        s=self.snap(t,x=318)

        g.publish_frame(s,dict(valid=True,mm=1000,age_ms=0),1)

        g.fields(s,now=t+.01)

        self.assertLess(abs(g.info['error_x']),4)

        self.assertGreater(abs(g.info['bearing_deg']-g.info['pan_deg']),4)

        self.assertFalse(g.plan['motion'])



if __name__=='__main__':unittest.main()

