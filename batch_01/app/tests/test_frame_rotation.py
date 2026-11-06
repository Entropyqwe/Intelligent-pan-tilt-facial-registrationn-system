import threading
import unittest
from unittest.mock import patch
import numpy as np
from main import LatestCamera
import desktop_control as d


class RotationTests(unittest.TestCase):
    def camera(self,rotation):
        c=LatestCamera.__new__(LatestCamera)
        c.mutex=threading.Lock();c.frame_rotation=rotation
        c.frame=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        c.timestamp=12.5;c.seq=17
        return c

    def test_rotation_is_before_consumers_and_retains_capture_metadata(self):
        c=self.camera(180);raw=c.frame.copy()
        view,stamp,seq=c.latest(16)
        np.testing.assert_array_equal(view,raw[::-1,::-1,:])
        np.testing.assert_array_equal(c.frame,raw)
        self.assertEqual((stamp,seq),(12.5,17))
        second,_,_=c.latest(16)
        np.testing.assert_array_equal(second,view)  # no double rotation
        self.assertIsNone(c.latest(17)[0])

    def test_unrotated_mode_is_explicit(self):
        c=self.camera(0);view,_,_=c.latest()
        np.testing.assert_array_equal(view,c.frame)
        self.assertFalse(np.shares_memory(view,c.frame))

    def test_gimbal_fixed_profile_keeps_gimbal_launcher(self):
        with patch('camera_exposure.load_profile') as load:
            cmd=d.command_for('gimbal_fixed')
            load.assert_called_once_with(0)
        self.assertEqual(cmd,['bash',str(d.ROOT/'run_gimbal_tracking.sh'),
                              '--camera-profile','saved-fixed'])

    def test_missing_fixed_profile_does_not_fall_into_legacy_control(self):
        with patch('camera_exposure.load_profile',side_effect=RuntimeError('missing')):
            with self.assertRaises(RuntimeError):d.command_for('gimbal_fixed')


if __name__=='__main__':unittest.main()
