import unittest
from unittest.mock import patch,Mock
from main import LatestCamera


class StartupGuardTests(unittest.TestCase):
    def test_unusable_saved_image_falls_back_before_capture_thread(self):
        cap=Mock();cap.isOpened.return_value=True
        cap.get.side_effect=lambda prop: {3:640,4:480,5:30}.get(prop,0)
        camera=LatestCamera.__new__(LatestCamera)
        camera.index=0;camera.backend=0;camera.width=640;camera.height=480
        camera.fps=30;camera.frame=None;camera.camera_profile='saved-fixed';camera.camera_warning=''
        with patch('main.cv2.VideoCapture',return_value=cap),\
             patch('camera_profiles.apply_profile') as apply,\
             patch('camera_exposure.load_profile',return_value=dict(test=dict(auto_baseline={}))),\
             patch('exposure_quality.verify_capture',return_value=('噪点过强',{})):
            self.assertIs(camera._open_capture(),cap)
        self.assertEqual(camera.camera_profile,'quality')
        self.assertIn('已恢复自动',camera.camera_warning)
        self.assertEqual([call.args[1] for call in apply.call_args_list],['saved-fixed','quality'])


if __name__=='__main__':unittest.main()
