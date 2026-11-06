import unittest
from unittest.mock import patch
from camera_profiles import apply_profile,PROFILES


class ProfileTests(unittest.TestCase):
    def test_other_camera_is_not_modified(self):
        with patch('camera_profiles.sys.platform','linux'),\
             patch('camera_profiles.usb_identity',return_value='1234:5678'),\
             patch('camera_profiles.subprocess.run') as run:
            self.assertFalse(apply_profile(0,'quality'))
            run.assert_not_called()

    def test_exposure_mode_is_set_before_manual_value(self):
        with patch('camera_profiles.sys.platform','linux'),\
             patch('camera_profiles.usb_identity',return_value='0c45:64ab'),\
             patch('camera_profiles.subprocess.run') as run:
            self.assertTrue(apply_profile(0,'fast-experimental'))
            commands=[call.args[0][-1] for call in run.call_args_list]
            self.assertLess(commands.index('--set-ctrl=auto_exposure=1'),
                            commands.index('--set-ctrl=exposure_time_absolute=10'))
            self.assertEqual(len(commands),len(PROFILES['fast-experimental']))

    def test_unchanged_never_calls_device_tools(self):
        with patch('camera_profiles.usb_identity') as identity:
            self.assertFalse(apply_profile(0,'unchanged'))
            identity.assert_not_called()


if __name__=='__main__':unittest.main()
