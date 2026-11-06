import unittest
from unittest.mock import patch
import desktop_control as d


class LauncherTests(unittest.TestCase):
    def test_default_and_fixed_vision_are_explicit_and_never_open_serial(self):
        with patch('camera_exposure.load_profile') as load:
            for action,profile in (('vision','quality'),('fixed_vision','saved-fixed')):
                cmd=d.command_for(action)
                self.assertEqual(cmd[-2:],['--camera-profile',profile])
                self.assertNotIn('--port',cmd)
            load.assert_called_once_with(0)

    def test_fixed_full_uses_existing_full_launcher(self):
        with patch('camera_exposure.load_profile'):
            cmd=d.command_for('fixed_full')
        self.assertEqual(cmd,['bash',str(d.ROOT/'run_human_tracking.sh'),'--camera-profile','saved-fixed'])

    def test_no_saved_parameters_blocks_fixed_launch(self):
        with patch('camera_exposure.load_profile',side_effect=RuntimeError('missing')):
            with self.assertRaises(RuntimeError):d.command_for('fixed_full')

    def test_tuner_command_contains_no_control_switch(self):
        cmd=d.command_for('exposure')
        self.assertIn('tools/exposure_tuner.py',cmd)
        self.assertNotIn('--port',cmd)


if __name__=='__main__':unittest.main()
