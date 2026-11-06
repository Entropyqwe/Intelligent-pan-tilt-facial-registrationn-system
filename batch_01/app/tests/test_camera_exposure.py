import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
import camera_exposure as e

CTRL=dict(auto_exposure=1,exposure_dynamic_framerate=0,exposure_time_absolute=100,
          brightness=128,white_balance_automatic=1,white_balance_temperature=4000)
INFO={k:dict(min=0,max=5000,step=1) for k in CTRL}
INFO['exposure_dynamic_framerate']['max']=1
BASE=dict(mean_luma=120.,dark_fraction=.1,clipped_fraction=.01,magenta_fraction=.01,chroma_noise=1.)
TEST=dict(seconds=5.,frames=101,fps=20.,**BASE,auto_baseline=BASE,replay_verified=True)


class ExposureTests(unittest.TestCase):
    def save(self,path):
        with patch.object(e,'capture_controls',return_value=CTRL),patch.object(e,'control_info',return_value=INFO),\
             patch.object(e,'camera_identity',return_value='0c45:64ab'):
            return e.save_profile(0,CTRL,TEST,path,quality_confirmed=True)

    def test_freeze_reads_before_switching_mode(self):
        calls=[]
        with patch.object(e,'capture_controls',side_effect=lambda _: (calls.append('read') or CTRL.copy())),\
             patch.object(e,'set_control',side_effect=lambda i,k,v:calls.append((k,v))):
            e.freeze_auto(0)
        self.assertEqual(calls,['read',('auto_exposure',1),('exposure_dynamic_framerate',0),'read'])

    def test_changed_controls_cannot_save_previous_test(self):
        with patch.object(e,'capture_controls',return_value=dict(CTRL,brightness=70)):
            with self.assertRaisesRegex(ValueError,'重新测试'):e.save_profile(0,CTRL,TEST)

    def test_test_required_and_black_frame_rejected(self):
        for test in (None,dict(TEST,seconds=1),dict(TEST,fps=float('nan')),
                     dict(TEST,dark_fraction=.99),dict(TEST,clipped_fraction=.99)):
            with self.subTest(test=test),self.assertRaises(ValueError):e.validate_test(test)

    def test_old_speed_only_test_cannot_be_loaded(self):
        old=dict(seconds=5.,frames=150,fps=30.,mean_luma=239.,dark_fraction=0.,clipped_fraction=.48)
        with self.assertRaises(ValueError):e.validate_test(old)

    def test_replay_and_visual_quality_confirmation_are_required(self):
        with self.assertRaisesRegex(ValueError,'重载'):e.validate_test(dict(TEST,replay_verified=False))
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'capture_controls',return_value=CTRL),\
             patch.object(e,'control_info',return_value=INFO):
            with self.assertRaisesRegex(ValueError,'确认'):e.save_profile(0,CTRL,TEST,Path(tmp)/'x.json')

    def test_saved_profile_roundtrip_and_wrong_camera_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixed.json';self.save(path)
            with patch.object(e,'camera_identity',return_value='0c45:64ab'),patch.object(e,'control_info',return_value=INFO):
                self.assertEqual(e.load_profile(0,path)['controls'],CTRL)
            with patch.object(e,'camera_identity',return_value='1234:5678'):
                with self.assertRaisesRegex(ValueError,'型号'):e.load_profile(0,path)

    def test_manual_mode_and_dynamic_framerate_are_validated(self):
        for changes in (dict(auto_exposure=3),dict(exposure_dynamic_framerate=1),dict(brightness=-1)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                e.validate_controls(dict(CTRL,**changes),INFO)

    def test_white_balance_temperature_can_vary_automatically(self):
        self.assertTrue(e.same_controls(CTRL,dict(CTRL,white_balance_temperature=4500)))

    def test_inactive_auto_temperature_is_not_written(self):
        with patch.object(e,'set_control') as setter:
            e.apply_controls(0,CTRL)
        self.assertNotIn('white_balance_temperature',[call.args[1] for call in setter.call_args_list])

    def test_missing_profile_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(RuntimeError,'曝光调试'):
            e.load_profile(0,Path(tmp)/'missing.json')

    def test_v4l2_info_parser_supports_menu_and_inactive_integer(self):
        output='''auto_exposure 0x009a0901 (menu) : min=0 max=3 default=3 value=3
        exposure_time_absolute 0x009a0902 (int) : min=10 max=2500 step=1 default=156 value=156 flags=inactive
        exposure_dynamic_framerate 0x009a0903 (bool) : default=0 value=0'''
        with patch.object(e,'ctl',return_value=output):info=e.control_info(0)
        self.assertTrue(info['exposure_time_absolute']['inactive'])
        self.assertEqual(info['exposure_dynamic_framerate']['max'],1)


if __name__=='__main__':unittest.main()
