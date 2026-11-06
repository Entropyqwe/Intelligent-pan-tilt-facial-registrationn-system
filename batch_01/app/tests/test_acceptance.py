import unittest
from tools.acceptance_report import evaluate


class AcceptanceTests(unittest.TestCase):
    def metrics(self):
        return dict(capture_width=640,capture_height=480,capture_channels=3,elapsed_seconds=61,
            steady_wall_fps=22,body_detection_fps=22,backend='VIPLite NPU / YOLOv5',
            valid_percent=95,longest_valid_seconds=15,serial_enabled=True,serial_error=None,
            steering_enabled=True,quality_samples=[dict(color_fraction=.2,mean_luma=120,
                dark_fraction=.1,clipped_fraction=.1)])

    def test_good_full_run_passes(self):
        self.assertTrue(evaluate(self.metrics())['passed'])

    def test_fast_camera_does_not_hide_slow_processing(self):
        m=self.metrics();m.update(steady_wall_fps=8,camera_fps=30)
        self.assertFalse(evaluate(m)['passed'])

    def test_empty_scene_and_black_frames_cannot_pass(self):
        for changes in (dict(valid_percent=0),dict(serial_enabled=False),
                        dict(quality_samples=[dict(color_fraction=.2,mean_luma=3,dark_fraction=.99,clipped_fraction=0)])):
            m=self.metrics();m.update(changes)
            self.assertFalse(evaluate(m)['passed'])
