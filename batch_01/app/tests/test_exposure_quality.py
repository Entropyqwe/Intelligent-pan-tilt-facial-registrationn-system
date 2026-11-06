import unittest
from unittest.mock import patch
import numpy as np
from exposure_quality import measure,rejection,verify_capture


class QualityTests(unittest.TestCase):
    def test_previously_saved_overexposed_frame_is_rejected(self):
        metrics=dict(mean_luma=239.1,clipped_fraction=.48,dark_fraction=0.,magenta_fraction=.2,chroma_noise=6.)
        self.assertIsNotNone(rejection(metrics))

    def test_lowering_brightness_does_not_hide_chroma_speckle(self):
        rng=np.random.default_rng(12)
        frame=np.full((480,640,3),100,np.float32)
        frame+=rng.normal(0,40,frame.shape)
        result=measure(np.clip(frame,0,255).astype(np.uint8))
        self.assertIsNotNone(rejection(result))

    def test_color_cast_worse_than_auto_is_rejected(self):
        baseline=dict(mean_luma=120.,clipped_fraction=.02,dark_fraction=.1,magenta_fraction=.01,chroma_noise=1.)
        self.assertIsNotNone(rejection(dict(baseline,magenta_fraction=.20),baseline))

    def test_normal_color_scene_is_not_automatically_rejected(self):
        frame=np.full((480,640,3),(70,100,130),np.uint8)
        self.assertIsNone(rejection(measure(frame)))

    def test_capture_verification_checks_real_pixels(self):
        class Capture:
            def read(self):return True,np.full((480,640,3),255,np.uint8)
        with patch('time.monotonic',side_effect=iter(range(100))):
            reason,stats=verify_capture(Capture(),warm_seconds=0.)
        self.assertIsNotNone(reason)


if __name__=='__main__':unittest.main()
