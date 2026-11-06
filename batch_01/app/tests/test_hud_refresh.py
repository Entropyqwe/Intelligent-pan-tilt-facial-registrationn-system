import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from body_hud import BodyHUD


class HUDRefreshTests(unittest.TestCase):
    def test_cached_frames_keep_clock_and_invalid_state_refreshes_immediately(self):
        target=SimpleNamespace(target_id=1,mode='BODY',locked=True)
        stats=dict(fps=25.,camera_fps=30.,body_ms=30.,appearance_ms=.1,age_ms=40.)
        frame=np.zeros((480,640,3),np.uint8)
        distance=dict(mm=500,valid=True,mode='TRACK_READY')
        with patch('body_hud.write') as write,patch('body_hud.time.monotonic',return_value=10.):
            hud=BodyHUD()
            hud.render(frame,[],target,stats,0.,True,'/dev/fake',10.,distance)
            write.reset_mock()
            hud.render(frame,[],target,stats,0.,True,'/dev/fake',10.,distance)
            self.assertEqual([call.args[1] for call in write.call_args_list],['T+00:10'])
            distance=dict(mm=0,valid=False,mode='STOP_SENSOR')
            write.reset_mock()
            hud.render(frame,[],target,stats,0.,True,'/dev/fake',10.,distance)
            self.assertIn('--.- cm',[call.args[1] for call in write.call_args_list])


if __name__=='__main__':unittest.main()
