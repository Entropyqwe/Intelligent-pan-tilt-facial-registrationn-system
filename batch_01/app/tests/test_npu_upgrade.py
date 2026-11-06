import unittest
from unittest.mock import patch
from pathlib import Path
import threading
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selection import automatic_candidate
from yolo_npu import decode_heads,decode_candidates
from inference_worker import PersonWorker
from core import Snapshot
from body_tracker import BodyTargetLock
from hud import TelemetryHUD
from main import LatestCamera


class SelectionTests(unittest.TestCase):
    def test_registered_unique_association_required(self):
        face=dict(person_id=1,box=(100,60,60,60))
        body=dict(box=(70,40,130,300))
        self.assertEqual(automatic_candidate([face],[body]),1)
        self.assertEqual(automatic_candidate([dict(face,person_id=0)],[body]),0)
        self.assertEqual(automatic_candidate([face],[]),0)
        self.assertEqual(automatic_candidate([face,dict(face,person_id=2)],[body]),0)
        self.assertEqual(automatic_candidate([face],[body,body]),0)
        self.assertEqual(automatic_candidate([face,dict(face,person_id=0)],[body]),0)

    def test_original_tracker_still_requires_confirmation(self):
        face=dict(person_id=1,box=(100,60,60,60),similarity=.8)
        body=dict(box=(70,40,130,300),score=.9)
        tracker=BodyTargetLock()
        tracker.select(automatic_candidate([face],[body]))
        for i in range(3):
            snap=tracker.update([face],[body],10+i*.04,640,480)
            self.assertEqual(snap.valid,i==2)


class DecoderTests(unittest.TestCase):
    def heads(self):
        return [np.full((1,3,g,g,85),-20.,np.float32) for g in (80,40,20)]

    def test_positive_coordinate_mapping_and_compact_equivalence(self):
        heads=self.heads()
        row=heads[2][0,2,10,10]
        row[:4]=0;row[4:6]=5
        result=decode_heads(heads,640,640,480,640,480,0,80)
        compact=np.array([[32,2,10,10,0,0,0,0,float(1/(1+np.exp(-5.)))**2]],np.float32)
        actual=decode_candidates(compact,640,480,640,480,0,80)
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0]['box'],(149.5,93.,373.,326.),atol=.001)
        np.testing.assert_allclose(actual[0]['box'],result[0]['box'],atol=.001)
        self.assertAlmostEqual(actual[0]['score'],result[0]['score'],places=6)
        row[6]=6
        self.assertEqual(decode_heads(heads,640,640,480,640,480,0,80),[])

    def test_empty_and_wrong_layout(self):
        self.assertEqual(decode_heads(self.heads(),640,640,480,640,480,0,80),[])
        with self.assertRaises(ValueError):decode_heads(self.heads()[:2],640,640,480,640,480,0,80)
        with self.assertRaises(ValueError):decode_heads([np.zeros((3,80,80,85))],640,640,480,640,480,0,80)


class WorkerTests(unittest.TestCase):
    def test_ownership_single_inflight_and_error_propagation(self):
        events=[]
        class Detector:
            def __init__(self):events.append(('create',threading.get_ident()))
            def process(self,frame):
                events.append(('process',threading.get_ident()))
                if frame[0]==-1:raise RuntimeError('inference failed')
                return frame
            def close(self):events.append(('close',threading.get_ident()))
        with patch.object(PersonWorker,'_create',side_effect=lambda *a:Detector()):
            worker=PersonWorker(Path('.'))
            frame=[1]
            worker.submit(frame)
            with self.assertRaises(RuntimeError):worker.submit(frame)
            self.assertIs(worker.result()[0],frame)
            worker.submit([-1])
            with self.assertRaisesRegex(RuntimeError,'inference failed'):worker.result()
            worker.close()
        self.assertEqual(len({x[1] for x in events}),1)
        self.assertNotEqual(events[0][1],threading.get_ident())


class HudTests(unittest.TestCase):
    def test_mouse_mapping_and_input_preserved(self):
        hud=TelemetryHUD();frame=np.zeros((480,640,3),np.uint8)
        stats=dict(fps=24,camera_fps=30,face_fps=8,face_ms=15,body_ms=32,age_ms=45)
        canvas=hud.render(frame,[],[],BodyTargetLock(),Snapshot(),{'valid':0},stats,'test')
        self.assertEqual(canvas.shape,(646,920,3))
        self.assertEqual(hud.to_image(20,88),(0,0))
        self.assertIsNone(hud.to_image(19,88))
        self.assertIsNone(hud.to_image(660,88))
        self.assertFalse(frame.any())

    def test_hd_preview_mapping_preserves_source_pixels(self):
        hud=TelemetryHUD();frame=np.zeros((720,1280,3),np.uint8)
        frame[100,200]=(10,20,30)
        stats=dict(fps=24,camera_fps=30,face_fps=8,face_ms=15,body_ms=32,age_ms=45)
        canvas=hud.render(frame,[],[],BodyTargetLock(),Snapshot(),{'valid':0},stats,'HD')
        self.assertEqual(canvas.shape,(886,1560,3))
        self.assertEqual(hud.to_image(220,188),(200,100))
        np.testing.assert_array_equal(canvas[188,220],frame[100,200])


class CameraModeTests(unittest.TestCase):
    def test_requested_hd_mode_is_sent_and_checked(self):
        import cv2
        with patch('main.cv2.VideoCapture') as factory,patch('main.threading.Thread'):
            cap=factory.return_value
            cap.isOpened.return_value=True
            cap.get.side_effect=lambda p:{cv2.CAP_PROP_FRAME_WIDTH:1280,
                                          cv2.CAP_PROP_FRAME_HEIGHT:720,
                                          cv2.CAP_PROP_FPS:60}[p]
            camera=LatestCamera(0,1280,720,30)
            cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT,720)
            camera.close()

    def test_driver_fallback_is_reported_not_called_hd(self):
        import cv2
        with patch('main.cv2.VideoCapture') as factory,patch('main.threading.Thread') as thread:
            cap=factory.return_value
            cap.isOpened.return_value=True
            cap.get.side_effect=lambda p:{cv2.CAP_PROP_FRAME_WIDTH:640,
                                          cv2.CAP_PROP_FRAME_HEIGHT:480}[p]
            with self.assertRaisesRegex(RuntimeError,'Camera returned'):
                LatestCamera(0,1280,720)
            cap.release.assert_called_once()
            thread.assert_not_called()


if __name__=='__main__':unittest.main()
