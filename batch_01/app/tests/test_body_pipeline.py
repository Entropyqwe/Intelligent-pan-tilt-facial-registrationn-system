"""Verify overlapping inference never associates detections with another frame."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import body_main


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, overlap, stale=False):
        pairs=[];submitted=[];closed=[]

        class Camera:
            failed=False
            disconnected=False
            number=0
            def __init__(self,*args,**kwargs):pass
            def latest(self,seq):
                self.number+=1
                if self.number%3==0:return None,time.monotonic(),seq
                frame=np.full((480,640,3),self.number%255,np.uint8)
                return frame,time.monotonic()-(1. if stale else 0.),self.number
            def close(self):closed.append('camera')

        class Worker:
            backend_name='mock'
            last_timing={}
            pending=None
            def __init__(self,*args):pass
            def submit(self,frame):
                if self.pending is not None:raise AssertionError('unbounded requests')
                self.pending=int(frame[0,0,0]);submitted.append(self.pending)
            def result(self):
                if self.pending is None:raise AssertionError('missing request')
                time.sleep(.002)
                value=self.pending;self.pending=None
                return [dict(frame=value)],2.
            def close(self):closed.append('worker')

        class Policy:
            def __init__(self,**kw):pass
            def update(self,frame,bodies,*args):
                pairs.append((int(frame[0,0,0]),bodies))
                return []
            def auto_select(self,*args):pass
            def metrics(self):return {}

        class Output:
            serial_error=None
            def __init__(self,*args,**kwargs):pass
            def publish(self,snapshot):pass
            def close(self):closed.append('output')

        with tempfile.TemporaryDirectory() as tmp:
            args=['body_main.py','--no-show','--max-seconds','.04',
                  '--inference-overlap',overlap,'--metrics-json',str(Path(tmp)/'metrics.json')]
            with patch('sys.argv',args),patch.object(body_main,'LatestCamera',Camera),\
                 patch.object(body_main,'PersonWorker',Worker),patch.object(body_main,'SinglePersonPolicy',Policy),\
                 patch.object(body_main,'Output',Output),patch.object(body_main,'BodyHUD'):
                body_main.main()
        self.assertGreater(len(pairs),2)
        self.assertEqual(closed,['output','camera','worker'])
        for frame,bodies in pairs:
            if stale:self.assertEqual(bodies,[])
            else:self.assertEqual(frame,bodies[0]['frame'])
        self.assertEqual(len(submitted),len(set(submitted)))

    def test_overlap_keeps_detection_frame_pairing(self):self.run_pipeline('on')
    def test_serial_mode_keeps_detection_frame_pairing(self):self.run_pipeline('off')
    def test_overlapped_stale_frame_is_not_used_for_control(self):self.run_pipeline('on',True)


if __name__=='__main__':unittest.main()
