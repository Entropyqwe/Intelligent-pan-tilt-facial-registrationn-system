"""Exercise face-enabled body entry without opening camera/serial/audio hardware."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import body_main
from named_faces import NamedGallery


class FacePipelineTests(unittest.TestCase):
    def run_pipeline(self,strict=False,face_failure=False):
        sent=[];closed=[]
        class Camera:
            failed=False;disconnected=False
            def __init__(self,*a,**kw):self.seq=0
            def latest(self,seq):
                self.seq+=1
                return np.zeros((480,640,3),np.uint8),time.monotonic(),self.seq
            def close(self):closed.append('camera')
        class Person:
            backend_name='fixture';last_timing={}
            def __init__(self,*_):pass
            def submit(self,*_):pass
            def result(self):
                time.sleep(.002)
                return [dict(box=(100,40,200,400),score=.9)],2.
            def close(self):closed.append('person')
        class Policy:
            count=1
            def __init__(self,**kw):pass
            def update(self,*_):return [dict(person_id=1,box=(100,40,200,400),score=.9,
                similarity=1.,confirmed=True,observed=True)]
            def auto_select(self,s):
                if s.target_id!=1:s.select(1)
            def metrics(self):return {}
        class Face:
            loaded=True;completed=0;last_ms=2.;error='fixture face error' if face_failure else ''
            def __init__(self,*_):self.pending=None
            def poll(self):
                value,self.pending=self.pending,None
                return value
            def submit(self,frame,captured,tracks,now):
                if not face_failure:
                    self.pending=dict(captured=captured,tracks=tracks,faces=[dict(
                        person_id=1,box=(160,60,60,80),score=.99,similarity=.9)])
                    self.completed+=1
            def close(self):closed.append('face')
        class Speech:
            status='fixture voice'
            def __init__(self,*_):pass
            def enqueue(self,name):return True
            def close(self):closed.append('speech')
        class Output:
            serial_error=None
            def __init__(self,*a,**kw):pass
            def publish(self,s):sent.append(s)
            def close(self):closed.append('output')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);gallery=NamedGallery(path/'names.json')
            value=np.zeros(128);value[0]=1;gallery.add('测试人员',[value])
            args=['body_main.py','--no-show','--faces','on','--speech','on','--max-seconds','.04',
                  '--gallery',str(gallery.path),'--metrics-json',str(path/'metrics.json')]
            if strict:args+=['--follow-face','1']
            with patch('sys.argv',args),patch.object(body_main,'LatestCamera',Camera),\
                 patch.object(body_main,'PersonWorker',Person),patch.object(body_main,'SinglePersonPolicy',Policy),\
                 patch.object(body_main,'Output',Output),patch.object(body_main,'BodyHUD'),\
                 patch('face_worker.FaceWorker',Face),patch('welcome_speech.SpeechPlayer',Speech):
                body_main.main()
            metrics=json.loads((path/'metrics.json').read_text())
        self.assertTrue(sent)
        self.assertTrue(metrics['face']['enabled']);self.assertTrue(metrics['face_models_loaded'])
        self.assertEqual(set(closed),{'output','camera','person','face','speech'})
        return sent

    def test_label_mode_keeps_existing_body_control_fields(self):
        sent=self.run_pipeline(face_failure=True)
        self.assertTrue(any(s.valid for s in sent))
        self.assertTrue(all(s.source==2 for s in sent if s.valid))

    def test_strict_mode_never_steers_on_unconfirmed_or_failed_faces(self):
        for fail in (False,True):
            sent=self.run_pipeline(strict=True,face_failure=fail)
            self.assertFalse(any(s.valid for s in sent))
            self.assertTrue(all(s.source==0 and s.observation is None for s in sent))


if __name__=='__main__':unittest.main()
