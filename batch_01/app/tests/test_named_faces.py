import tempfile
import unittest
from pathlib import Path
import numpy as np
from named_faces import NamedGallery,embedding,checked_name
from face_link import FaceBindings,associate
from welcome_speech import WelcomeGate,SpeechPlayer
from face_worker import FaceWorker
from launch import command_for
import time
from types import SimpleNamespace


def vector(n=0):
    value=np.zeros(128,np.float32);value[n]=1
    return value


def track(tid=1,box=(100,50,200,400)):
    return dict(person_id=tid,box=box,confirmed=True,observed=True)


def face(pid=1,box=(160,65,60,75)):
    return dict(person_id=pid,box=box,similarity=.9,score=.99)


class NamedGalleryTests(unittest.TestCase):
    def test_roundtrip_names_features_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'names.json';g=NamedGallery(path)
            pid=g.add('张三',[vector()]);g.add('张三',[vector(1)])
            g=NamedGallery(path)
            self.assertEqual(g.name(pid),'张三');self.assertEqual(len(g.people),1)
            self.assertEqual(g.match(vector())[0],pid)
            g.delete(pid);self.assertFalse(NamedGallery(path).people)

    def test_margin_rejects_conflicting_enrollments(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=NamedGallery(Path(tmp)/'g.json');g.add('甲',[vector()]);g.add('乙',[vector()])
            self.assertEqual(g.match(vector())[0],0)

    def test_invalid_features_and_names_rejected(self):
        for v in (np.zeros(128),np.ones(512),np.full(128,np.nan)):
            with self.assertRaises(ValueError):embedding(v)
        for name in ('','../../张三','A\nB','a'*25):
            with self.assertRaises(ValueError):checked_name(name)

    def test_empty_unknown_and_id_rename_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=NamedGallery(Path(tmp)/'g.json')
            self.assertEqual(g.match(vector())[0],0)
            pid=g.add('甲',[vector()])
            with self.assertRaises(ValueError):g.add('乙',[vector(1)],pid)
            self.assertEqual(g.match(vector(2))[0],0)


class FaceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.g=NamedGallery(Path(self.tmp.name)/'g.json');self.g.add('张三',[vector()])
        self.b=FaceBindings(self.g)

    def tearDown(self):self.tmp.cleanup()

    def feed(self,stamp,pid=1,faces=None,tracks=None):
        self.b.observe(tracks if tracks is not None else [track()],stamp)
        packet=dict(captured=stamp,tracks=self.b.tokens(),faces=[face(pid)] if faces is None else faces)
        return self.b.accept(packet,stamp+.05)

    def confirm(self):
        # Body observations continue between sparse face results.
        for t in (0,.25,.5,.75,1.):self.feed(t)

    def test_needs_three_consistent_recognitions_and_time(self):
        self.feed(0);self.feed(.3)
        self.assertFalse(self.b.labels(.35)[1]['verified'])
        self.feed(.6);self.assertEqual(self.b.labels(.65)[1]['name'],'张三')

    def test_absent_slot_and_abrupt_jump_revoke_name(self):
        self.confirm();self.b.observe([],1.1);self.b.observe([track()],1.2)
        self.assertNotIn(1,self.b.labels(1.25))
        self.confirm();self.b.observe([track(box=(500,50,100,300))],1.1)
        self.assertNotIn(1,self.b.labels(1.2))

    def test_stale_out_of_order_old_tokens_never_bind(self):
        self.b.observe([track()],0)
        old=dict(captured=0,tracks=self.b.tokens(),faces=[face()])
        self.assertFalse(self.b.accept(old,1.))
        self.b.observe([], .1);self.b.observe([track()], .2)
        self.b.accept(old,.3)
        self.assertFalse(self.b.labels(.3))
        self.assertFalse(self.b.accept(old,.3))

    def test_unknown_and_missing_face_clear_verified_state(self):
        self.confirm();self.feed(1.25,pid=0)
        self.assertEqual(self.b.labels(1.3)[1]['identity_id'],0)
        self.feed(1.5,faces=[])
        self.assertFalse(self.b.labels(1.6)[1]['verified'])

    def test_identity_expires_even_if_body_continues(self):
        self.confirm();self.b.observe([track()],1.2)
        self.assertFalse(self.b.labels(2.6))

    def test_overlapping_bodies_two_faces_and_outside_are_ambiguous(self):
        tracks={1:dict(box=track()['box']),2:dict(box=track()['box'])}
        self.assertFalse(associate([face()],tracks))
        self.assertFalse(associate([face(),face()],{1:tracks[1]}))
        self.assertFalse(associate([face(box=(0,400,60,70))],{1:tracks[1]}))

    def test_same_identity_on_two_bodies_is_suppressed_across_results(self):
        other=track(2,box=(350,50,200,400))
        for t in (0,.25,.5):
            self.feed(t,tracks=[track(),other],faces=[face(),face(0,(410,65,60,75))])
        self.assertTrue(self.b.labels(.55)[1]['verified'])
        for t in (.75,1.,1.25):
            self.feed(t,tracks=[track(),other],faces=[face(),face(1,(410,65,60,75))])
        self.assertTrue(all(not row['verified'] for row in self.b.labels(1.3).values()))


class WelcomeTests(unittest.TestCase):
    def test_welcome_once_then_departure_and_cooldown(self):
        g=WelcomeGate();said=[]
        def enqueue(name):said.append(name);return True
        for now in (0,1,2,4,65):g.update({1:'张三'},now,enqueue)
        # A 61-second absence allows the second welcome, continuous presence does not.
        self.assertEqual(said,['张三','张三'])
        g.update({1:'张三'},66,enqueue);g.update({},75,enqueue)
        g.update({1:'张三'},80,enqueue)
        self.assertEqual(len(said),2)
        g.update({1:'张三'},145,enqueue);self.assertEqual(len(said),3)

    def test_continuous_presence_and_unknown_do_not_repeat(self):
        g=WelcomeGate();said=[]
        def enqueue(n):said.append(n);return True
        for now in range(200):g.update({1:'李四'},now,enqueue)
        for now in range(200,205):g.update({},now,enqueue)
        self.assertEqual(said,['李四'])

    def test_failed_audio_can_retry_without_blocking(self):
        g=WelcomeGate();g.update({1:'甲'},0,lambda n:False)
        said=[];g.update({1:'甲'},.1,lambda n:said.append(n) or True)
        self.assertEqual(said,['甲'])

    def test_subprocess_uses_stdin_argv_and_no_shell(self):
        calls=[]
        def run(args,**kw):calls.append((args,kw));return SimpleNamespace(returncode=0,stderr='')
        p=SpeechPlayer(executable='test-espeak',runner=run)
        try:
            self.assertTrue(p.enqueue('张三'));p.jobs.join()
            self.assertEqual(calls[0][0],['test-espeak','-v','cmn','-s','155','--stdin'])
            self.assertEqual(calls[0][1]['input'],'张三，欢迎光临。\n')
            self.assertNotIn('shell',calls[0][1]);self.assertIn('已欢迎',p.status)
        finally:p.close()

    def test_missing_audio_error_does_not_escape_worker(self):
        def run(*a,**k):raise OSError('no sound card')
        p=SpeechPlayer(executable='test-espeak',runner=run)
        try:p.enqueue('甲');p.jobs.join();self.assertEqual(p.status,'音频不可用')
        finally:p.close()


class FaceWorkerTests(unittest.TestCase):
    def test_bounded_sparse_queue_no_biometric_payload(self):
        import threading
        gate=threading.Event()
        class Detector:
            def process(self,frame,gallery):
                gate.wait(1)
                return [dict(face(),feature=vector(),aligned=frame)]
        worker=FaceWorker(Path('.'),None,factory=Detector)
        try:
            frame=np.zeros((16,16,3),np.uint8)
            self.assertTrue(worker.submit(frame,0,{1:dict(token=1,box=(0,0,16,16))},0))
            self.assertFalse(worker.submit(frame,.1,{},.1))
            self.assertIsNone(worker.poll());gate.set()
            until=time.monotonic()+2
            result=None
            while result is None and time.monotonic()<until:result=worker.poll();time.sleep(.001)
            self.assertIsNotNone(result)
            self.assertNotIn('feature',result['faces'][0]);self.assertNotIn('aligned',result['faces'][0])
            self.assertFalse(worker.submit(frame,.2,{1:{}},.2))
        finally:gate.set();worker.close()

    def test_inference_exception_disables_only_faces(self):
        class Detector:
            def process(self,*_):raise RuntimeError('broken face model')
        worker=FaceWorker(Path('.'),None,factory=Detector)
        try:
            worker.submit(np.zeros((2,2,3)),0,{1:{}},0)
            until=time.monotonic()+2
            while not worker.error and time.monotonic()<until:worker.poll();time.sleep(.001)
            self.assertIn('broken face model',worker.error)
        finally:worker.close()


class NewLauncherTests(unittest.TestCase):
    def test_preview_never_opens_serial_full_uses_original_geometry(self):
        for mode in ('preview','body-only'):
            cmd=command_for(mode,port='/dev/ttyUSB0')
            self.assertNotIn('--port',cmd)
        cmd=command_for('full',port='/dev/ttyUSB0')
        self.assertIn('--port',cmd);self.assertIn('--gimbal-config',cmd)
        self.assertEqual(cmd[cmd.index('--frame-rotation')+1],'180')
        self.assertEqual(cmd[cmd.index('--faces')+1],'on')
        self.assertEqual(cmd[cmd.index('--speech')+1],'on')


if __name__=='__main__':unittest.main()
