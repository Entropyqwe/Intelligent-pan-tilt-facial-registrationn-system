"""Sparse CPU face recognition; one request, no image backlog or serial access."""
from concurrent.futures import ThreadPoolExecutor
import time


class FaceWorker:
    def __init__(self, root, gallery, threads=2, hz=2., factory=None):
        self.root, self.gallery, self.threads = root, gallery, threads
        self.interval = 1./hz
        self.future = None
        self.last_submit = -float('inf')
        self.closed = False
        self.error = ''
        self.loaded = False
        self.completed = 0
        self.last_ms = 0.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='named-face')
        self.factory = factory
        self.detector = None

    def _process(self, frame, captured, tracks):
        if self.detector is None:
            if self.factory:
                self.detector = self.factory()
            else:
                from vision import FaceVision
                self.detector = FaceVision(self.root, threads=self.threads, max_faces=3,
                                           configure_threads=False, reject_overflow=True)
            self.loaded = True
        started = time.monotonic()
        faces = self.detector.process(frame, self.gallery)
        # Drop biometric arrays and crops immediately; only local gallery persists.
        clean = [{key: face[key] for key in ('person_id', 'similarity', 'box', 'score')} for face in faces]
        return dict(captured=captured, tracks=tracks, faces=clean, ms=(time.monotonic()-started)*1000)

    def submit(self, frame, captured, tracks, now):
        if self.closed or self.error or self.future is not None or not tracks or now-self.last_submit < self.interval:
            return False
        self.last_submit = now
        self.future = self.pool.submit(self._process, frame.copy(), captured,
                                      {tid: dict(row) for tid, row in tracks.items()})
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        try:
            result = future.result()
            self.completed += 1
            self.last_ms = result['ms']
            return result
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            print('人脸功能已停用：'+self.error, flush=True)
            return None

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
