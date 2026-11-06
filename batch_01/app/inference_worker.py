"""Bounded one-frame person worker; CPU faces and NPU see the SAME frame.

No queue of old images, no reuse of detections as a fresh observation. VIPLite
network is constructed, run and destroyed entirely on its owning thread.
"""
from concurrent.futures import ThreadPoolExecutor


class PersonWorker:
    def __init__(self, root, backend='npu', threshold=.45,
                 model='/opt/yolov5/model/yolov5.nb'):
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='person-inference')
        self.future = None
        self.closed = False
        self.last_timing = {}
        try:
            self.detector = self.executor.submit(self._create,root,backend,threshold,model).result()
        except BaseException:
            self.executor.shutdown(wait=True)
            raise
        self.backend_name = getattr(self.detector,'backend_name','OpenCV CPU / NanoDet')

    @staticmethod
    def _create(root,backend,threshold,model):
        if backend == 'npu':
            from yolo_npu import YoloNpuVision
            return YoloNpuVision(model,threshold)
        if backend == 'cpu':
            from body_vision import PersonVision
            return PersonVision(root,threshold)
        raise ValueError('Explicit person backend must be npu or cpu')

    def submit(self, frame):
        if self.closed or self.future is not None:
            raise RuntimeError('Only one same-frame person request may be in flight')
        self.future = self.executor.submit(self._process,frame)

    def _process(self,frame):
        import time
        start=time.monotonic()
        detections=self.detector.process(frame)
        return detections,(time.monotonic()-start)*1000.,dict(getattr(self.detector,'last_timing',{}))

    def result(self):
        if self.future is None:
            raise RuntimeError('No submitted person frame')
        try:
            detections,elapsed,self.last_timing=self.future.result()
            return detections,elapsed
        finally:
            self.future = None

    def close(self):
        if self.closed:
            return
        self.closed=True
        try:
            if self.future is not None:
                try:self.future.result()
                finally:self.future=None
        finally:
            try:
                closer=getattr(self.detector,'close',None)
                if closer:self.executor.submit(closer).result()
            finally:self.executor.shutdown(wait=True)
