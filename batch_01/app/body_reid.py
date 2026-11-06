"""Optional sparse OSNet x0.25 body embeddings through OpenCV CPU.

YOLO detection remains on VIPLite NPU. This module does not classify gender,
register faces, or promise permanent identity. Never share the Net across threads.
"""
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np


class BodyReID:
    backend_name='OSNet x0.25 / OpenCV CPU (sparse)'

    def __init__(self,model,expected_sha256=None):
        path=Path(model)
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256!=expected_sha256:
            raise ValueError('OSNet model SHA256 mismatch')
        self.net=cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.mean=np.array([.485,.456,.406],np.float32)[None,:,None,None]
        self.std=np.array([.229,.224,.225],np.float32)[None,:,None,None]
        self.calls=0;self.total_ms=0.;self.max_ms=0.

    def encode(self,frame,box):
        x,y,w,h=box;ih,iw=frame.shape[:2]
        x1,y1=max(0,int(x)),max(0,int(y));x2,y2=min(iw,int(x+w)),min(ih,int(y+h))
        if x2-x1<20 or y2-y1<40:return None
        start=time.monotonic()
        # Matches Torchreid test transform: RGB, resize H256/W128, ImageNet norm.
        crop=cv2.resize(frame[y1:y2,x1:x2],(128,256),interpolation=cv2.INTER_LINEAR)
        blob=crop[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.
        self.net.setInput((blob-self.mean)/self.std)
        embedding=self.net.forward().reshape(-1).astype(np.float32)
        if embedding.shape!=(512,) or not np.all(np.isfinite(embedding)):
            raise RuntimeError('Invalid OSNet embedding')
        norm=float(np.linalg.norm(embedding))
        if norm<1e-6:return None
        elapsed=(time.monotonic()-start)*1000
        self.calls+=1;self.total_ms+=elapsed;self.max_ms=max(self.max_ms,elapsed)
        return embedding/norm

    def metrics(self):
        return dict(backend=self.backend_name,model_sha256=self.sha256,calls=self.calls,
                    mean_ms=self.total_ms/max(self.calls,1),max_ms=self.max_ms)
