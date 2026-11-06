"""Measure actual VIPLite inference; this is NOT the full tracking FPS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from yolo_npu import YoloNpuVision,decode_heads

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',type=Path,required=True)
p.add_argument('--model',default='/opt/yolov5/model/yolov5.nb')
p.add_argument('--frames',type=int,default=100)
p.add_argument('--threshold',type=float,default=.45)
p.add_argument('--require-person',action='store_true')
a=p.parse_args()
if a.frames<1 or not 0<a.threshold<1:p.error('frames >= 1; threshold between 0 and 1')
frame=cv2.imread(str(a.image))
if frame is None:p.error('Image could not be opened')
cv2.setNumThreads(2)
detector=YoloNpuVision(a.model,a.threshold)
try:
    for _ in range(5):detector.process(frame)
    samples=[];start=time.monotonic()
    for _ in range(a.frames):
        bodies=detector.process(frame);samples.append(dict(detector.last_timing))
    elapsed=time.monotonic()-start
    full=detector.network.run(detector.tensor)
    h,w=frame.shape[:2];scale=min(640/w,640/h);rw,rh=int(w*scale),int(h*scale)
    reference=decode_heads(full,640,w,h,rw,rh,(640-rw)//2,(640-rh)//2,a.threshold,.5)
    assert len(bodies)==len(reference),(bodies,reference)
    for compact,complete in zip(bodies,reference):
        np.testing.assert_allclose(compact['box'],complete['box'],atol=.002)
        np.testing.assert_allclose(compact['score'],complete['score'],atol=.00001)
    if a.require_person and not bodies:raise RuntimeError('No person detected; nonempty correctness check not satisfied')
    print(json.dumps(dict(model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                          inference_only_fps=a.frames/elapsed,
                          mean_ms={k:float(np.mean([r[k] for r in samples])) for k in samples[0]},
                          detections=bodies,compact_matches_full=True),indent=2))
finally:detector.close()
