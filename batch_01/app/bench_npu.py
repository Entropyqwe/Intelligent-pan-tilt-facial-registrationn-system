import json
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_npu import YoloNpuVision

cv2.setNumThreads(2)
image=cv2.imread('/opt/yolov5/input_data/dog.jpg')
vision=YoloNpuVision(threshold=.25)
try:
    for _ in range(5):vision.process(image)
    values=[]
    for _ in range(60):
        vision.process(image)
        values.append(vision.last_timing)
    print('TIMING',json.dumps({k:float(np.mean([x[k] for x in values])) for k in values[0]}))
    print('DOG_PERSON_RESULT',vision.process(image))
    # Alternate image inputs to catch stale input/output caching.
    root=Path(__file__).resolve().parent
    person=cv2.imread(str(root/'logs/camera_test.jpg')) if (root/'logs/camera_test.jpg').exists() else None
    if person is not None:print('CAMERA_IMAGE_RESULT',vision.process(person))
    camera=cv2.VideoCapture(0,cv2.CAP_V4L2)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        camera.set(cv2.CAP_PROP_FPS,30)
        ok,frame=camera.read()
        if ok:
            # Live inference only; do not silently save a person's photograph.
            print('LIVE_PERSON_RESULT',vision.process(frame))
        camera.release()
finally:vision.close()
