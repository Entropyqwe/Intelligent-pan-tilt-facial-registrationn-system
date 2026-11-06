"""Check camera and save a photo. No face recognition and no serial output."""
import argparse
import time
from pathlib import Path
import cv2
from main import LatestCamera

ap = argparse.ArgumentParser()
ap.add_argument("--camera",type=int,default=0)
ap.add_argument("--no-show",action="store_true")
args = ap.parse_args()
cam = LatestCamera(args.camera)
start = time.monotonic()
saved = False
try:
    while True:
        if cam.failed:
            raise RuntimeError("Camera read failed")
        frame,stamp,seq = cam.latest()
        if frame is None:
            if time.monotonic()-start > 5:
                raise RuntimeError("Camera opened but no frame received")
            time.sleep(.02)
            continue
        if not saved:
            path = Path(__file__).resolve().parent/"logs"/"camera_test.jpg"
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path),frame)
            print("Camera OK",frame.shape,"saved",path)
            saved = True
            if args.no_show:
                break
        cv2.imshow("Camera check - Q to quit",frame)
        if cv2.waitKey(1)&255 in (ord("q"),27):
            break
finally:
    cam.close()
    if not args.no_show:
        cv2.destroyAllWindows()
