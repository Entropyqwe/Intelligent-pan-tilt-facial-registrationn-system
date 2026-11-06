"""Repeated real-image compute benchmark. This is NOT live-camera acceptance."""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vision import FaceVision
from core import Gallery,control_fields
from body_tracker import BodyTargetLock
from inference_worker import PersonWorker
from selection import automatic_candidate
from display_faces import DisplayFaces
from hud import TelemetryHUD
from performance import PerformanceMeter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    frame=cv2.imread(str(args.image))
    if frame is None:raise SystemExit('Cannot read image')
    vision=FaceVision(root,threads=args.threads,detector_size=224)
    gallery=Gallery(root/'data/gallery.json')
    worker=PersonWorker(root)
    target=BodyTargetLock();heads=DisplayFaces();hud=TelemetryHUD()
    start=time.monotonic();perf=PerformanceMeter(start);count=valid_count=measured=0
    try:
        while time.monotonic()-start<args.seconds:
            captured=time.monotonic();check=not target.locked or count%3==0
            worker.submit(frame)
            faces=vision.process(frame,gallery) if check else []
            face_done=time.monotonic();bodies,body_ms=worker.result()
            if not target.target_id:
                candidate=automatic_candidate(faces,bodies)
                if candidate:target.select(candidate)
            snap=target.update(faces,bodies,captured,frame.shape[1],frame.shape[0],check)
            fields=control_fields(snap)
            done=time.monotonic();count+=1
            perf.add(done,count,check,True,(face_done-captured)*1000,body_ms,
                     (done-captured)*1000,(done-captured)*1000,bool(fields['valid']))
            display=heads.update(frame,faces,captured,check)
            hud.render(frame,display,bodies,target,snap,fields,perf.read(),'COMPUTE BENCHMARK / REPEATED IMAGE')
            if done-start>=2:
                measured+=1;valid_count+=bool(fields['valid'])
        duration=time.monotonic()-start
        stats=perf.read()
        stats.pop('camera_fps',None)
        result=dict(kind='repeated_real_image_compute_only',live_camera_acceptance=False,
                    serial_enabled=False,source_image=args.image.name,shape=list(frame.shape),
                    threads=args.threads,elapsed_seconds=duration,frames=count,
                    steady_compute_fps=measured/max(duration-2,1e-9),
                    fixture_valid_percent=100*valid_count/max(measured,1),final_window=stats,
                    includes='face/body/identity/association/display-flow/HUD composition; excludes camera and GUI presentation')
        args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:worker.close()


if __name__=='__main__':main()
