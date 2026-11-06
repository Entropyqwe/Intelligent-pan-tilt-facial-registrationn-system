"""Offline old/new comparison on a local clip; never opens a serial port."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from yolo_npu import YoloNpuVision
from body_identity import BodyIdentities,BodySelection
from core import control_fields


def load_old(path):
    spec=importlib.util.spec_from_file_location('baseline_identity',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.BodyIdentities()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('--timing',type=Path,required=True)
    p.add_argument('--old-module',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,help='Local detection JSON cache, same clip only')
    p.add_argument('--model',default='/opt/yolov5/model/yolov5.nb')
    p.add_argument('--reid-model',type=Path)
    a=p.parse_args();cv2.setNumThreads(2)
    timing=json.loads(a.timing.read_text());cached=None
    if a.cache and a.cache.exists():cached=json.loads(a.cache.read_text())
    video=cv2.VideoCapture(str(a.video));assert video.isOpened()
    detector=None if cached is not None else YoloNpuVision(a.model,threshold=.1)
    old=load_old(a.old_module) if a.old_module else None
    encoder=None
    if a.reid_model:
        from body_reid import BodyReID
        encoder=BodyReID(a.reid_model)
    tracker=BodyIdentities(reid=encoder);selector=BodySelection();rows=[];samples=[];cache=[];tiles=[]
    valid=detected=0;start=time.monotonic()
    try:
        for i,stamp in enumerate(timing):
            ok,frame=video.read()
            if not ok:break
            bodies=cached[i] if cached is not None else detector.process(frame);cache.append(bodies)
            captured=100.+stamp['captured'];t=time.monotonic()
            tracks=tracker.update(frame,bodies,captured);samples.append((time.monotonic()-t)*1000)
            ready=[x for x in tracks if x['confirmed'] and x['person_id']>0]
            if not selector.target_id and len(ready)==1 and len(bodies)==1:selector.select(ready[0]['person_id'])
            snap=selector.update(tracks,captured,640,480);valid+=bool(control_fields(snap,now=captured)['valid'])
            detected+=any(b['score']>=.35 for b in bodies)
            previous=old.update(frame,[b for b in bodies if b['score']>=.35],captured) if old else []
            rows.append(dict(t=stamp['captured'],boxes=bodies,new=tracks,old_ids=[b['person_id'] for b in previous],
                             target=selector.target_id,mode=selector.mode))
            if len(tiles)<16 and stamp['captured']>=len(tiles)*2.2:
                image=frame.copy()
                for track in tracks:
                    x,y,w,h=map(int,track['box']);color=(0,255,0) if track['confirmed'] else (0,180,255)
                    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                    cv2.putText(image,str(track['person_id']),(x,max(20,y+20)),0,.6,color,2)
                cv2.rectangle(image,(0,0),(640,28),(0,0,0),-1)
                cv2.putText(image,f"t={stamp['captured']:.1f} target={selector.target_id} {selector.mode}",(5,20),0,.55,(255,255,255),1)
                tiles.append(cv2.resize(image,(320,240)))
            if i%150==0:print('REPLAY',i,tracker.metrics(),flush=True)
        result=dict(frames=len(rows),clip_seconds=timing[min(len(rows),len(timing))-1]['captured'],
                    offline_elapsed_seconds=time.monotonic()-start,valid_percent=100*valid/max(len(rows),1),
                    high_detection_percent=100*detected/max(len(rows),1),tracking=tracker.metrics(),
                    old_public_ids_created=old.next_id-1 if old else None,
                    association_mean_ms=float(np.mean(samples)),association_p95_ms=float(np.percentile(samples,95)),
                    note='This is unlabelled clip comparison, not measured IDSW/accuracy or live FPS.')
        a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
        a.output.with_suffix('.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
        if a.cache:a.cache.write_text(json.dumps(cache),encoding='utf8')
        if tiles:
            sheet=np.zeros((240*((len(tiles)+3)//4),1280,3),np.uint8)
            for i,tile in enumerate(tiles):sheet[(i//4)*240:(i//4+1)*240,(i%4)*320:(i%4+1)*320]=tile
            cv2.imwrite(str(a.output.with_suffix('.jpg')),sheet)
        print(json.dumps(result,indent=2),flush=True)
    finally:
        video.release()
        if detector:detector.close()


if __name__=='__main__':main()
