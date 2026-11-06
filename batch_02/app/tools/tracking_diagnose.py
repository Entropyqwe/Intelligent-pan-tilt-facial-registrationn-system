"""Same-frame old/new association comparison and real NPU/camera profiling.

Stores detection/track metadata, not videos. No serial output is opened.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from yolo_npu import YoloNpuVision
from body_identity import BodyIdentities,BodySelection
from core import control_fields
from main import LatestCamera


def summarize(samples):
    return {name:{'mean_ms':float(np.mean(values)), 'p95_ms':float(np.percentile(values,95))}
            for name,values in samples.items() if values}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=35.)
    p.add_argument('--old-module',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample',type=Path)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    old=None
    if a.old_module:
        spec=importlib.util.spec_from_file_location('baseline_identity',a.old_module)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        old=module.BodyIdentities()
    cv2.setNumThreads(2);detector=YoloNpuVision(threshold=.1)
    tracker=BodyIdentities();selector=BodySelection()
    camera=LatestCamera(0,640,480,30);seq=0;frames=0;first_seq=0
    times={k:[] for k in ('detector','association','old_association','pipeline','frame_age')}
    records=[];start=time.monotonic();valid=0
    try:
        while time.monotonic()-start<a.seconds:
            frame,captured,number=camera.latest(seq)
            if frame is None:
                if camera.failed:raise RuntimeError('Camera failed')
                time.sleep(.001);continue
            if not frames:first_seq=number
            seq=number;t=time.monotonic();bodies=detector.process(frame);infer=time.monotonic()
            tracks=tracker.update(frame,bodies,captured);associated=time.monotonic()
            if not selector.target_id:
                ready=[x for x in tracks if x['confirmed']]
                if len(ready)==1 and len(bodies)==1:selector.select(ready[0]['person_id'])
            snap=selector.update(tracks,captured,640,480);valid+=bool(control_fields(snap)['valid'])
            old_tracks=[];old_start=time.monotonic()
            if old:old_tracks=old.update(frame,[b for b in bodies if b['score']>=.35],captured)
            done=time.monotonic();frames+=1
            values=dict(detector=1000*(infer-t),association=1000*(associated-infer),
                        old_association=1000*(done-old_start),pipeline=1000*(associated-t),
                        frame_age=1000*(associated-captured))
            if time.monotonic()-start>2:
                for k,v in values.items():times[k].append(v)
                for k,v in detector.last_timing.items():times.setdefault(k,[]).append(v)
            records.append(dict(t=captured-start,boxes=bodies,new=tracks,
                                old_ids=[x['person_id'] for x in old_tracks],
                                target=selector.target_id,mode=selector.mode))
            if frames%40==0:print('PROGRESS',frames,tracker.metrics(),selector.mode,flush=True)
        if a.sample and frames:cv2.imwrite(str(a.sample),frame)
        elapsed=time.monotonic()-start
        result=dict(elapsed_seconds=elapsed,frames=frames,wall_fps=frames/elapsed,
                    observed_capture_fps=(seq-first_seq)/elapsed,
                    valid_percent=100*valid/max(frames,1),new=tracker.metrics(),
                    old_public_ids_created=old.next_id-1 if old else None,
                    bodies_frames=sum(bool(r['boxes']) for r in records),stages=summarize(times),
                    last_target=selector.target_id,last_mode=selector.mode,
                    note='Same input comparison, no ground truth; ID counts alone are not accuracy.')
        a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
        a.output.with_suffix('.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf8')
        print(json.dumps(result,indent=2),flush=True)
    finally:camera.close();detector.close()


if __name__=='__main__':main()
