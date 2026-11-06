"""NPU body tracking with optional sparse named faces and offline greetings."""
import argparse
from collections import deque
import json
import signal
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from main import LatestCamera, Output
from core import Snapshot, control_fields
from steering import Steering
from inference_worker import PersonWorker
from body_identity import BodyIdentities, BodySelection
from single_person import SinglePersonPolicy
from body_hud import BodyHUD
from performance import PerformanceMeter
from dataclasses import replace

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--camera-fps',type=int,default=30)
    p.add_argument('--camera-profile',choices=('quality','saved-fixed','fast-experimental','unchanged'),default='quality',
                   help='quality retains auto exposure; fast-experimental trades image quality for measured speed')
    p.add_argument('--frame-rotation',type=int,choices=(0,180),default=180,
                   help='Rotate raw frames before analysis and display; default 180 degrees')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--person-backend',choices=('npu','cpu'),default='npu',
                   help='人体检测后端；香橙派默认npu，Windows对照测试使用cpu')
    p.add_argument('--person-confidence',type=float,default=.35)
    p.add_argument('--person-low-confidence',type=float,default=.10,
                   help='Low-score candidates retain existing tracks only')
    p.add_argument('--reid-model',type=Path,help='Optional verified OSNet x0.25 ONNX; sparse CPU, detector stays NPU')
    p.add_argument('--port',default='')
    p.add_argument('--steering-config',type=Path,default=ROOT/'steering_config.json')
    p.add_argument('--hfov',type=float)
    p.add_argument('--gimbal-config',type=Path,help='Enable pan/tilt G1 extension; S1 fields unchanged')
    p.add_argument('--lock-mode',choices=('auto','manual'),default='auto')
    p.add_argument('--single-person-slot',choices=('on','off'),default='on',
                   help='单人连续确认后编号归一为1；关闭则使用原多人编号')
    p.add_argument('--no-show',action='store_true')
    p.add_argument('--inference-overlap',choices=('on','off'),default='on',
                   help='Overlap next-frame NPU inference with current-frame UI; bounded to one request')
    p.add_argument('--max-seconds',type=float,default=0.)
    p.add_argument('--metrics-json',type=Path)
    p.add_argument('--trace-jsonl',type=Path,help='Optional local detection/association metadata for validation')
    p.add_argument('--evidence-image',type=Path,help='Optional local screenshot; never stored by default')
    p.add_argument('--faces',choices=('on','off'),default='off')
    p.add_argument('--face-hz',type=float,default=2.,help='Sparse CPU recognition rate, independent of body NPU')
    p.add_argument('--gallery',type=Path,default=ROOT/'data'/'named_faces.json')
    p.add_argument('--speech',choices=('on','off'),default='off')
    p.add_argument('--welcome-cooldown',type=float,default=60.)
    p.add_argument('--follow-face',type=int,default=0,help='Optional strict registered face ID; 0 keeps body selection')
    a=p.parse_args()
    if not .2<=a.face_hz<=5 or not 10<=a.welcome_cooldown<=3600 or a.follow_face<0:
        p.error('Invalid face frequency / welcome cooldown / identity')
    if (a.speech=='on' or a.follow_face) and a.faces!='on':
        p.error('Speech or strict identity following requires --faces on')
    if a.camera_profile=='saved-fixed' and (a.width,a.height)!=(640,480):
        p.error('Saved fixed exposure was tested at 640x480; use that resolution or retest')
    if (a.width,a.height) not in ((640,480),(1280,720)) or not 1<=a.threads<=8 or not .1<=a.person_confidence<=.9 or not 0<a.camera_fps<=120 or not 0<=a.max_seconds<float('inf'):
        p.error('Invalid camera/threads/confidence/duration')
    if not .01<=a.person_low_confidence<a.person_confidence:
        p.error('Low confidence must be positive and below high confidence')
    if a.gimbal_config:
        from gimbal_control import GimbalSteering
        steering=GimbalSteering(a.steering_config,a.gimbal_config,a.hfov)
    else:
        steering=Steering(a.steering_config,a.hfov)
    if steering.intrinsics:
        c=steering.intrinsics
        if abs(a.width/a.height-c['image_width']/c['image_height'])>.01:p.error('Aspect differs from calibration')
    elif a.port and abs(a.width/a.height-4/3)>.01 and a.hfov is None:
        p.error('720p serial steering requires verified --hfov or matching camera calibration')
    show_bearing=not(abs(a.width/a.height-4/3)>.01 and a.hfov is None and not steering.intrinsics)
    cv2.setNumThreads(a.threads)
    worker=camera=output=trace=face_worker=speech=bindings=welcome=None
    face_labels={}
    stop=False
    runtime_error=None
    def shutdown(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    encoder=None
    if a.reid_model:
        from body_reid import BodyReID
        encoder=BodyReID(a.reid_model)
    selector=BodySelection();identities=BodyIdentities(high_threshold=a.person_confidence,
                              low_threshold=a.person_low_confidence,reid=encoder);hud=BodyHUD()
    auto=a.lock_mode=='auto';clicked=[];tracks=[]
    single=SinglePersonPolicy(high=a.person_confidence) if a.single_person_slot=='on' else None
    def mouse(event,x,y,*_):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        point=hud.to_image(x,y)
        if point is None:return
        px,py=point
        for track in tracks:
            if track['person_id']<=0:continue
            x,y,w,h=track['box']
            if x<=px<=x+w and y<=py<=y+h:clicked.append(track['person_id']);break
    # Keep the Qt lookup key ASCII; use a separate localized display title.
    window='BodyLock / NPU Guidance'
    samples=deque(maxlen=18000);quality=[];modes={}
    count=measured=valid_count=0;sum_compute=sum_age=0.;seq=first_seq=0
    pending_frame=None
    valid_start=None;longest=0.;stats={};snapshot=Snapshot()
    start=time.monotonic();last_print=start
    try:
        if a.faces=='on':
            from named_faces import NamedGallery
            from face_worker import FaceWorker
            from face_link import FaceBindings
            from welcome_speech import SpeechPlayer,WelcomeGate
            gallery=NamedGallery(a.gallery)
            if a.follow_face and a.follow_face not in gallery.people:
                raise ValueError('指定人脸ID尚未登记；请先在控制中心登记姓名')
            bindings=FaceBindings(gallery)
            face_worker=FaceWorker(ROOT,gallery,a.threads,a.face_hz)
            speech=SpeechPlayer(a.speech=='on')
            welcome=WelcomeGate(a.welcome_cooldown)
        if a.trace_jsonl:
            a.trace_jsonl.parent.mkdir(parents=True,exist_ok=True)
            trace=a.trace_jsonl.open('w',encoding='utf-8',buffering=1)
        worker=PersonWorker(ROOT,a.person_backend,a.person_low_confidence)
        output=Output(a.port,ROOT/'logs'/f'body_{time.time_ns()}.csv',.4,120,'F2',steering,print_packets=False)
        camera=LatestCamera(a.camera,a.width,a.height,a.camera_fps,camera_profile=a.camera_profile,frame_rotation=a.frame_rotation)
        print(f'BODY TRACKING / FACES={a.faces} / SPEECH={a.speech} / BACKEND='+worker.backend_name,flush=True)
        print(f'FRAME_ROTATION={a.frame_rotation} / analysis=display / project={ROOT}',flush=True)
        if not a.no_show:
            # Native pixels: avoids resizing small text through Qt and VNC.
            cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowTitle(window,'人体跟踪与舵机转向协同')
            cv2.moveWindow(window,10,34);cv2.setMouseCallback(window,mouse)
        start=time.monotonic();last_print=start;meter=PerformanceMeter(start)
        while not stop:
            # Serial unplug/driver re-enumeration is shown in the HUD and retried
            # by Output; it must not terminate the body-only preview.
            if camera.failed:raise RuntimeError('Camera failed after 15 s USB reconnect timeout')
            if pending_frame is not None:
                frame,captured,frame_seq,t=pending_frame
                pending_frame=None
                submitted=True
            else:
                frame,captured,frame_seq=camera.latest(seq)
                t=time.monotonic()
                submitted=False
            if frame is None:
                if camera.disconnected:
                    # LatestCamera is attempting to recover a transient USB
                    # reset.  Do not mistake the old frame timestamp for a
                    # stalled camera and exit before the reconnect timeout.
                    time.sleep(.05)
                    continue
                if time.monotonic()-(captured or start)>5:raise RuntimeError('Camera stalled')
                time.sleep(.001);continue
            if not count:first_seq=frame_seq
            seq=frame_seq
            if not submitted:worker.submit(frame)
            bodies,body_ms=worker.result();matched=time.monotonic()
            fresh=matched-captured<=.4
            if a.gimbal_config:
                steering.compensate_tracks(identities,single,
                    Snapshot(captured=captured,image_width=frame.shape[1],image_height=frame.shape[0]))
            if single:
                tracks=single.update(frame,bodies if fresh else [],captured,identities)
            else:
                tracks=identities.update(frame,bodies if fresh else [],captured)
            if bindings:
                bindings.observe(tracks if fresh else [],captured)
                face_result=face_worker.poll()
                if face_result:bindings.accept(face_result,time.monotonic())
                face_labels=bindings.labels(time.monotonic())
                face_worker.submit(frame,captured,bindings.tokens(),time.monotonic())
                welcome.update({row['identity_id']:row['name'] for row in face_labels.values()
                                if row['verified']},time.monotonic(),speech.enqueue)
            if clicked:
                selector.select(clicked[-1]);clicked.clear()
            if auto and single:
                single.auto_select(selector)
            elif auto and not selector.target_id:
                confirmed=[track for track in tracks if track['confirmed'] and track['person_id']>0]
                # Auto selection is only unambiguous with one visible detection.
                if len(confirmed)==1 and len(bodies)==1:selector.select(confirmed[0]['person_id'])
            if auto and a.follow_face:
                qualified=[tid for tid,row in face_labels.items() if row['identity_id']==a.follow_face]
                if len(qualified)==1 and selector.target_id!=qualified[0]:selector.select(qualified[0])
            snapshot=selector.update(tracks if fresh else [],captured,a.width,a.height)
            if a.follow_face and face_labels.get(selector.target_id,{}).get('identity_id')!=a.follow_face:
                # Revoke control immediately; a previous face frame is never a steering observation.
                snapshot=replace(snapshot,valid=False,source=0,observation=None)
                selector.locked=False;selector.mode='VERIFY'
            if a.gimbal_config:
                count_people=single.count if single else len(bodies)
                steering.publish_frame(snapshot,output.distance,count_people)
            output.publish(snapshot);done=time.monotonic();fields=control_fields(snapshot,now=done)
            if trace:
                trace.write(json.dumps(dict(t=captured-start,seq=seq,boxes=bodies,tracks=tracks,
                            target=selector.target_id,mode=selector.mode,valid=int(fields['valid']),
                            gimbal=steering.status() if a.gimbal_config else None))+'\n')
            count+=1;appearance_ms=(done-matched)*1000;compute_ms=(done-t)*1000;age_ms=(done-captured)*1000
            meter.add(done,seq,False,True,0.,body_ms,compute_ms,age_ms,bool(fields['valid']))
            stats=meter.read();stats['appearance_ms']=appearance_ms
            stats.update(worker.last_timing)
            if done-start>=2:
                measured+=1;valid_count+=bool(fields['valid']);sum_compute+=compute_ms;sum_age+=age_ms
                samples.append((compute_ms,age_ms,body_ms,appearance_ms,
                                worker.last_timing.get('pre_ms',0.),worker.last_timing.get('npu_and_copy_ms',0.),
                                worker.last_timing.get('post_ms',0.),0.))
                modes[selector.mode]=modes.get(selector.mode,0)+1
                if fields['valid']:
                    if valid_start is None:valid_start=done
                    longest=max(longest,done-valid_start)
                else:valid_start=None
                if len(quality)<60 and (not quality or done-quality[-1]['time']>=1):
                    small=frame[::8,::8];gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
                    quality.append(dict(time=done,mean_luma=float(gray.mean()),color_fraction=float(np.mean(np.ptp(small.astype(np.int16),axis=2)>10)),dark_fraction=float(np.mean(gray<12)),clipped_fraction=float(np.mean(gray>245))))
            if done-last_print>=2:
                print(f"[BODY] FPS={stats['fps']:.2f} CAMERA={stats['camera_fps']:.2f} NPU={body_ms:.1f}ms MATCH={appearance_ms:.1f}ms TARGET={selector.target_id} MODE={selector.mode} bodies={len(bodies)} tracks={len(tracks)} IDS={identities.counters['public_ids_created']} VALID={stats['valid_percent']:.0f}%",flush=True)
                last_print=done
            # Start only the latest available next frame before drawing this
            # frame. The worker has exactly one request; its detections remain
            # attached to that frame and its original host timestamp.
            if a.inference_overlap=='on':
                next_frame,next_stamp,next_seq=camera.latest(seq)
                if next_frame is not None:
                    next_started=time.monotonic()
                    worker.submit(next_frame)
                    pending_frame=(next_frame,next_stamp,next_seq,next_started)
            canvas=None
            if not a.no_show:
                ui_started=time.monotonic()
                angle=steering.bearing(snapshot) if show_bearing and fields['valid'] else None
                if a.gimbal_config:
                    stats['gimbal']=steering.status()
                    angle=stats['gimbal'].get('bearing_deg') if fields['valid'] else None
                canvas=hud.render(frame,tracks,selector,stats,angle,auto,a.port,done-start,
                                  distance=getattr(output,'distance',None),
                                  motor=getattr(output,'motor',None),
                                  serial_error=getattr(output,'serial_error',None))
                if bindings:
                    from face_hud import overlay_faces
                    overlay_faces(canvas,frame,tracks,face_labels,
                                  face_worker.error,speech.status,a.follow_face)
                if getattr(camera,'camera_warning',''):
                    from body_hud import write,AMBER
                    write(canvas,camera.camera_warning,22,82,.43,AMBER)
                cv2.imshow(window,canvas);key=cv2.waitKey(1)&255
                if done-start>=2 and samples:
                    samples[-1]=(*samples[-1][:-1],(time.monotonic()-ui_started)*1000)
                if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
                if key==ord('u'):selector.select(0);auto=False
                if key==ord('a'):auto=not auto
                if ord('1')<=key<=ord('9'):
                    pid=key-ord('0')
                    if any(track['person_id']==pid for track in tracks):clicked.append(pid)
            if a.max_seconds and time.monotonic()-start>=a.max_seconds:break
        duration=time.monotonic()-start;measured_duration=max(duration-2,1e-9)
        if a.evidence_image and count:
            a.evidence_image.parent.mkdir(parents=True,exist_ok=True)
            image=frame if a.no_show else canvas
            if image is not None:cv2.imwrite(str(a.evidence_image),image)
        if a.metrics_json:
            result=dict(algorithm='body-gimbal-named-face-v1' if bindings else ('body-gimbal-v1' if a.gimbal_config else 'body-byte-association-v2'),face_models_loaded=bool(face_worker and face_worker.loaded),registration_required=bool(a.follow_face),
                gimbal=steering.status() if a.gimbal_config else None,
                biometric_identity=bool(bindings),gender_inference=False,processed_frames=count,elapsed_seconds=duration,
                face=dict(enabled=bool(bindings),hz=a.face_hz,completed=face_worker.completed if face_worker else 0,
                          last_ms=face_worker.last_ms if face_worker else 0,error=face_worker.error if face_worker else '',
                          follow_identity=a.follow_face,speech=speech.status if speech else 'disabled'),
                warmup_seconds=2,measured_frames=measured,steady_wall_fps=measured/measured_duration,
                observed_capture_fps=(seq-first_seq)/max(duration,1e-9),body_detection_fps=measured/measured_duration,
                valid_percent=100*valid_count/max(measured,1),longest_valid_seconds=longest,
                mean_pipeline_ms=sum_compute/max(measured,1),mean_host_frame_age_ms=sum_age/max(measured,1),
                pipeline_p95_ms=float(np.percentile([s[0] for s in samples],95)) if samples else None,
                host_frame_age_p95_ms=float(np.percentile([s[1] for s in samples],95)) if samples else None,
                frame_rotation_degrees=a.frame_rotation,capture_width=a.width,capture_height=a.height,capture_channels=3,capture_format='MJPG',
                camera_profile_requested=a.camera_profile,
                camera_profile_effective=getattr(camera,'camera_profile',a.camera_profile),
                camera_warning=getattr(camera,'camera_warning',''),
                backend=worker.backend_name,inference_overlap=a.inference_overlap,
                ui='disabled' if a.no_show else 'native960x660',target_id=selector.target_id,
                mode_frame_counts=modes,quality_samples=[{k:v for k,v in s.items() if k!='time'} for s in quality],
                serial_enabled=bool(a.port),serial_error=(output.serial_error if output else None),
                steering_enabled=True,source_for_valid_body=2,threads=a.threads,person_confidence=a.person_confidence,
                person_low_confidence=a.person_low_confidence,tracking=identities.metrics(),
                single_person=single.metrics() if single else None,
                stages={name:dict(mean_ms=float(np.mean([s[index] for s in samples])),
                                 p95_ms=float(np.percentile([s[index] for s in samples],95)))
                        for index,name in enumerate(('pipeline','frame_age','detection','association',
                                                      'preprocess','npu_and_copy','postprocess','ui'))} if samples else {},
                final_window=stats)
            a.metrics_json.parent.mkdir(parents=True,exist_ok=True)
            a.metrics_json.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print('METRICS_SAVED',a.metrics_json,flush=True)
    except Exception as exc:
        # Keep VNC launches diagnosable: an uncaught exception otherwise looks
        # like a window that simply flashed and disappeared.
        runtime_error=exc
        print(f'程序异常退出: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
    finally:
        if output:output.close()
        if camera:camera.close()
        if worker:worker.close()
        if face_worker:face_worker.close()
        if speech:speech.close()
        if trace:trace.close()
        if not a.no_show:cv2.destroyAllWindows()


if __name__=='__main__':main()
