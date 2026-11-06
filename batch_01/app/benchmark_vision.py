"""Benchmark actual frames on the deployment host; no serial or motor output."""
import argparse
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from core import Gallery
from vision import FaceVision
from body_vision import PersonVision


def summary(values):
    return dict(mean_ms=round(float(np.mean(values)), 2),
                p50_ms=round(float(np.percentile(values, 50)), 2),
                p95_ms=round(float(np.percentile(values, 95)), 2),
                max_ms=round(float(np.max(values)), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--face-size", type=int, choices=(160,224,320), default=320)
    ap.add_argument("--face-every", type=int, default=1,
                    help="1 = full face pipeline each frame (acquisition); 3 approximates locked cadence")
    ap.add_argument("--output", type=Path, default=Path("logs/benchmark_vision.json"))
    args = ap.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.face_every < 1 or not 1 <= args.threads <= 64:
        ap.error("frames and face-every must be positive; warmup must be nonnegative")
    root = Path(__file__).resolve().parent
    gallery = Gallery(root/"data/gallery.json")
    face = FaceVision(root, threads=args.threads, detector_size=args.face_size)
    person = PersonVision(root)
    camera, video = None, None
    image = cv2.imread(str(args.image)) if args.image else None
    if args.image and image is None:
        ap.error(f"Cannot read image: {args.image}")
    try:
        if args.camera is not None:
            from main import LatestCamera
            camera = LatestCamera(args.camera)
        elif args.video:
            video = cv2.VideoCapture(str(args.video))
            if not video.isOpened():
                ap.error(f"Cannot open video: {args.video}")
        timings, face_times, active_face_times, body_times, ages, counts = [], [], [], [], [], []
        previous_seq = 0
        for i in range(args.frames+args.warmup):
            if i == args.warmup:
                measured_started = time.monotonic()
                measured_camera_seq = previous_seq
            captured = None
            if camera:
                deadline = time.monotonic()+5
                while True:
                    frame, captured, seq = camera.latest()
                    if camera.failed:
                        raise RuntimeError("Camera read failed")
                    if frame is not None and seq != previous_seq:
                        previous_seq = seq
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("No new camera frame within 5 seconds")
                    time.sleep(.005)
            elif video:
                ok, frame = video.read()
                if not ok:
                    break
            else:
                frame = image
            started = time.monotonic()
            faces = face.process(frame, gallery) if i % args.face_every == 0 else []
            face_done = time.monotonic()
            bodies = person.process(frame)
            done = time.monotonic()
            if i >= args.warmup:
                timings.append((done-started)*1000)
                face_times.append((face_done-started)*1000)
                if i % args.face_every == 0:
                    active_face_times.append((face_done-started)*1000)
                body_times.append((done-face_done)*1000)
                counts.append((len(faces), len(bodies)))
                if captured:
                    ages.append((done-captured)*1000)
        if not timings:
            raise RuntimeError("No frames measured; provide a longer video or reduce warmup")
        wall_seconds = done-measured_started
        report = dict(host=dict(system=platform.system(), machine=platform.machine(),
                                python=platform.python_version(), opencv=cv2.__version__),
                      frames=len(timings), image_size=[frame.shape[1], frame.shape[0]],
                      face_every=args.face_every, threads=cv2.getNumThreads(), face_size=args.face_size,
                      pipeline=summary(timings),
                      face=summary(face_times), person=summary(body_times),
                      face_checked_frames=len(active_face_times),
                      face_when_checked=summary(active_face_times) if active_face_times else None,
                      compute_fps=round(1000/float(np.mean(timings)), 2),
                      loop_fps=round(len(timings)/wall_seconds, 2),
                      camera_received_fps=(round((previous_seq-measured_camera_seq)/wall_seconds, 2)
                                           if camera else None),
                      max_detected_faces=max(x[0] for x in counts),
                      max_detected_persons=max(x[1] for x in counts),
                      camera_age=summary(ages) if ages else None,
                      fraction_over_400ms=round(float(np.mean(np.asarray(ages or timings)>400)), 4),
                      note="Compute timing excludes drawing, serial, tracking and heartbeat. "
                           "Loop FPS also includes waiting for new frames and camera copies, not the GUI. "
                           "Camera received FPS counts host reads, not sensor exposure events. "
                           "Face mean_ms is amortized over all frames; face_when_checked is per face check. "
                           "Camera age includes inference, but excludes sensor exposure latency. "
                           "Image/video results do not measure live camera age or identity-switch rate.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Saved", args.output)
    finally:
        if camera:
            camera.close()
        if video:
            video.release()


if __name__ == "__main__":
    main()
