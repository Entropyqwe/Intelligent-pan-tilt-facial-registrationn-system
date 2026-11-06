import argparse
import csv
import json
from collections import deque
from dataclasses import replace
import sys
import threading
import time
from pathlib import Path
import cv2
from core import (Gallery, Snapshot, TargetLock, WIRE_FIELDS, WIRE_FIELDS_V2,
                  control_fields, encode_packet)
from vision import FaceVision, observations
from performance import PerformanceMeter, describe
from hud import TelemetryHUD
from display_faces import DisplayFaces
from selection import automatic_candidate

ROOT = Path(__file__).resolve().parent
# 从 VS Code 运行本文件。N 录入人脸，点击已登记的人脸锁定，U 解锁，Q 退出。
# 不填 --port 时只在终端输出并保存 CSV；接电控后再指定串口。


class LatestCamera:
    def __init__(self, index, width=640, height=480, fps=30, camera_profile=None, frame_rotation=180):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        if frame_rotation not in (0,180):
            raise ValueError("frame_rotation must be 0 or 180")
        self.frame_rotation=frame_rotation
        self.camera_profile=camera_profile
        self.camera_warning=''
        self.frame = None
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        self.backend = backend
        self.cap = self._open_capture()
        if self.cap is None:
            raise RuntimeError(f"Cannot open camera {index}; try --camera 1")
        self.mutex = threading.Lock()
        self.frame, self.timestamp, self.seq = None, 0., 0
        self.failed = False
        self.disconnected = False
        self.disconnect_since = None
        self.reconnect_count = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_capture(self):
        """Open and configure the camera; return None while USB is absent."""
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (self.width,self.height):
            cap.release()
            if self.frame is None:
                raise RuntimeError(f'Camera returned {actual}, requested '
                                   f'{(self.width,self.height)}; select a supported mode')
            print(f'Camera returned {actual}, requested '
                  f'{(self.width,self.height)}; retrying', file=sys.stderr, flush=True)
            return None
        if self.camera_profile is not None:
            try:
                from camera_profiles import apply_profile
                applied=apply_profile(self.index,self.camera_profile)
                if self.camera_profile=='saved-fixed' and applied:
                    from camera_exposure import load_profile
                    from exposure_quality import verify_capture
                    saved=load_profile(self.index)
                    reason,_=verify_capture(cap,saved['test']['auto_baseline'])
                    if reason:
                        apply_profile(self.index,'quality')
                        self.camera_profile='quality'
                        self.camera_warning='固定曝光画质异常 / 已恢复自动'
                        print(self.camera_warning+': '+reason,file=sys.stderr,flush=True)
                    else:self.camera_warning=''
            except Exception as exc:
                if self.camera_profile=='saved-fixed':
                    from camera_profiles import apply_profile
                    apply_profile(self.index,'quality')
                    self.camera_profile='quality'
                    self.camera_warning='固定曝光不可用 / 已恢复自动'
                    print(self.camera_warning+': '+str(exc),file=sys.stderr,flush=True)
                    return cap
                cap.release()
                if self.frame is not None:
                    print(f'Camera profile failed during reconnect: {exc}; retrying',
                          file=sys.stderr,flush=True)
                    return None
                raise
        print(f'Camera actual resolution: {self.width}x{self.height}; reported FPS: '
              f'{cap.get(cv2.CAP_PROP_FPS):.1f} (measure actual FPS separately)',
              file=sys.stderr, flush=True)
        return cap

    def _run(self):
        try:
            while not self.stop.is_set():
                if self.cap is None:
                    if self.disconnect_since is None:
                        self.disconnect_since = time.monotonic()
                        self.disconnected = True
                        print('Camera disconnected; waiting for USB reconnect',
                              file=sys.stderr, flush=True)
                    if time.monotonic()-self.disconnect_since >= 15.0:
                        self.failed = True
                        print('Camera reconnect timeout after 15 seconds',
                              file=sys.stderr, flush=True)
                        break
                    self.cap = self._open_capture()
                    if self.cap is None:
                        self.stop.wait(.5)
                        continue
                    self.reconnect_count += 1
                    self.disconnect_since = None
                    self.disconnected = False
                    print(f'Camera reconnected (attempt {self.reconnect_count})',
                          file=sys.stderr, flush=True)
                ok, frame = self.cap.read()
                if not ok:
                    try: self.cap.release()
                    except Exception: pass
                    self.cap = None
                    continue
                # Host receive time, not the sensor's exposure timestamp.
                with self.mutex:
                    self.frame, self.timestamp = frame, time.monotonic()
                    self.seq += 1
        finally:
            if self.cap is not None:
                self.cap.release()

    def latest(self, after_seq=None):
        with self.mutex:
            frame=None if self.frame is None or self.seq==after_seq else self.frame.copy()
            timestamp,seq=self.timestamp,self.seq
        # Rotate exactly once, before NPU/association/geometry/UI. Retain the
        # original host timestamp; discarded camera frames incur no rotate cost.
        if frame is not None and self.frame_rotation==180:
            frame=cv2.rotate(frame,cv2.ROTATE_180)
        return frame,timestamp,seq

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)


class Output:
    """Independent bounded heartbeat; stale observations become invalid."""
    def __init__(self, port, log_path, max_age, deadband, protocol="F1", steering=None,
                 print_packets=True):
        self.mutex = threading.Lock()
        self.snapshot = Snapshot()
        self.stop = threading.Event()
        self.max_age, self.deadband = max_age, deadband
        self.protocol = protocol
        self.steering = steering
        self.print_packets = print_packets
        self.control_hz = float(steering.config.get('control_hz', 10.0)) if steering else 10.0
        if not 5 <= self.control_hz <= 30:
            raise ValueError('control_hz must be between 5 and 30')
        from steering import S1_FIELDS
        self.wire_fields = (S1_FIELDS if steering else
                            (WIRE_FIELDS_V2 if protocol == "F2" else WIRE_FIELDS))
        self.serial = None
        # pyserial objects are shared by the 10 Hz S1 writer and the
        # continuous telemetry reader.  Without a lock, unplug/re-enumerate
        # can race with read/write and make the distance stream disappear.
        self._serial_lock = threading.RLock()
        self.error = None
        self.serial_error = None
        self.port = port
        self._last_serial_attempt = 0.0
        self._distance = dict(mm=0, valid=False, mode='NO_DATA', age_ms=999999)
        self._distance_received = None
        self._motor = dict(valid=False)
        self._motor_received = None
        self._last_ack = None
        self._telemetry_file = None
        self._telemetry_log_lock = threading.Lock()
        self._serial_rx = bytearray()
        self._rx_thread = None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = log_path.open("w", newline="", encoding="utf-8")
            self._telemetry_file = log_path.with_suffix('.telemetry.jsonl').open('w', encoding='utf-8', buffering=1)
        except Exception:
            if self.serial:
                self.serial.close()
            raise
        self.csv = csv.writer(self.file)
        self.csv.writerow(["unix_ms", *self.wire_fields])
        self.seq = 0
        if port:
            self._try_open_serial()
            self._rx_thread = threading.Thread(target=self._receive_loop,
                                               name="stm32-telemetry-rx",
                                               daemon=True)
            self._rx_thread.start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def publish(self, snapshot):
        with self.mutex:
            self.snapshot = replace(snapshot)

    def _try_open_serial(self):
        if not self.port:
            return False
        now = time.monotonic()
        with self._serial_lock:
            if self.serial is not None:
                return True
            if now - self._last_serial_attempt < 1.0:
                return False
            self._last_serial_attempt = now
            try:
                import serial
                self.serial = serial.serial_for_url(
                    self.port, 115200, timeout=0, write_timeout=0.05)
                self.serial_error = None
                return True
            except Exception as exc:
                self.serial = None
                self.serial_error = str(exc)
                return False

    def _drop_serial(self, exc):
        self.serial_error = str(exc)
        with self._serial_lock:
            try:
                if self.serial:
                    self.serial.close()
            except Exception:
                pass
            self.serial = None

    def _receive_loop(self):
        """Continuously receive asynchronous #D1 telemetry.

        STM32 sends distance telemetry independently of S1 commands.  Reading
        only immediately after a write loses most frames, especially when the
        detector/UI is busy.  This loop drains the UART whenever data arrives
        and reconnects after USB-TTL re-enumeration.
        """
        while not self.stop.is_set():
            if self.serial is None:
                self._try_open_serial()
            read_error = None
            with self._serial_lock:
                ser = self.serial
                try:
                    pending = ser.in_waiting if ser is not None else 0
                    data = ser.read(min(pending, 4096)) if pending else b''
                except Exception as exc:
                    data = None
                    read_error = exc
            if data:
                self._consume_telemetry(data)
                self.serial_error = None
            elif data is None:
                self._drop_serial(read_error or RuntimeError('serial read failed'))
            else:
                self.stop.wait(0.01)

    def _send(self, snapshot):
        self.seq = self.seq % 4294967295 + 1
        if self.steering:
            from steering import encode_s1
            fields = self.steering.fields(snapshot, max_age=self.max_age)
            packet = encode_s1(self.seq, fields)
            if hasattr(self.steering, 'extra_packet'):
                packet = self.steering.extra_packet(self.seq) + packet
        else:
            fields = control_fields(snapshot, max_age=self.max_age, deadband=self.deadband)
            packet = encode_packet(self.seq, fields, self.protocol)
        if self.port:
            self._try_open_serial()
        written = False
        with self._serial_lock:
            ser = self.serial
            try:
                if ser:
                    if ser.write(packet) != len(packet):
                        raise RuntimeError('Incomplete S1 serial write')
                    written = True
                    self.serial_error = None
            except Exception as exc:
                # USB-TTL unplug, brownout, or driver re-enumeration must not kill
                # the body-only preview. Drop the handle and retry next tick.
                self._drop_serial(exc)
        if self.print_packets:
            print(packet.decode("ascii"), end="", flush=True)
        values = dict(fields, seq=self.seq)
        self.csv.writerow([round(time.time()*1000)] + [values[k] for k in self.wire_fields])
        self.file.flush()
        self._record_telemetry('TX', dict(seq=self.seq, **fields, serial_written=written,
                                        serial_error=self.serial_error))

    def _record_telemetry(self, kind, data):
        file = getattr(self, '_telemetry_file', None)
        if file is not None:
            record = dict(unix_ms=round(time.time()*1000), monotonic=time.monotonic(),
                          kind=kind, data=data)
            with self._telemetry_log_lock:
                file.write(json.dumps(record, ensure_ascii=False)+'\n')

    @property
    def motor(self):
        with self.mutex:
            result = dict(getattr(self, '_motor', dict(valid=False)))
            received = getattr(self, '_motor_received', None)
        age = round((time.monotonic()-received)*1000) if received is not None else 999999
        result.update(age_ms=age, valid=result.get('valid', False) and age<600)
        return result

    @property
    def distance(self):
        # This receive-only diagnostic never changes S1 or motor decisions.
        with self.mutex:
            result = dict(self._distance)
            received = self._distance_received
        if received is not None:
            result['age_ms'] = max(0, round((time.monotonic()-received)*1000))
            if result['age_ms'] >= 500:
                result.update(valid=False, mode='STALE')
        return result

    def _consume_telemetry(self, data):
        self._serial_rx.extend(data)
        while b'\n' in self._serial_rx:
            raw, _, rest = self._serial_rx.partition(b'\n')
            self._serial_rx = bytearray(rest)
            if len(raw) > 256:
                continue
            try:
                line = raw.decode('ascii').strip()
            except UnicodeDecodeError:
                continue
            self._parse_telemetry(line)
        if len(self._serial_rx) > 256:
            self._serial_rx.clear()

    def _parse_telemetry(self, line):
        # Extra diagnostics do not alter S1 parsing or control semantics.
        if line.startswith('#G2,') and self.steering and hasattr(self.steering,'accept_pose'):
            try:
                values=list(map(int,line.split(',')[1:]))
                if len(values)!=8:return
                if self.steering.accept_pose(*values):
                    self._record_telemetry('GIMBAL',dict(values=values))
            except (ValueError,TypeError):pass
            return
        if line.startswith('#A1,'):
            fields = line.split(',')
            try:
                if len(fields)!=5:return
                seq, valid, pulse, bearing = map(int, fields[1:])
                if not (1<=seq<=4294967295 and valid in (0,1) and 1200<=pulse<=1800 and abs(bearing)<=89000):return
            except ValueError:return
            with self.mutex:
                self._last_ack = dict(seq=seq, valid=valid, pulse_us=pulse, bearing_mdeg=bearing)
            self._record_telemetry('ACK', self._last_ack)
            return
        if line.startswith('#M1,'):
            fields = line.split(',')
            try:
                if len(fields)!=8:return
                seq, demand, applied, left, right, target = map(int, fields[1:7])
                if not (0<=seq<=4294967295 and 1200<=demand<=1800 and 1200<=applied<=1800
                        and all(0<=x<=999 for x in (left,right,target))):return
                mode=fields[7]
                if mode not in ('SEARCH','TRACK_READY','STOP_NEAR','STOP_VISUAL','STOP_SENSOR','START_WAIT','STOP_AIM','FAULT'):return
            except ValueError:return
            with self.mutex:
                self._motor = dict(valid=True,seq=seq,demand_us=demand,applied_us=applied,
                                   left_pwm=left,right_pwm=right,target_pwm=target,mode=mode)
                self._motor_received = time.monotonic()
            self._record_telemetry('MOTOR', self._motor)
            return
        if line.startswith('#FW,') or line.startswith('#READY,'):
            self._record_telemetry('FIRMWARE', dict(line=line))
            return
        if not line.startswith('#D1,'):
            return
        fields=line.split(',')
        if len(fields)!=4:
            return
        try:
            mm=int(fields[1]); valid=int(fields[2])
        except ValueError:
            return
        if mm<0 or mm>30000 or valid not in (0,1):
            return
        mode=fields[3]
        if not mode or len(mode)>24 or not all(c.isascii() and (c.isalnum() or c=='_') for c in mode):
            return
        with self.mutex:
            self._distance=dict(mm=mm,valid=bool(valid) and mm>0,mode=mode,age_ms=0)
            self._distance_received=time.monotonic()
        self._record_telemetry('DISTANCE', dict(mm=mm,valid=bool(valid) and mm>0,mode=mode))

    def _run(self):
        try:
            while not self.stop.is_set():
                start = time.monotonic()
                with self.mutex:
                    current = replace(self.snapshot)
                self._send(current)
                self.stop.wait(max(0., 1.0/self.control_hz-(time.monotonic()-start)))
        except Exception as exc:
            self.error = exc
            print(f"Output stopped: {exc}", file=sys.stderr)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            try:
                self._send(Snapshot(target_id=self.snapshot.target_id))
            except Exception:
                pass
            if self._rx_thread:
                self._rx_thread.join(timeout=1)
            with self._serial_lock:
                try:
                    if self.serial:
                        self.serial.close()
                except Exception:
                    pass
                self.serial = None
            self.file.close()
            if self._telemetry_file:
                self._telemetry_file.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width",type=int,default=640,help="Actual camera capture width; this camera supports 640 or 1280")
    ap.add_argument("--height",type=int,default=480,help="Actual capture height; use 480 or 720 with matching width")
    ap.add_argument("--camera-fps",type=int,default=30,help="Requested camera rate; firmware may negotiate another rate")
    ap.add_argument("--window-width",type=int,default=980,help="Preview window width; captured image retains full resolution")
    ap.add_argument("--port", default="", help="COM3 or /dev/ttyUSB0; omitted = no serial")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--target", type=int, default=0, help="Previously registered ID")
    ap.add_argument("--similarity", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-age-ms", type=int, default=400)
    ap.add_argument("--deadband", type=int, default=120)
    ap.add_argument("--body", action="store_true", help="Person detection and face-bound body continuity; outputs F2")
    ap.add_argument("--person-confidence", type=float, default=0.45)
    ap.add_argument("--face-every", type=int, default=3,
                    help="Check face every N processed frames while body is locked")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV CPU threads, 1..64; benchmark on the board")
    ap.add_argument("--face-size", type=int, choices=(160,224,320), default=224,
                    help="YuNet input size; smaller sizes can miss small or distant faces")
    ap.add_argument("--quiet-packets", action="store_true", help="Keep serial and CSV output; hide packet prints")
    ap.add_argument("--steering", action="store_true", help="S1 front-wheel steering protocol (requires --body)")
    ap.add_argument("--steering-config", default=str(ROOT/"steering_config.json"))
    ap.add_argument("--hfov", type=float, help="Explicit horizontal FOV in degrees (estimate, not camera calibration)")
    ap.add_argument("--person-backend", choices=("npu","cpu"), default="npu",
                    help="Explicit backend; NPU failures stop instead of silently using CPU")
    ap.add_argument("--npu-model", default="/opt/yolov5/model/yolov5.nb")
    ap.add_argument("--ui", choices=("telemetry","simple"), default="telemetry")
    ap.add_argument("--lock-mode", choices=("auto","manual"), default="auto",
                    help="Auto selects one unambiguous registered identity; A toggles, U pauses auto")
    ap.add_argument("--max-frames", type=int, default=0, help="Exit after this many real processed frames; 0 unlimited")
    ap.add_argument("--max-seconds", type=float, default=0., help="Stop after a bounded measurement; 0 unlimited")
    ap.add_argument("--metrics-json", type=Path, help="Write measured pipeline statistics after a normal exit")
    args = ap.parse_args()
    if min(args.width,args.height,args.camera_fps,args.window_width) <= 0:
        ap.error('Camera dimensions, rate and window width must be positive')
    if args.max_frames < 0 or not 0 <= args.max_seconds < float('inf'):
        ap.error("--max-frames and --max-seconds must be finite and >= 0")
    if not 1 <= args.threads <= 64:
        ap.error("--threads must be in 1..64")
    steering = None
    if args.steering:
        if not args.body:
            ap.error("--steering requires --body, so front-wheel control always uses body center")
        if not 1 <= args.max_age_ms <= 400:
            ap.error("S1 firmware requires --max-age-ms in 1..400; measure performance before use")
        from steering import Steering
        steering = Steering(args.steering_config, args.hfov)
        if steering.intrinsics:
            c=steering.intrinsics
            if abs(args.width/args.height-c['image_width']/c['image_height'])>.01:
                ap.error('Capture aspect ratio differs from camera calibration; use matching calibration')
        if (not steering.intrinsics and abs(args.width/args.height-4/3)>.01 and args.hfov is None):
            ap.error('Changed camera aspect ratio needs verified --hfov for this mode or matching calibration; '
                     'first preview 1280x720 without --steering, or retain --width 640 --height 480')
    if args.face_every < 1 or not 0 < args.person_confidence < 1:
        ap.error("--face-every must be >= 1 and --person-confidence must be between 0 and 1")
    gallery = Gallery(ROOT/"data"/"gallery.json")
    if args.target and args.target not in gallery.people:
        raise SystemExit("Target ID is not registered; first start with a display and press N")
    if args.no_show and not args.target and args.lock_mode == "manual":
        raise SystemExit("--no-show requires --target with a registered identity")
    vision = FaceVision(ROOT, args.similarity, args.margin,
                        threads=args.threads, detector_size=args.face_size)
    person_vision = None
    if args.body:
        from inference_worker import PersonWorker
        from body_tracker import BodyTargetLock
        person_vision = PersonWorker(ROOT,args.person_backend,args.person_confidence,args.npu_model)
        print("Person backend:",person_vision.backend_name,file=sys.stderr,flush=True)
        target = BodyTargetLock()
    else:
        target = TargetLock()
    target.select(args.target)
    try:
        output = Output(args.port, ROOT/"logs"/f"run_{time.time_ns()}.csv",
                        args.max_age_ms/1000., args.deadband, "F2" if args.body else "F1", steering=steering,
                        print_packets=not args.quiet_packets)
    except BaseException:
        if person_vision: person_vision.close()
        raise
    camera = None
    selected = []
    visible_faces = []
    display_faces = []
    visible_bodies = []
    processed = 0
    enroll = None
    enroll_requested = False
    auto_enabled = args.lock_mode == "auto"
    status = "AUTO ready | N register | click/1-9 lock | A auto/manual | U unlock | Q quit"
    window = "FaceLockCar / Guidance"
    hud = TelemetryHUD()
    face_display = DisplayFaces()
    display_bearing = steering
    if display_bearing is None:
        try:
            from steering import Steering
            display_bearing = Steering(args.steering_config,args.hfov)
            if (not display_bearing.intrinsics and args.hfov is None
                    and abs(args.width/args.height-4/3)>.01):
                display_bearing = None
        except (ValueError,OSError,KeyError):
            pass

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if args.ui == "telemetry":
                point = hud.to_image(x,y)
                if point is None: return
                x,y = point
            for face in display_faces:
                bx, by, bw, bh = face["box"]
                if bx <= x <= bx+bw and by <= y <= by+bh and face["person_id"]:
                    selected.append(face["person_id"])
                    break

    try:
        camera = LatestCamera(args.camera,args.width,args.height,args.camera_fps)
        if not args.no_show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            canvas_width=args.width+280 if args.ui=='telemetry' else args.width
            canvas_height=max(args.height+166,640) if args.ui=='telemetry' else args.height
            cv2.resizeWindow(window,args.window_width,round(args.window_width*canvas_height/canvas_width))
            cv2.moveWindow(window,10,35)
            cv2.setMouseCallback(window, on_mouse)
        seq = 0
        boot = time.monotonic()
        perf = PerformanceMeter(boot)
        last_perf = boot
        measured_frames=measured_valid=0
        pipeline_total=age_total=0.
        first_camera_seq = 0
        measured_pipeline,measured_age,quality_samples = deque(maxlen=18000),deque(maxlen=18000),[]
        measured_faces=measured_bodies=measured_head_frames=measured_body_frames=measured_identity_frames=0
        measured_modes={}
        valid_start=None
        longest_valid_seconds=0.
        loop_finished=None
        while True:
            if output.error:
                raise RuntimeError(f"Output failed: {output.error}")
            if camera.failed:
                raise RuntimeError("Camera disconnected or read failed")
            frame, captured, frame_seq = camera.latest(seq)
            if frame is None or frame_seq == seq:
                if time.monotonic()-(captured or boot) > 5:
                    raise RuntimeError("No new camera frame received for 5 seconds")
                time.sleep(0.005)
                continue
            if processed == 0:
                first_camera_seq = frame_seq
            seq = frame_seq
            start = time.monotonic()
            face_checked = (not args.body or not target.locked or enroll is not None or enroll_requested
                            or bool(selected) or processed % args.face_every == 0)
            # CPU identity and NPU person detector run concurrently on this exact frame.
            if person_vision: person_vision.submit(frame)
            visible_faces = vision.process(frame, gallery) if face_checked else []
            face_done = time.monotonic()
            if person_vision:
                visible_bodies,body_ms = person_vision.result()
            else:
                visible_bodies,body_ms = [],0.
            body_done = time.monotonic()
            if not args.no_show:
                display_faces = face_display.update(frame,visible_faces,captured,face_checked)
            processed += 1
            fresh = time.monotonic()-captured <= args.max_age_ms/1000.
            if selected:
                target.select(selected[-1])
                selected.clear()
                enroll = None
                enroll_requested = False
            if enroll_requested:
                enroll_requested = False
                if len(visible_faces) == 1 and fresh:
                    enroll = dict(samples=[],last=0.,person_id=visible_faces[0]["person_id"] or None)
                    target.select(0)
                    output.publish(Snapshot())
                    status = "Registration started: keep one clear face visible"
                else:
                    status = "Press N with exactly one visible face"
            if enroll is not None:
                # Registration is only possible with exactly one sufficiently large face.
                if len(visible_faces) != 1 or not fresh:
                    enroll = None
                    status = "Registration cancelled: one clear face required"
                else:
                    face = visible_faces[0]
                    now = time.monotonic()
                    if min(face["box"][2:]) >= 70 and now-enroll["last"] >= 0.2:
                        samples = enroll["samples"]
                        if samples and float(samples[0] @ face["feature"]) < 0.40:
                            enroll = None
                            status = "Registration cancelled: face changed; press N again"
                        else:
                            samples.append(face["feature"].copy())
                            enroll["last"] = now
                            status = f"Registering {len(samples)}/8: move head slightly"
                            if len(samples) >= 8:
                                pid = gallery.add(samples, enroll["person_id"])
                                target.select(pid)
                                enroll = None
                                status = f"Saved ID {pid}; checking identity before lock"
            if auto_enabled and not target.target_id and enroll is None and not enroll_requested and fresh:
                candidate = automatic_candidate(visible_faces,visible_bodies,args.body)
                if candidate:
                    target.select(candidate)
                    status = f"AUTO selected ID {candidate}; confirming identity and body"
            if args.body:
                snap = target.update(visible_faces if fresh else [],
                                     visible_bodies if fresh else [], captured,
                                     frame.shape[1], frame.shape[0], face_checked)
            else:
                obs = observations(visible_faces) if fresh else []
                snap = target.update(obs, captured, frame.shape[1], frame.shape[0])
            output.publish(snap)
            done = time.monotonic()
            control = control_fields(snap, now=done, max_age=args.max_age_ms/1000.,
                                     deadband=args.deadband)
            perf.add(done, frame_seq, face_checked, person_vision is not None,
                     (face_done-start)*1000., body_ms,
                     (done-start)*1000., (done-captured)*1000., bool(control['valid']))
            stats = perf.read()
            if done-boot >= 2.:
                measured_frames+=1
                measured_valid+=bool(control['valid'])
                pipeline_total+=(done-start)*1000.
                age_total+=(done-captured)*1000.
                measured_pipeline.append((done-start)*1000.)
                measured_age.append((done-captured)*1000.)
                measured_faces += face_checked
                measured_bodies += person_vision is not None
                measured_head_frames += bool(visible_faces)
                measured_body_frames += bool(visible_bodies)
                measured_identity_frames += any(f['person_id'] > 0 for f in visible_faces)
                mode_name = getattr(target,'mode','FACE')
                measured_modes[mode_name] = measured_modes.get(mode_name,0)+1
                if control['valid']:
                    valid_start = done if valid_start is None else valid_start
                    longest_valid_seconds = max(longest_valid_seconds,done-valid_start)
                else:
                    valid_start = None
                if len(quality_samples) < 30 and (not quality_samples or done-quality_samples[-1]['time'] >= 1.):
                    import numpy as np
                    sample=frame[::8,::8]
                    gray=cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
                    quality_samples.append(dict(time=done,mean_luma=float(gray.mean()),
                        color_fraction=float(np.mean(np.ptp(sample.astype(np.int16),axis=2)>10)),
                        dark_fraction=float(np.mean(gray<12)),clipped_fraction=float(np.mean(gray>245))))
            if done-last_perf >= 2.:
                print(describe(stats)+f" target={target.target_id} mode={getattr(target,'mode','FACE')}"
                      +f" heads={len(visible_faces)} bodies={len(visible_bodies)}", file=sys.stderr, flush=True)
                last_perf = done
            if not args.no_show:
                fields = control
                angle = display_bearing.bearing(snap) if display_bearing and fields["valid"] else None
                if args.ui == "telemetry":
                    canvas = hud.render(frame,display_faces,visible_bodies,target,snap,fields,stats,status,
                                        bearing=angle,backend=person_vision.backend_name if person_vision else "CPU",
                                        port=args.port,elapsed=done-boot,
                                        face_age=max(0.,captured-face_display.checked),serial_error=bool(output.error),
                                        auto_enabled=auto_enabled)
                else:
                    canvas = frame.copy()
                    for body in visible_bodies:
                        x,y,w,h=map(int,body["box"])
                        cv2.rectangle(canvas,(x,y),(x+w,y+h),(210,190,100),1)
                    for face in display_faces:
                        x,y,w,h=map(int,face["box"])
                        cv2.rectangle(canvas,(x,y),(x+w,y+h),(100,240,160),1)
                        cv2.putText(canvas,f"ID {face['person_id']} sim {face['similarity']:.2f}",
                                    (x+2,y+13),cv2.FONT_HERSHEY_SIMPLEX,.35,(100,240,160),1)
                    angle_label=f"{angle:+.1f} deg" if angle is not None else "-- deg"
                    cv2.putText(canvas,f"FPS {stats['fps']:.1f} | {angle_label}",
                                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                    cv2.putText(canvas,status,(8,frame.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1)
                cv2.imshow(window,canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("u"):
                    auto_enabled = False
                    enroll = None
                    enroll_requested = False
                    selected.clear()
                    target.select(0)
                    output.publish(Snapshot())
                    status = "Unlocked / AUTO PAUSED | A resume auto | click or 1-9 manual lock"
                if key == ord("a"):
                    auto_enabled = not auto_enabled
                    status = "AUTO enabled; current selection retained" if auto_enabled else "MANUAL mode; click or 1-9 select"
                if ord("1") <= key <= ord("9"):
                    pid = key-ord("0")
                    if pid in gallery.people:
                        selected.append(pid)
                if key == ord("n"):
                    if args.body and not face_checked:
                        enroll_requested = True
                        status = "Registration requested; checking one clear face"
                    elif len(visible_faces) == 1 and fresh:
                        enroll = dict(samples=[],last=0.,person_id=visible_faces[0]["person_id"] or None)
                        target.select(0)
                        output.publish(Snapshot())
                        status = "Registration started: keep one clear face visible"
                    else:
                        status = "Press N with exactly one visible face"
            if args.max_frames and processed >= args.max_frames:
                break
            if args.max_seconds and time.monotonic()-boot >= args.max_seconds:
                break
            elapsed = time.monotonic()-start
            if elapsed > args.max_age_ms/1000.:
                print(f"Slow inference {elapsed:.3f}s; output is invalid when stale",file=sys.stderr)
        loop_finished=time.monotonic()
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True,exist_ok=True)
            duration=loop_finished-boot
            measured_duration=max(0.,loop_finished-boot-2.)
            summary=dict(processed_frames=processed,elapsed_seconds=duration,
                         wall_fps=processed/duration if duration else 0.,
                         warmup_seconds=2.,measured_frames=measured_frames,
                         steady_wall_fps=measured_frames/measured_duration if measured_duration else 0.,
                         valid_percent=100*measured_valid/max(1,measured_frames),
                         mean_pipeline_ms=pipeline_total/max(1,measured_frames),
                         mean_host_frame_age_ms=age_total/max(1,measured_frames),
                         backend=person_vision.backend_name if person_vision else 'CPU face',
                         ui='disabled' if args.no_show else args.ui,
                         target_id=target.target_id,face_size=args.face_size,threads=args.threads,
                         final_window=stats)
            import numpy as np
            summary.update(capture_width=frame.shape[1],capture_height=frame.shape[0],
                           capture_channels=frame.shape[2] if frame.ndim==3 else 1,
                           capture_format='MJPG',requested_camera_fps=args.camera_fps,
                           observed_capture_fps=(seq-first_camera_seq)/duration,
                           measured_valid_frames=measured_valid,
                           face_checked_frames=measured_faces,body_checked_frames=measured_bodies,
                           face_check_fps=measured_faces/max(measured_duration,1e-9),
                           body_detection_fps=measured_bodies/max(measured_duration,1e-9),
                           head_detected_frames=measured_head_frames,body_detected_frames=measured_body_frames,
                           recognized_identity_frames=measured_identity_frames,
                           mode_frame_counts=measured_modes,longest_valid_seconds=longest_valid_seconds,
                           pipeline_p95_ms=float(np.percentile(measured_pipeline,95)) if measured_pipeline else None,
                           host_frame_age_p95_ms=float(np.percentile(measured_age,95)) if measured_age else None,
                           percentile_sample_frames=len(measured_pipeline),
                           quality_samples=[{k:v for k,v in sample.items() if k!='time'} for sample in quality_samples],
                           serial_enabled=bool(args.port),steering_enabled=args.steering,
                           serial_error=str(output.error) if output.error else None)
            args.metrics_json.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    except KeyboardInterrupt:
        pass
    finally:
        output.close()
        if camera:
            camera.close()
        if person_vision:
            person_vision.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(1)
