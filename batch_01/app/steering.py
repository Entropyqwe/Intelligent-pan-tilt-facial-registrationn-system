"""Camera bearing -> bounded FRONT-WHEEL servo demand; S1 serial protocol.

Bearing is an optical ray angle, not a measured wheel angle. Camera must be
fixed forward on chassis. No integral term: do not use this for a pan camera.
"""
import json
import math
from pathlib import Path
import time
from core import control_fields

S1_FIELDS = ('seq', 'valid', 'target_id', 'bearing_mdeg', 'servo_us', 'age_ms', 'source')


class Steering:
    def __init__(self, path, hfov=None):
        self.config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        c = self.config
        self.center = int(c['servo_center_us'])
        self.minimum, self.maximum = int(c['servo_min_us']), int(c['servo_max_us'])
        if self.center != 1500:
            raise ValueError('This firmware uses neutral 1500 us; mechanically center the horn first')
        if not 1200 <= self.minimum <= self.center <= self.maximum <= 1800:
            raise ValueError('Pulse limits must be inside firmware 1200..1800 us')
        self.sign = int(c['servo_direction'])
        self.gain = float(c['gain_us_per_degree'])
        self.deadband = float(c['bearing_deadband_deg'])
        self.tau = float(c['filter_tau_seconds'])
        self.rate = float(c['max_slew_us_per_second'])
        if (self.sign not in (-1, 1) or not all(math.isfinite(x) for x in
            (self.gain,self.deadband,self.tau,self.rate)) or self.gain < 0 or
            not 0 <= self.deadband < 30 or self.tau < 0 or self.rate <= 0):
            raise ValueError('Invalid steering settings')
        self.hfov = float(hfov) if hfov is not None else c.get('horizontal_fov_deg')
        self.intrinsics = None
        calibration = c.get('camera_calibration_file')
        if calibration:
            p = Path(calibration)
            if not p.is_absolute(): p = Path(path).resolve().parent / p
            self.intrinsics = json.loads(p.read_text(encoding='utf-8-sig'))
            import numpy as np
            self.matrix = np.array(self.intrinsics['camera_matrix'], dtype=float)
            self.dist = np.array(self.intrinsics['dist_coeffs'], dtype=float)
            if (self.matrix.shape != (3,3) or not np.isfinite(self.matrix).all()
                or not np.isfinite(self.dist).all() or self.matrix[0,0] <= 0
                or self.matrix[1,1] <= 0 or self.intrinsics['image_width'] <= 0
                or self.intrinsics['image_height'] <= 0):
                raise ValueError('Invalid camera calibration')
        elif self.hfov is None or not math.isfinite(float(self.hfov)) or not 5 < float(self.hfov) < 170:
            raise ValueError('Set measured horizontal_fov_deg or camera_calibration_file in steering_config.json; see integration guide')
        self.last_time = None
        self.last_target = None
        self.filtered = 0.
        self.current = float(self.center)
        self.last_valid_time = None
        self.recovery_memory = 3.0
        self.observation_lost = False
        self.last_capture = None

    def bearing(self, snapshot):
        o = snapshot.observation
        if o is None: return 0.
        w, h = snapshot.image_width, snapshot.image_height
        if w <= 0 or h <= 0: raise ValueError('Invalid image dimensions')
        u, v = o.x + o.w/2, o.y + o.h/2
        if self.intrinsics:
            import cv2
            import numpy as np
            cw, ch = self.intrinsics['image_width'], self.intrinsics['image_height']
            # A crop/aspect change requires recalibration, not just focal scaling.
            if abs(w/h-cw/ch) > .01: raise ValueError('Camera aspect ratio differs from calibration')
            k = self.matrix.copy()
            k[0,:] *= w/cw
            k[1,:] *= h/ch
            ray = cv2.undistortPoints(np.array([[[u,v]]],np.float64), k, self.dist)
            return math.degrees(math.atan(float(ray[0,0,0])))
        fx = w/(2*math.tan(math.radians(float(self.hfov))/2))
        # Pinhole approximation; assumes principal point is image center.
        return math.degrees(math.atan((u-w/2)/fx))

    def fields(self, snapshot, now=None, max_age=.4):
        now = time.monotonic() if now is None else now
        f = control_fields(snapshot, now=now, max_age=max_age)
        r = dict(valid=0,target_id=snapshot.target_id,bearing_mdeg=0,
                 servo_us=self.center, age_ms=min(f['age_ms'],999999),source=0)
        if not f['valid']:
            self.observation_lost = True
            # Wire packet stays invalid/neutral so the motor stops immediately.
            # Keep filter memory for a short reacquisition, not valid old data.
            same_target=snapshot.target_id>0 and snapshot.target_id==self.last_target
            if (not same_target or self.last_valid_time is None
                    or now-self.last_valid_time>=self.recovery_memory):
                self.last_time = None
                self.last_target = None
                self.filtered = 0.
                self.current = float(self.center)
            return r
        reacquired = (snapshot.target_id == self.last_target
                      and self.last_valid_time is not None
                      and 0 <= now-self.last_valid_time < self.recovery_memory
                      and (self.observation_lost or
                           (self.last_capture is not None and
                            snapshot.captured-self.last_capture > max_age)))
        if self.last_target != snapshot.target_id:
            self.last_time = None
            self.filtered = 0.
            self.current = float(self.center)
        self.last_target = snapshot.target_id
        self.last_valid_time = now
        self.last_capture = snapshot.captured
        self.observation_lost = False
        bearing = self.bearing(snapshot)
        if not math.isfinite(bearing) or abs(bearing) > 89:
            raise ValueError('Invalid optical bearing')
        dt = .1 if self.last_time is None else max(0., min(.2,now-self.last_time))
        self.last_time = now
        wanted = 0. if abs(bearing) <= self.deadband else bearing
        alpha = 1. if self.tau == 0 else 1-math.exp(-dt/self.tau)
        # On verified reacquisition use the CURRENT optical ray. Do not blend
        # the previous side into the new demand. The MCU still slews real PWM.
        if reacquired:self.filtered = wanted
        else:self.filtered += alpha*(wanted-self.filtered)
        demand = self.center + self.sign*self.gain*self.filtered
        demand = max(self.minimum,min(self.maximum,demand))
        step = self.rate*dt
        if reacquired:self.current = demand
        else:self.current += max(-step,min(step,demand-self.current))
        r.update(valid=1,bearing_mdeg=round(bearing*1000),servo_us=round(self.current),source=f['source'])
        return r


def encode_s1(seq, fields):
    values = dict(fields,seq=seq)
    body = 'S1,'+','.join(str(values[k]) for k in S1_FIELDS)
    check = 0
    for b in body.encode('ascii'): check ^= b
    return f'@{body}*{check:02X}\n'.encode('ascii')
