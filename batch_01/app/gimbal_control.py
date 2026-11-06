"""Pan/tilt centering and chassis bearing. No image-size-based ranging.

PWM-derived angles are estimates, not SG90 feedback. All samples use host
monotonic timestamps. Range is HC-SR04 cone echo, never asserted identity.
"""
from collections import deque
from dataclasses import replace
import json
import math
import threading
import time
from core import control_fields
from steering import Steering
from gimbal_pan_tracker import PanTracker, validate_tracking
from gimbal_tilt_tracker import TiltFramer, framing_target, validate_framing


def checked_frame(kind, values):
    body=kind+','+','.join(str(v) for v in values)
    checksum=0
    for byte in body.encode('ascii'):checksum^=byte
    return f'@{body}*{checksum:02X}\n'.encode('ascii')


def ray_to_chassis(x, y, pan_deg, tilt_deg):
    """Camera normalized right/down -> chassis right/forward/up."""
    p,t=math.radians(pan_deg),math.radians(tilt_deg)
    right,forward,up=x,math.cos(t)+math.sin(t)*y,math.sin(t)-math.cos(t)*y
    return (math.cos(p)*right+math.sin(p)*forward,
            -math.sin(p)*right+math.cos(p)*forward, up)


class GimbalSteering(Steering):
    def __init__(self, steering_path, config_path, hfov=None):
        super().__init__(steering_path,hfov)
        self.g=json.loads(config_path.read_text(encoding='utf-8-sig'))
        self._validate()
        self.lock=threading.RLock()
        self.poses=deque(maxlen=240)
        self.history=deque(maxlen=80)
        self.feedback=None
        self.distance_state=dict(valid=False)
        self.person_count=0
        self.distance_stamp=0.
        self.plan=dict(valid=0,pan_us=1500,tilt_us=1500,motion=0,age_ms=999999)
        self.info=dict(reason='等待新版 G2 云台遥测',motion=False)
        self.last_control_capture=None
        self.last_gimbal_time=None
        self._last_publish=0.
        self.pan_tracker=PanTracker(self.g['pan']['tracking']) if 'tracking' in self.g['pan'] else None
        self.pan_info={}
        self.tilt_framer=TiltFramer(self.g['tilt']['framing']) if 'framing' in self.g['tilt'] else None
        self.tilt_info={}

    def _validate(self):
        g=self.g
        def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        if g.get('schema')!=1 or g.get('range_mount') not in ('pan_only','chassis'):
            raise ValueError('Invalid gimbal schema/range_mount')
        if type(g.get('calibration_confirmed')) is not bool:raise ValueError('Calibration flag must be boolean')
        for name in ('pan','tilt'):
            a=g[name]
            if not all(finite(a[k]) for k in ('center_us','min_us','max_us','direction','us_per_degree')):
                raise ValueError('Invalid axis values')
            low,high=(1000,2000) if name=='pan' else (1300,1700)
            if (a['center_us']!=1500 or not low<=a['min_us']<=1500<=a['max_us']<=high
                    or a['direction'] not in (-1,1) or not 2<=a['us_per_degree']<=30):
                raise ValueError('Axis outside initial firmware limits')
            if 'deadband_deg' in a and (not finite(a['deadband_deg']) or a['deadband_deg']<0):
                raise ValueError('Invalid '+name+' deadband')
            slopes=('us_per_degree_negative','us_per_degree_positive')
            if any(k in a for k in slopes):
                if not all(k in a and finite(a[k]) and 2<=a[k]<=30 for k in slopes):
                    raise ValueError('Both measured '+name+' slopes are required')
            if 'tracking' in a:
                if name!='pan':raise ValueError('Filtered tracking is pan-only')
                validate_tracking(a['tracking'])
            if 'framing' in a:
                if name!='tilt':raise ValueError('Upper-body framing is tilt-only')
                validate_framing(a['framing'])
        for k in ('max_wheel_angle_deg','lookahead_min_m','target_box_y_fraction','center_y_fraction',
                  'deadband_deg','proportional_gain','max_tracking_deg_per_second','range_alignment_deg',
                  'max_motion_bearing_deg','max_pose_rate_for_motion_deg_per_second','distance_fresh_ms','pose_fresh_ms'):
            if not finite(g[k]) or g[k]<=0:raise ValueError('Invalid '+k)
        if (g['target_box_y_fraction']>1 or g['center_y_fraction']>1 or g['max_wheel_angle_deg']>35
            or g['max_motion_bearing_deg']>40 or g['range_alignment_deg']>8
            or g['distance_fresh_ms']>400 or g['pose_fresh_ms']>400
            or g['max_tracking_deg_per_second']>40):raise ValueError('Unsafe geometry limits')
        for k in ('camera_offset_right_m','camera_offset_forward_m','range_origin_offset_m'):
            if not finite(g[k]) or abs(g[k])>1:raise ValueError('Invalid measured offset')
        if g['front_direction'] not in (-1,1):raise ValueError('Invalid wheel direction')
        if g['calibration_confirmed']:
            if not finite(g['wheelbase_m']) or not .05<=g['wheelbase_m']<=1.5:raise ValueError('Measure wheelbase')
            if not finite(g['front_us_per_wheel_degree']) or not 1<=g['front_us_per_wheel_degree']<=50:
                raise ValueError('Measure steering linkage')

    def accept_pose(self, seq, valid, pan, tilt, settled, motion, ranged, age_ms, stamp=None):
        stamp=time.monotonic() if stamp is None else stamp
        if (not 0<=seq<=4294967295 or any(v not in (0,1) for v in (valid,settled,motion,ranged))
                or not 1000<=pan<=2000 or not 1300<=tilt<=1700 or not 0<=age_ms<=4294967295):
            return False
        item=dict(seq=seq,valid=bool(valid),pan_us=pan,tilt_us=tilt,settled=bool(settled),
                  motion=bool(motion),range_ready=bool(ranged),age_ms=age_ms,stamp=stamp)
        with self.lock:
            if self.poses and stamp<=self.poses[-1]['stamp']:return False
            self.feedback=item
            self.poses.append(item)
        return True

    def publish_frame(self,snapshot,distance,person_count):
        with self.lock:
            self.distance_state=dict(distance)
            self.distance_stamp=time.monotonic()-distance.get('age_ms',999999)/1000
            self.person_count=person_count
            self.history.append((replace(snapshot),person_count))

    def _angle(self,name,pulse):
        a=self.g[name]
        signed=(pulse-a['center_us'])/a['direction']
        key='us_per_degree_positive' if signed>=0 else 'us_per_degree_negative'
        return signed/a.get(key,a['us_per_degree'])

    def _pulse(self,name,angle):
        a=self.g[name]
        key='us_per_degree_positive' if angle>=0 else 'us_per_degree_negative'
        return a['center_us']+a['direction']*a.get(key,a['us_per_degree'])*angle

    def pose_at(self,stamp):
        with self.lock:poses=list(self.poses)
        # Never use a post-capture value as if it was the capture orientation.
        older=[p for p in poses if p['stamp']<=stamp]
        if not older:return None
        a=older[-1]
        if stamp-a['stamp']>self.g['pose_fresh_ms']/1000:return None
        newer=next((p for p in poses if p['stamp']>stamp),None)
        pan,tilt=a['pan_us'],a['tilt_us']
        if newer and newer['stamp']-a['stamp']<=.35:
            t=(stamp-a['stamp'])/(newer['stamp']-a['stamp'])
            pan+=(newer['pan_us']-pan)*t;tilt+=(newer['tilt_us']-tilt)*t
        return self._angle('pan',pan),self._angle('tilt',tilt)

    def rays(self,snapshot):
        o=snapshot.observation;w,h=snapshot.image_width,snapshot.image_height
        u=o.x+o.w/2;v=o.y+o.h*self.g['target_box_y_fraction']
        if self.intrinsics:
            import cv2
            import numpy as np
            cw,ch=self.intrinsics['image_width'],self.intrinsics['image_height']
            if abs(w/h-cw/ch)>.01:raise ValueError('Gimbal calibration aspect mismatch')
            k=self.matrix.copy();k[0,:]*=w/cw;k[1,:]*=h/ch
            points=np.array([[[u,v]],[[w/2,h*self.g['center_y_fraction']]]],np.float64)
            q=cv2.undistortPoints(points,k,self.dist).reshape(2,2)
            return float(q[0,0]),float(q[0,1]),float(q[1,1])
        fx=w/(2*math.tan(math.radians(float(self.hfov))/2))
        return (u-w/2)/fx,(v-h/2)/fx,(h*self.g['center_y_fraction']-h/2)/fx

    def angles(self,snapshot,pose=None):
        pose=self.pose_at(snapshot.captured) if pose is None else pose
        if pose is None:return None
        x,y,y0=self.rays(snapshot)
        right,forward,up=ray_to_chassis(x,y,*pose)
        if forward<=.05:return None
        return dict(bearing_deg=math.degrees(math.atan2(right,forward)),
                    error_x=math.degrees(math.atan(x)),
                    error_y=math.degrees(math.atan(y)-math.atan(y0)),
                    pan_deg=pose[0],tilt_deg=pose[1])

    def bearing(self,snapshot):
        with self.lock:
            value=self.angles(snapshot)
            return value['bearing_deg'] if value else 0.

    def framing_error(self,snapshot):
        # Separate composition from rays(): the S1 bearing and range alignment
        # retain their original reference point and geometry.
        o=snapshot.observation;w,h=snapshot.image_width,snapshot.image_height
        observed,desired=framing_target((o.x,o.y,o.w,o.h),h,self.g['tilt']['framing'])
        if self.intrinsics:
            import cv2
            import numpy as np
            k=self.matrix.copy();k[0,:]*=w/self.intrinsics['image_width'];k[1,:]*=h/self.intrinsics['image_height']
            u=o.x+o.w/2
            points=np.array([[[u,observed]],[[u,desired]]],np.float64)
            q=cv2.undistortPoints(points,k,self.dist).reshape(2,2)
            error=math.degrees(math.atan(q[0,1])-math.atan(q[1,1]))
        else:
            fx=w/(2*math.tan(math.radians(float(self.hfov))/2))
            error=math.degrees(math.atan((observed-h/2)/fx)-math.atan((desired-h/2)/fx))
        return error,(observed-desired)*480/h,desired/h

    def reset_trackers(self):
        if self.pan_tracker:self.pan_tracker.reset()
        if self.tilt_framer:self.tilt_framer.reset()
        self.pan_info={};self.tilt_info={}

    def fields(self,snapshot,now=None,max_age=.4):
        now=time.monotonic() if now is None else now
        with self.lock:return self._fields(snapshot,now,max_age)

    def _fields(self,snapshot,now,max_age):
        f=control_fields(snapshot,now=now,max_age=max_age)
        matching=[count for item,count in self.history if item.captured==snapshot.captured
                  and item.target_id==snapshot.target_id]
        person_count=matching[-1] if matching else 0
        result=dict(valid=0,target_id=snapshot.target_id,bearing_mdeg=0,
                    servo_us=1500,age_ms=min(f['age_ms'],999999),source=0)
        self.plan=dict(valid=0,pan_us=1500,tilt_us=1500,motion=0,age_ms=result['age_ms'])
        reason='视觉无效或目标丢失'
        pose=self.feedback
        if not f['valid']:
            self.reset_trackers()
            self.info=dict(reason=reason,motion=False);return result
        if pose is None or now-pose['stamp']>.35:
            self.reset_trackers()
            self.info=dict(reason='等待新版 G2 / 遥测超时',motion=False);return result
        angles=self.angles(snapshot)
        if angles is None or abs(angles['bearing_deg'])>80:
            self.reset_trackers()
            self.info=dict(reason='拍摄时刻角度不可用',motion=False);return result
        dt=.05 if self.last_control_capture is None else max(0,min(.2,snapshot.captured-self.last_control_capture))
        pan,tilt=pose['pan_us'],pose['tilt_us']
        # Integrate only once per new image; a 20Hz UART heartbeat cannot
        # repeatedly integrate an 8Hz observation. Use capture-time orientation
        # for absolute targets and current applied PWM to bound each step.
        if snapshot.captured!=self.last_control_capture:
            self.last_control_capture=snapshot.captured
            for name,error,direction in (('pan',angles['error_x'],1),('tilt',angles['error_y'],-1)):
                a=self.g[name];old=pose[name+'_us']
                if name=='pan' and self.pan_tracker:
                    o=snapshot.observation
                    error_px=(o.x+o.w/2-snapshot.image_width/2)*640/snapshot.image_width
                    pan,self.pan_info=self.pan_tracker.update(captured=snapshot.captured,
                        target_id=snapshot.target_id,now=now,error_deg=error,error_px=error_px,
                        capture_pan_deg=angles['pan_deg'],applied_us=old,
                        to_angle=lambda pulse:self._angle('pan',pulse),
                        to_pulse=lambda angle:self._pulse('pan',angle),
                        minimum=a['min_us'],maximum=a['max_us'])
                    continue
                if name=='tilt' and self.tilt_framer:
                    framing_deg,framing_px,head_y=self.framing_error(snapshot)
                    tilt,self.tilt_info=self.tilt_framer.update(captured=snapshot.captured,
                        target_id=snapshot.target_id,error_deg=framing_deg,error_px=framing_px,
                        capture_tilt_deg=angles['tilt_deg'],applied_us=old,
                        to_angle=lambda pulse:self._angle('tilt',pulse),
                        to_pulse=lambda angle:self._pulse('tilt',angle),
                        minimum=a['min_us'],maximum=a['max_us'])
                    self.tilt_info['tilt_head_target_fraction']=head_y
                    continue
                # A zero pan deadband starts correction on each newly observed
                # horizontal offset; tilt can retain its original deadband.
                deadband=a.get('deadband_deg',self.g['deadband_deg'])
                error=0 if abs(error)<=deadband else error
                wanted=angles[name+'_deg']+direction*error*self.g['proportional_gain']
                demand=self._pulse(name,wanted)
                limit=self.g['max_tracking_deg_per_second']*a['us_per_degree']*max(dt,.05)
                value=round(max(a['min_us'],min(a['max_us'],old+max(-limit,min(limit,demand-old)))))
                if name=='pan':pan=value
                else:tilt=value
            self._last_publish=now
            self._last_demands=(pan,tilt)
        else:
            pan,tilt=getattr(self,'_last_demands',(pan,tilt))
        beta=angles['bearing_deg'];distance=self.distance_state
        distance_ok=(distance.get('valid',False) and 0<=now-self.distance_stamp<=self.g['distance_fresh_ms']/1000
                     and 20<=distance.get('mm',0)<=5000)
        # Ranger stays horizontal while the camera tilts: compare horizontal
        # chassis azimuths, not the tilted camera's raw x optical angle.
        aligned=abs(beta-angles['pan_deg'])<=self.g['range_alignment_deg']
        if self.g['range_mount']=='chassis':aligned=aligned and abs(beta)<=self.g['range_alignment_deg']
        rate=999.
        if len(self.poses)>1:
            old=self.poses[-2];elapsed=pose['stamp']-old['stamp']
            if elapsed>0:rate=max(abs(self._angle(a,pose[a+'_us'])-self._angle(a,old[a+'_us']))/elapsed for a in ('pan','tilt'))
        reason='对准完成'
        if not self.g['calibration_confirmed']:reason='待标定 / 电机禁止'
        elif person_count!=1:reason='多人或无人体 / 电机禁止'
        elif not aligned:reason='超声波未对准 / 电机禁止'
        elif not distance_ok:reason='测距无效或超时'
        elif not pose['settled'] or rate>self.g['max_pose_rate_for_motion_deg_per_second']:reason='云台转动中 / 电机禁止'
        elif not pose['range_ready']:reason='等待转动后的新距离'
        elif abs(beta)>self.g['max_motion_bearing_deg']:reason='目标偏角过大 / 电机禁止'
        motion=(reason=='对准完成')
        # Without a proven measured geometry, retain bounded legacy angular
        # steering for VM-disconnected bench work; never permit motor motion.
        wheel=None
        if self.g['calibration_confirmed'] and distance_ok and aligned:
            r=max(.02,distance['mm']/1000+self.g['range_origin_offset_m'])
            right=r*math.sin(math.radians(beta))+self.g['camera_offset_right_m']
            forward=r*math.cos(math.radians(beta))+self.g['camera_offset_forward_m']
            length2=max(right*right+forward*forward,self.g['lookahead_min_m']**2)
            wheel=math.degrees(math.atan2(2*self.g['wheelbase_m']*right,length2))
            wheel=max(-self.g['max_wheel_angle_deg'],min(self.g['max_wheel_angle_deg'],wheel))
            demand=self.center+self.g['front_direction']*self.g['front_us_per_wheel_degree']*wheel
        else:demand=self.center+self.sign*self.gain*beta
        self.plan=dict(valid=1,pan_us=pan,tilt_us=tilt,motion=int(motion),age_ms=result['age_ms'])
        result.update(valid=1,bearing_mdeg=round(beta*1000),
                      servo_us=round(max(self.minimum,min(self.maximum,demand))),source=f['source'])
        self.info=dict(**angles,wheel_deg=wheel,reason=reason,motion=motion,
                       range_aligned=aligned,pose_rate=rate,pan_us=pan,tilt_us=tilt,
                       range_candidate=bool(aligned and distance_ok and pose['range_ready'] and person_count==1))
        self.info.update(self.pan_info)
        self.info.update(self.tilt_info)
        return result

    def extra_packet(self,seq):
        with self.lock:
            p=self.plan
            return checked_frame('G1',(seq,p['valid'],p['pan_us'],p['tilt_us'],p['motion'],p['age_ms']))

    def status(self):
        with self.lock:return dict(self.info)

    def compensate_tracks(self,identities,single,snapshot):
        """Apply rotational homography before association. Estimates only;
        projected boxes cannot become valid observations or new identities.
        """
        import numpy as np
        current=self.pose_at(snapshot.captured)
        previous=getattr(self,'_association_pose',None)
        self._association_pose=current
        if current is None or previous is None:return
        if max(abs(a-b) for a,b in zip(current,previous))<.05:return
        w,h=snapshot.image_width,snapshot.image_height
        if self.intrinsics:
            k=self.matrix.copy();k[0,:]*=w/self.intrinsics['image_width'];k[1,:]*=h/self.intrinsics['image_height']
        else:
            fx=w/(2*math.tan(math.radians(float(self.hfov))/2))
            k=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
        def rotation(pose):
            # Camera basis columns in chassis coordinates (right,down,forward).
            columns=[]
            for x,y in ((1,0),(0,1),(0,0)):
                columns.append(np.array(ray_to_chassis(x,y,*pose)))
            forward=columns[2]
            return np.column_stack((columns[0]-forward,columns[1]-forward,forward))
        homography=k@rotation(current).T@rotation(previous)@np.linalg.inv(k)
        def warp(box):
            x,y,bw,bh=box
            corners=np.array([[x,y,1],[x+bw,y,1],[x,y+bh,1],[x+bw,y+bh,1.]]).T
            q=homography@corners
            if np.any(q[2]<=.1):return box
            q=q[:2]/q[2:];lo=q.min(axis=1);hi=q.max(axis=1)
            return float(lo[0]),float(lo[1]),max(4.,float(hi[0]-lo[0])),max(8.,float(hi[1]-lo[1]))
        from tracking_motion import BoxMotion
        for track in identities.tracks.values():
            track.box=warp(track.box)
            # Reset image velocity after camera rotation; no invented person velocity.
            track.motion=BoxMotion(track.box)
        if single is not None and single.last_box is not None:single.last_box=warp(single.last_box)
