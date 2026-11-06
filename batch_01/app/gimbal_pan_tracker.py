"""Frame-clocked pan control: filtered absolute bearing, bounded lead and PD rate.

Applied PWM is a command estimate, not a measured SG90 shaft position. This
controller never receives predicted-only or invalid visual observations.
"""
import math


def clamp(value,low,high):
    return max(low,min(high,value))


def validate_tracking(c):
    if not isinstance(c,dict) or c.get('mode')!='filtered_pd':
        raise ValueError('Invalid pan tracking mode')
    limits={
        'hold_px':(0,8),'resume_px':(1,16),'filter_tau_s':(.01,.3),
        'velocity_tau_s':(.05,.5),'prediction_s':(0,.15),
        'max_prediction_deg':(0,2),'max_target_rate_deg_s':(1,80),
        'kp_near':(.5,10),'kp_far':(.5,10),'kd':(0,1),
        'max_rate_deg_s':(1,40),'max_accel_deg_s2':(20,400),
        'min_command_step_us':(1,6),'reset_gap_s':(.15,.4),
    }
    for key,(lo,hi) in limits.items():
        value=c.get(key)
        if (not isinstance(value,(int,float)) or isinstance(value,bool)
                or not math.isfinite(value) or not lo<=value<=hi):
            raise ValueError('Invalid pan tracking '+key)
    if c['resume_px']<=c['hold_px'] or c['kp_far']<c['kp_near']:
        raise ValueError('Invalid pan tracking hysteresis/gains')


class PanTracker:
    def __init__(self,config):
        validate_tracking(config)
        self.c=dict(config)
        self.reset()

    def reset(self):
        self.stamp=None
        self.target_id=None
        self.filtered=None
        self.target_rate=0.
        self.command_rate=0.
        self.last_pose=None
        self.held=True
        self.last_pulse=None
        self.last_info={}

    def update(self,*,captured,target_id,now,error_deg,error_px,capture_pan_deg,
               applied_us,to_angle,to_pulse,minimum,maximum):
        c=self.c
        if self.stamp is not None and captured<=self.stamp:
            return self.last_pulse,dict(self.last_info)
        new=(self.stamp is None or target_id!=self.target_id
             or captured-self.stamp>c['reset_gap_s'])
        dt=.05 if new else clamp(captured-self.stamp,.01,.2)
        measured=capture_pan_deg+error_deg
        current=to_angle(applied_us)
        if new:
            self.filtered=measured
            self.target_rate=0.
            self.command_rate=0.
            self.last_pose=current
            self.held=True
        # Filter in chassis bearing coordinates. Filtering image x alone would
        # mistake the camera's own rotation for movement of the target.
        factor=clamp(abs(error_px)/80.,0.,1.)
        tau=c['filter_tau_s']*(1-.65*factor)
        alpha=dt/(tau+dt)
        previous=self.filtered
        self.filtered+=alpha*(measured-self.filtered)
        slope=clamp((self.filtered-previous)/dt,-c['max_target_rate_deg_s'],c['max_target_rate_deg_s'])
        va=dt/(c['velocity_tau_s']+dt)
        self.target_rate+=va*(slope-self.target_rate)
        if abs(error_px)>c['resume_px'] and self.target_rate*error_deg<0:
            # A reversal must not keep leading in the old direction.
            self.target_rate=0.
        if abs(error_px)<=c['hold_px']:
            self.held=True
        elif abs(error_px)>=c['resume_px']:
            self.held=False
        pose_rate=0. if new else clamp((current-self.last_pose)/dt,-60.,60.)
        lead=0.
        if not self.held and abs(error_px)>2*c['resume_px']:
            horizon=min(c['prediction_s'],max(0.,now-captured)+.5*dt)
            lead=clamp(self.target_rate*horizon,-c['max_prediction_deg'],c['max_prediction_deg'])
        aim=self.filtered+lead
        if self.held:
            # Freeze the last demanded position, never integrate box jitter or
            # roll back to a pre-capture PWM while the subject is centered.
            pulse=applied_us if new else self.last_pulse
            self.command_rate=0.
            self.target_rate=0.
            state='HOLD'
        else:
            gain=c['kp_near']+(c['kp_far']-c['kp_near'])*factor
            desired=clamp(gain*(aim-current)-c['kd']*pose_rate,
                          -c['max_rate_deg_s'],c['max_rate_deg_s'])
            # Real observed reversal: brake first, without a stale momentum tail.
            if self.command_rate*desired<0:self.command_rate=0.
            accel=c['max_accel_deg_s2']*dt
            self.command_rate+=clamp(desired-self.command_rate,-accel,accel)
            demand=to_pulse(current+self.command_rate*dt)
            pulse=round(clamp(demand,minimum,maximum))
            previous_pulse=applied_us if new else self.last_pulse
            if abs(pulse-previous_pulse)<c['min_command_step_us'] and abs(error_px)<2*c['resume_px']:
                pulse=previous_pulse
            state='TRACK'
        limited=pulse in (minimum,maximum)
        if limited:
            # No integral/residual is accumulated beyond the mechanical limit.
            self.command_rate=0.
        self.stamp=captured;self.target_id=target_id
        self.last_pose=current;self.last_pulse=int(pulse)
        self.last_info=dict(pan_state=state,pan_at_limit=limited,
            pan_limit_side=('MIN' if pulse==minimum else 'MAX') if limited else None,
            pan_command_us=int(pulse),pan_min_us=minimum,pan_max_us=maximum,
            pan_error_px=error_px,pan_filtered_bearing_deg=self.filtered,
            pan_lead_deg=lead,pan_target_rate_deg_s=self.target_rate,
            pan_command_rate_deg_s=self.command_rate)
        return self.last_pulse,dict(self.last_info)
