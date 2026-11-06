"""Lightweight upper-body framing from an observed person box.

The detector does not provide anatomical waist/head keypoints. The upper part
of its box is a framing heuristic, never a body measurement or a range input.
All vertical pixel thresholds are normalized to a 480-pixel image height.
"""
import math


def clamp(value, low, high):
    return max(low, min(high, value))


def validate_framing(c):
    if not isinstance(c, dict) or c.get('mode') != 'upper_body':
        raise ValueError('Invalid tilt framing mode')
    limits = {
        'upper_box_fraction': (.4, .75), 'center_y_fraction': (.3, .55),
        'head_margin_fraction': (.06, .2), 'max_head_y_fraction': (.2, .4),
        'large_box_start_fraction': (.4, .7), 'large_box_full_fraction': (.65, .95),
        'hold_px': (2, 10), 'resume_px': (4, 20), 'filter_tau_s': (.04, .35),
        'kp': (.5, 6), 'max_rate_deg_s': (1, 25),
        'max_accel_deg_s2': (20, 200), 'reset_gap_s': (.15, .4),
    }
    for key, (low, high) in limits.items():
        value = c.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError('Invalid tilt framing ' + key)
    if (c['resume_px'] <= c['hold_px']
            or c['large_box_full_fraction'] <= c['large_box_start_fraction']
            or c['max_head_y_fraction'] <= c['head_margin_fraction']):
        raise ValueError('Invalid tilt framing thresholds')


def framing_target(box, image_height, config):
    """Return observed and desired head/top heights; no extra inference.

    Small boxes: place the estimated upper portion near the vertical center.
    Large boxes: progressively reserve headroom and most of the view below it.
    A bottom-truncated box cannot reveal where the actual waist lies.
    """
    _, y, _, height = box
    visible_top = clamp(y, 0., float(image_height))
    visible_bottom = clamp(y + height, visible_top, float(image_height))
    fraction = (visible_bottom - visible_top) / image_height
    c = config
    desired = clamp(c['center_y_fraction'] - .5*c['upper_box_fraction']*fraction,
                    c['head_margin_fraction'], c['max_head_y_fraction'])
    close = clamp((fraction-c['large_box_start_fraction']) /
                  (c['large_box_full_fraction']-c['large_box_start_fraction']), 0., 1.)
    desired += close*(c['head_margin_fraction']-desired)
    return visible_top, desired*image_height


class TiltFramer:
    def __init__(self, config):
        validate_framing(config)
        self.c = dict(config)
        self.reset()

    def reset(self):
        self.stamp = self.target_id = self.filtered = None
        self.command_rate = 0.
        self.held = True
        self.last_pulse = None
        self.last_info = {}

    def update(self, *, captured, target_id, error_deg, error_px, capture_tilt_deg,
               applied_us, to_angle, to_pulse, minimum, maximum):
        if self.stamp is not None and captured <= self.stamp:
            return self.last_pulse, dict(self.last_info)
        c = self.c
        new = (self.stamp is None or target_id != self.target_id
               or captured-self.stamp > c['reset_gap_s'])
        dt = .05 if new else clamp(captured-self.stamp, .01, .2)
        measured = capture_tilt_deg-error_deg  # positive image y needs pitch down
        if new:
            self.filtered = measured
            self.command_rate = 0.
            self.held = True
        alpha = dt/(c['filter_tau_s']+dt)
        self.filtered += alpha*(measured-self.filtered)
        if abs(error_px) <= c['hold_px']:
            self.held = True
        elif abs(error_px) >= c['resume_px']:
            self.held = False
        if self.held:
            pulse = applied_us if new else self.last_pulse
            self.command_rate = 0.
            state = 'HOLD'
        else:
            current = to_angle(applied_us)
            desired_rate = clamp(c['kp']*(self.filtered-current),
                                 -c['max_rate_deg_s'], c['max_rate_deg_s'])
            # A new observed reversal must not continue momentum in the old direction.
            if desired_rate*self.command_rate < 0:
                self.command_rate = 0.
            acceleration = c['max_accel_deg_s2']*dt
            self.command_rate += clamp(desired_rate-self.command_rate,
                                       -acceleration, acceleration)
            pulse = round(to_pulse(current+self.command_rate*dt))
            state = 'TRACK'
        pulse = int(clamp(pulse, minimum, maximum))
        at_limit = pulse in (minimum, maximum)
        if at_limit:
            self.command_rate = 0.
        self.stamp, self.target_id, self.last_pulse = captured, target_id, pulse
        self.last_info = dict(tilt_state=state, tilt_at_limit=at_limit,
            tilt_command_us=pulse, tilt_error_px=error_px,
            tilt_command_rate_deg_s=self.command_rate, tilt_framing='upper_body',
            tilt_min_us=minimum, tilt_max_us=maximum)
        return pulse, dict(self.last_info)
