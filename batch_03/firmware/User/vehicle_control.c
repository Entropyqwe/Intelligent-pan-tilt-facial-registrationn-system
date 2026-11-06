#include "vehicle_control.h"

void Vehicle_Init(VehicleState *s) {
    s->distance_mm=0; s->distance_ms=0; s->distance_valid=0;
    s->visual_valid=0; s->visual_ms=0;
    s->aim_permit=1; /* Main's gimbal gate overrides this before any output. */
    s->left_duty=0; s->right_duty=0; s->target_duty=0; s->mode=VEHICLE_SEARCH;
    s->rearm_required=1; s->clear_samples=0; s->clear_seen=0; s->visual_seen=0;
    s->clear_since_ms=0; s->clear_sample_ms=0; s->visual_since_ms=0; s->ramp_ms=0;
}

void Vehicle_SetDistance(VehicleState *s, uint32_t distance_mm,
                         uint8_t valid, uint32_t now_ms) {
    s->distance_mm=distance_mm; s->distance_valid=valid; s->distance_ms=now_ms;
}

void Vehicle_SetVisual(VehicleState *s, const SteerPacket *packet,
                       uint32_t now_ms) {
    s->visual_valid=(uint8_t)(packet && packet->valid);
    s->visual_ms=now_ms;
}

/* Every inhibit removes PWM immediately. Acceleration alone is ramped.
 * Recovery always requires new stable observations, never predicted ranges. */
static void stop_motion(VehicleState *s, VehicleMode mode, uint32_t now_ms) {
    s->left_duty=0; s->right_duty=0; s->target_duty=0; s->mode=mode;
    s->rearm_required=1; s->clear_samples=0; s->clear_seen=0; s->ramp_ms=now_ms;
}

void Vehicle_Update(VehicleState *s, uint32_t now_ms) {
    uint32_t elapsed, step;
    uint8_t visual_ok=s->visual_valid &&
        (uint32_t)(now_ms-s->visual_ms)<VEHICLE_VISUAL_TIMEOUT_MS;
    uint8_t distance_ok=s->distance_valid &&
        (uint32_t)(now_ms-s->distance_ms)<VEHICLE_DISTANCE_TIMEOUT_MS;
    if(!visual_ok) {
        s->visual_seen=0;
        stop_motion(s,VEHICLE_STOP_VISUAL,now_ms);
        return;
    }
    if(!s->visual_seen) {s->visual_seen=1; s->visual_since_ms=now_ms;}
    if(!distance_ok) {
        stop_motion(s,VEHICLE_STOP_SENSOR,now_ms);
        return;
    }
    if(s->distance_mm<=VEHICLE_STOP_MM) {
        stop_motion(s,VEHICLE_STOP_NEAR,now_ms);
        return;
    }
    if(!s->aim_permit) {
        stop_motion(s,VEHICLE_STOP_AIM,now_ms);
        return;
    }
    if(s->rearm_required) {
        if(s->distance_mm<VEHICLE_RELEASE_MM) {
            stop_motion(s,VEHICLE_START_WAIT,now_ms);
            return;
        }
        /* Repeated loop iterations must not count the same ECHO sample. */
        if(!s->clear_seen || s->distance_ms!=s->clear_sample_ms) {
            if(!s->clear_seen) s->clear_since_ms=now_ms;
            s->clear_seen=1; s->clear_sample_ms=s->distance_ms;
            if(s->clear_samples<VEHICLE_REARM_DISTANCE_SAMPLES) ++s->clear_samples;
        }
        if(s->clear_samples<VEHICLE_REARM_DISTANCE_SAMPLES ||
           (uint32_t)(now_ms-s->clear_since_ms)<VEHICLE_REARM_DISTANCE_MS ||
           (uint32_t)(now_ms-s->visual_since_ms)<VEHICLE_REARM_VISUAL_MS) {
            s->mode=VEHICLE_START_WAIT; s->ramp_ms=now_ms;
            return;
        }
        s->rearm_required=0; s->ramp_ms=now_ms;
    }
    s->target_duty=MOTOR_FOLLOW_DUTY>MOTOR_MAX_DUTY?MOTOR_MAX_DUTY:MOTOR_FOLLOW_DUTY;
    s->mode=VEHICLE_TRACK_READY;
    elapsed=(uint32_t)(now_ms-s->ramp_ms);
    if(elapsed<MOTOR_RAMP_INTERVAL_MS) return;
    /* Do not jump after a delayed main loop. Never catch up more than 50ms. */
    if(elapsed>50u) elapsed=50u;
    s->ramp_ms=now_ms;
    step=MOTOR_RAMP_PER_SECOND*elapsed/1000u;
    if(s->left_duty<s->target_duty) {
        uint32_t next=s->left_duty+step;
        s->left_duty=(uint16_t)(next>s->target_duty?s->target_duty:next);
    } else s->left_duty=s->target_duty;
    s->right_duty=s->left_duty;
}

uint8_t Vehicle_MotionAllowed(const VehicleState *s) {
    return s->mode==VEHICLE_TRACK_READY;
}

const char *Vehicle_ModeName(VehicleMode mode) {
    switch(mode) {
    case VEHICLE_TRACK_READY:return "TRACK_READY";
    case VEHICLE_STOP_NEAR:return "STOP_NEAR";
    case VEHICLE_STOP_VISUAL:return "STOP_VISUAL";
    case VEHICLE_STOP_SENSOR:return "STOP_SENSOR";
    case VEHICLE_FAULT:return "FAULT";
    case VEHICLE_START_WAIT:return "START_WAIT";
    case VEHICLE_STOP_AIM:return "STOP_AIM";
    default:return "SEARCH";
    }
}
