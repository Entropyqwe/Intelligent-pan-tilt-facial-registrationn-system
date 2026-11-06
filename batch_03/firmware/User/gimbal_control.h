#ifndef GIMBAL_CONTROL_H
#define GIMBAL_CONTROL_H
#include <stdint.h>
#define GIMBAL_PAN_MIN_US 1000u
#define GIMBAL_PAN_MAX_US 2000u
#define GIMBAL_MIN_US 1300u
#define GIMBAL_MAX_US 1700u
#define GIMBAL_TIMEOUT_MS 500u
#define GIMBAL_SETTLE_MS 180u
typedef struct {
    uint32_t seq, age_ms;
    uint16_t pan_us, tilt_us;
    uint8_t valid, motion;
} GimbalPacket;
typedef struct {
    uint32_t seq, received_ms, step_ms, stable_ms, loss_ms;
    uint16_t pan_us, tilt_us, pan_demand, tilt_demand;
    uint8_t received, valid, motion, stable, settled, lost;
} GimbalState;
void Gimbal_Init(GimbalState *s);
int Gimbal_Parse(const char *line, GimbalPacket *out);
void Gimbal_Accept(GimbalState *s, const GimbalPacket *p, uint32_t now);
void Gimbal_Update(GimbalState *s, uint32_t now);
uint8_t Gimbal_MotionAllowed(const GimbalState *s, uint32_t now,
                              uint32_t visual_seq);
uint8_t Gimbal_RangeReady(const GimbalState *s,uint32_t sample_ms,uint32_t now);
#endif
