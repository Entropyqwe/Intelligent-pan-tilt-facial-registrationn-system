#ifndef STEERING_PROTOCOL_H
#define STEERING_PROTOCOL_H
#include <stdint.h>
#define STEER_MIN_US 1200
#define STEER_CENTER_US 1500
#define STEER_MAX_US 1800
#define STEER_TIMEOUT_MS 500u
#define STEER_LOSS_HOLD_MS 3000u
typedef struct {
    uint32_t seq, target_id, age_ms;
    int32_t bearing_mdeg;
    uint16_t servo_us;
    uint8_t valid, source;
} SteerPacket;
int Steer_Parse(const char *line, SteerPacket *out);
/* Same portable watchdog state is exercised by host tests and firmware. */
typedef struct {
    uint32_t last_ms, loss_ms;
    uint16_t demand;
    uint8_t received, tracking, holding;
} SteerState;
void Steer_Accept(SteerState *s, const SteerPacket *p, uint32_t now);
uint16_t Steer_Demand(SteerState *s, uint32_t now);
#endif
