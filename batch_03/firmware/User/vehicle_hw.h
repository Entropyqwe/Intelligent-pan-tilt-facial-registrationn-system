#ifndef VEHICLE_HW_H
#define VEHICLE_HW_H
#include <stdint.h>
#include "vehicle_control.h"

void VehicleHw_Init(void);
void VehicleHw_UltrasonicTask(uint32_t now_ms, VehicleState *state);
void VehicleHw_ApplyMotors(const VehicleState *state);

#endif
