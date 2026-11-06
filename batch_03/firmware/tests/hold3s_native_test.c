
#include <assert.h>
#include <stdio.h>
#include "vehicle_control.h"
int main(void){
 SteerState s={0};VehicleState v;SteerPacket p={0};
 p.valid=1;p.target_id=1;p.source=2;p.servo_us=1600;
 assert(Steer_Demand(&s,0)==1500);
 Steer_Accept(&s,&p,100);assert(Steer_Demand(&s,101)==1600);
 p.valid=0;p.source=0;p.servo_us=1500;
 Steer_Accept(&s,&p,200);assert(!s.tracking && s.holding);
 for(unsigned t=300;t<3200;t+=100){
  Steer_Accept(&s,&p,t);assert(Steer_Demand(&s,t)==1600);
 }
 Steer_Accept(&s,&p,3200);assert(Steer_Demand(&s,3200)==1500 && !s.holding);
 p.valid=1;p.source=2;p.servo_us=1400;
 Steer_Accept(&s,&p,4000);assert(Steer_Demand(&s,4499)==1400);
 assert(Steer_Demand(&s,4500)==1500);
 p.valid=0;p.source=0;p.servo_us=1500;
 Steer_Accept(&s,&p,4501);assert(Steer_Demand(&s,4501)==1500);
 p.valid=1;p.source=2;p.servo_us=1600;
 Steer_Accept(&s,&p,5000);p.valid=0;p.source=0;p.servo_us=1500;
 Steer_Accept(&s,&p,5100);
 p.valid=1;p.source=2;p.servo_us=1450;
 Steer_Accept(&s,&p,5200);assert(!s.holding && Steer_Demand(&s,5200)==1450);
 /* Unsigned tick wrap with valid serial heartbeats. */
 SteerState wrap={0};Steer_Accept(&wrap,&p,0xffffff00u);
 p.valid=0;p.source=0;p.servo_us=1500;
 Steer_Accept(&wrap,&p,0xffffff64u);
 for(unsigned delta=100;delta<3000;delta+=100){
  unsigned stamp=0xffffff64u+delta;Steer_Accept(&wrap,&p,stamp);
  assert(Steer_Demand(&wrap,stamp)==1450);
 }
 Steer_Accept(&wrap,&p,0xffffff64u+3000u);
 assert(Steer_Demand(&wrap,0xffffff64u+3000u)==1500);
 Vehicle_Init(&v);p.valid=1;Vehicle_SetVisual(&v,&p,100);
 Vehicle_SetDistance(&v,500,1,100);Vehicle_Update(&v,100);
 assert(!Vehicle_MotionAllowed(&v) && v.mode==VEHICLE_START_WAIT);
 p.valid=0;Vehicle_SetVisual(&v,&p,101);Vehicle_Update(&v,101);
 assert(!Vehicle_MotionAllowed(&v) && !v.left_duty && !v.right_duty);
 puts("PASS: exact 3000ms visual-loss hold; repeated invalid does not extend; immediate motor invalid; 500ms link timeout; recovery; tick wrap.");
 return 0;
}
