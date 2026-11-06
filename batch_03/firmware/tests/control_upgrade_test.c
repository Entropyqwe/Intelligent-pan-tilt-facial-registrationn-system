#include <assert.h>
#include <stdio.h>
#include "vehicle_control.h"

static SteerPacket packet;

static void tick(VehicleState *v, uint32_t t, uint32_t mm, uint8_t valid) {
    Vehicle_SetVisual(v,&packet,t);
    Vehicle_SetDistance(v,mm,valid,t);
    Vehicle_Update(v,t);
}

static void stable_start(VehicleState *v, uint32_t start) {
    uint32_t dt;
    Vehicle_Init(v); packet.valid=1;
    for(dt=0;dt<=200;dt+=10) tick(v,start+dt,500,1);
    assert(v->mode==VEHICLE_TRACK_READY && v->left_duty==0);
    for(dt=210;dt<=1200;dt+=10) tick(v,start+dt,500,1);
    assert(v->left_duty==300 && v->right_duty==300);
}

int main(void) {
    VehicleState v; uint32_t dt,now; uint16_t previous;
    packet.valid=1;packet.target_id=1;packet.source=2;packet.servo_us=1600;
    Vehicle_Init(&v);
    for(dt=0;dt<=190;dt+=10) {
        tick(&v,dt,500,1);assert(!v.left_duty && v.mode==VEHICLE_START_WAIT);
    }
    tick(&v,200,500,1);assert(v.mode==VEHICLE_TRACK_READY && !v.left_duty);
    previous=0;
    for(dt=210;dt<=1200;dt+=10) {
        tick(&v,dt,500,1);
        assert(v.left_duty>=previous && v.left_duty-previous<=3);
        assert(v.left_duty<=300 && v.left_duty==v.right_duty);
        previous=v.left_duty;
    }
    assert(v.left_duty==300);
    /* Once running, 301..349mm does not chatter. <=300 stops at once. */
    tick(&v,1210,301,1);assert(v.left_duty==300);
    tick(&v,1220,300,1);assert(!v.left_duty && v.mode==VEHICLE_STOP_NEAR);
    for(dt=1230;dt<1500;dt+=10) {
        tick(&v,dt,349,1);assert(!v.left_duty && v.mode==VEHICLE_START_WAIT);
    }
    for(dt=1500;dt<=1620;dt+=60)tick(&v,dt,350,1);
    assert(v.mode==VEHICLE_TRACK_READY && !v.left_duty);
    tick(&v,1630,350,1);assert(v.left_duty==3);
    tick(&v,1640,500,0);assert(!v.left_duty && v.mode==VEHICLE_STOP_SENSOR);

    /* One stored distance reading cannot satisfy the three-sample gate. */
    Vehicle_Init(&v);Vehicle_SetDistance(&v,500,1,0);
    for(dt=0;dt<250;dt+=10) {Vehicle_SetVisual(&v,&packet,dt);Vehicle_Update(&v,dt);}
    assert(v.clear_samples==1 && !v.left_duty);
    Vehicle_SetVisual(&v,&packet,250);Vehicle_Update(&v,250);
    assert(v.mode==VEHICLE_STOP_SENSOR);

    stable_start(&v,10000);packet.valid=0;tick(&v,11201,500,1);
    assert(v.mode==VEHICLE_STOP_VISUAL && !v.left_duty);
    stable_start(&v,20000);
    Vehicle_SetDistance(&v,500,1,21700);Vehicle_Update(&v,21700);
    assert(v.mode==VEHICLE_STOP_VISUAL && !v.left_duty);
    stable_start(&v,30000);
    Vehicle_SetVisual(&v,&packet,31450);Vehicle_Update(&v,31450);
    assert(v.mode==VEHICLE_STOP_SENSOR && !v.left_duty);

    /* A slow loop cannot jump to full output in a single step. */
    Vehicle_Init(&v);packet.valid=1;
    for(dt=0;dt<=200;dt+=10)tick(&v,dt,500,1);
    tick(&v,400,500,1);assert(v.left_duty<=15);

    /* Unsigned time wrap, and unchanged 3s servo visual-loss semantics. */
    stable_start(&v,0xffffff00u);
    { SteerState s={0};
      packet.valid=1;Steer_Accept(&s,&packet,0xffffff00u);
      packet.valid=0;Steer_Accept(&s,&packet,0xffffff64u);
      for(dt=100;dt<3000;dt+=100) {
        now=0xffffff64u+dt;Steer_Accept(&s,&packet,now);assert(Steer_Demand(&s,now)==1600);
      }
      now=0xffffff64u+3000u;Steer_Accept(&s,&packet,now);assert(Steer_Demand(&s,now)==1500);
      packet.valid=1;Steer_Accept(&s,&packet,5000);assert(Steer_Demand(&s,5500)==1500);
    }
    puts("PASS: stable rearm, distinct samples, exact stop/release boundaries, bounded 1s ramp, immediate invalid stop, stale link/range, delayed loop, tick wrap, original servo hold.");
    return 0;
}
