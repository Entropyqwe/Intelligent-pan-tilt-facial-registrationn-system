#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "gimbal_control.h"
#include "vehicle_control.h"
static void frame(const char *body,char *out) {
    unsigned sum=0;const char *p=body;
    while(*p)sum^=(unsigned char)*p++;
    sprintf(out,"@%s*%02X\n",body,sum);
}
int main(void) {
    GimbalPacket p={0};GimbalState g;VehicleState v;SteerPacket vision={0};
    char line[128];unsigned t;
    frame("G1,1,1,1600,1400,0,20",line);assert(Gimbal_Parse(line,&p));
    assert(p.pan_us==1600 && p.tilt_us==1400 && !p.motion);
    line[10]^=1;assert(!Gimbal_Parse(line,&p));
    /* Pan candidate expands independently. Tilt keeps its original limits. */
    frame("G1,1,1,1000,1500,0,0",line);assert(Gimbal_Parse(line,&p));
    frame("G1,1,1,2000,1500,0,0",line);assert(Gimbal_Parse(line,&p));
    frame("G1,1,1,999,1500,0,0",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,1,1,2001,1500,0,0",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,1,1,1500,1299,0,0",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,1,1,1500,1701,0,0",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,1,1,1500,1300,0,0",line);assert(Gimbal_Parse(line,&p));
    frame("G1,1,1,1500,1700,0,0",line);assert(Gimbal_Parse(line,&p));
    frame("G1,1,0,1500,1500,1,0",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,1,1,1500,1500,1,401",line);assert(!Gimbal_Parse(line,&p));
    frame("G1,4294967296,1,1500,1500,0,0",line);assert(!Gimbal_Parse(line,&p));
    assert(!Gimbal_Parse("@G1,",&p));assert(!Gimbal_Parse("@G1,*",&p));
    Gimbal_Init(&g);assert(!Gimbal_MotionAllowed(&g,0,0));
    frame("G1,1,1,1500,1500,1,0",line);assert(Gimbal_Parse(line,&p));
    Gimbal_Accept(&g,&p,100);Gimbal_Update(&g,100);Gimbal_Update(&g,280);
    assert(g.settled && Gimbal_MotionAllowed(&g,280,1));
    assert(!Gimbal_MotionAllowed(&g,280,2));
    assert(!Gimbal_RangeReady(&g,300,310));
    assert(Gimbal_RangeReady(&g,320,330));
    assert(!Gimbal_MotionAllowed(&g,600,1));
    Gimbal_Update(&g,600);assert(!g.valid && !g.settled);
    /* Hold applied pose for 3s while invalid heartbeats continue. */
    Gimbal_Init(&g);p.valid=1;p.motion=0;p.pan_us=1600;p.tilt_us=1400;
    for(t=100;t<=500;t+=20){Gimbal_Accept(&g,&p,t);Gimbal_Update(&g,t);}
    assert(g.pan_us==1600 && g.tilt_us==1400);
    p.valid=0;p.pan_us=p.tilt_us=1500;
    for(t=520;t<3520;t+=100){Gimbal_Accept(&g,&p,t);Gimbal_Update(&g,t);assert(g.pan_us==1600);}
    Gimbal_Accept(&g,&p,3520);Gimbal_Update(&g,3540);assert(g.pan_us<1600);
    /* Candidate endpoints slew at the original rate; heartbeat timeout centers. */
    Gimbal_Init(&g);p.valid=1;p.motion=0;p.pan_us=1000;p.tilt_us=1500;
    for(t=20;t<=1000;t+=20){
        unsigned before=g.pan_us;
        Gimbal_Accept(&g,&p,t);Gimbal_Update(&g,t);
        assert(before>=g.pan_us && before-g.pan_us<=12u);
    }
    assert(g.pan_us==1000 && g.tilt_us==1500);
    p.pan_us=2000;
    for(t=1020;t<=3000;t+=20){
        unsigned before=g.pan_us;
        Gimbal_Accept(&g,&p,t);Gimbal_Update(&g,t);
        assert(g.pan_us>=before && g.pan_us-before<=12u);
    }
    assert(g.pan_us==2000 && g.tilt_us==1500);
    for(t=3500;t<=5000;t+=20)Gimbal_Update(&g,t);
    assert(g.pan_us==1500 && !g.valid && !g.motion);
    /* Startup gates and stop/rearm remain in effect with the new aim gate. */
    Vehicle_Init(&v);vision.valid=1;vision.target_id=1;vision.source=2;
    v.aim_permit=0;
    for(t=0;t<=1000;t+=10) {
        Vehicle_SetVisual(&v,&vision,t);Vehicle_SetDistance(&v,600,1,t);Vehicle_Update(&v,t);
        assert(v.mode==VEHICLE_STOP_AIM && !v.left_duty);
    }
    v.aim_permit=1;
    for(t=1010;t<=2500;t+=10) {
        Vehicle_SetVisual(&v,&vision,t);Vehicle_SetDistance(&v,600,1,t);Vehicle_Update(&v,t);
    }
    assert(v.left_duty==300);
    v.aim_permit=0;Vehicle_Update(&v,2501);assert(!v.left_duty);
    v.aim_permit=1;Vehicle_SetDistance(&v,300,1,2502);Vehicle_Update(&v,2502);
    assert(v.mode==VEHICLE_STOP_NEAR && !v.left_duty);
    /* Unsigned tick rollover preserves gate ordering. */
    Gimbal_Init(&g);p.valid=1;p.motion=1;p.pan_us=p.tilt_us=1500;
    Gimbal_Accept(&g,&p,0xffffff00u);Gimbal_Update(&g,0xffffff00u);Gimbal_Update(&g,0xffffffb4u);
    assert(Gimbal_MotionAllowed(&g,0xffffffb4u,p.seq));
    assert(Gimbal_RangeReady(&g,0xffffffe0u,0x10u));
    puts("gimbal_native: PASS");return 0;
}
