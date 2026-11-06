#include "gimbal_control.h"
#include <string.h>
#include <stdlib.h>
/* G1 is additive. S1/A1/D1/M1 keep their original fields and checksum. */
static int hex_digit(char c) {
    if(c>='0' && c<='9') return c-'0';
    if(c>='A' && c<='F') return c-'A'+10;
    if(c>='a' && c<='f') return c-'a'+10;
    return -1;
}
int Gimbal_Parse(const char *line, GimbalPacket *out) {
    const char *p,*star; uint32_t v[6]; uint8_t sum=0;
    unsigned i; int hi,lo;
    if(!line || !out || strncmp(line,"@G1,",4)) return 0;
    for(i=1;i<100 && line[i] && line[i]!='*';++i) sum^=(uint8_t)line[i];
    if(i==100 || line[i]!='*') return 0;
    star=line+i;
    if(!star[1] || !star[2])return 0;
    hi=hex_digit(star[1]);lo=hex_digit(star[2]);
    if(hi<0 || lo<0 || ((hi<<4)|lo)!=sum)return 0;
    p=star+3;if(*p=='\r')++p;if(*p=='\n')++p;if(*p)return 0;
    p=line+4;
    for(i=0;i<6;++i) {
        uint64_t value=0; unsigned digits=0;
        while(*p>='0' && *p<='9') {
            if(++digits>10)return 0;
            value=value*10u+(unsigned)(*p++-'0');
            if(value>4294967295ULL)return 0;
        }
        if(!digits || *p!=(i==5?'*':','))return 0;
        ++p;v[i]=(uint32_t)value;
    }
    if(p!=star+1 || !v[0] || v[1]>1 || v[4]>1 || v[5]>999999)return 0;
    if(v[2]<GIMBAL_PAN_MIN_US || v[2]>GIMBAL_PAN_MAX_US)return 0;
    if(v[3]<GIMBAL_MIN_US || v[3]>GIMBAL_MAX_US)return 0;
    if(v[1] && v[5]>400)return 0;
    if(!v[1] && (v[2]!=1500 || v[3]!=1500 || v[4]))return 0;
    out->seq=v[0];out->valid=(uint8_t)v[1];out->pan_us=(uint16_t)v[2];
    out->tilt_us=(uint16_t)v[3];out->motion=(uint8_t)v[4];out->age_ms=v[5];
    return 1;
}
void Gimbal_Init(GimbalState *s) {
    memset(s,0,sizeof(*s));
    s->pan_us=s->tilt_us=s->pan_demand=s->tilt_demand=1500;
}
static unsigned gap(unsigned a,unsigned b) { return a>b?a-b:b-a; }
void Gimbal_Accept(GimbalState *s, const GimbalPacket *p, uint32_t now) {
    Gimbal_Update(s,now);
    if(!p->valid && s->valid) {
        s->lost=1;s->loss_ms=now;
        s->pan_demand=s->pan_us;s->tilt_demand=s->tilt_us;
    }
    if(p->valid) {
        s->pan_demand=p->pan_us;s->tilt_demand=p->tilt_us;s->lost=0;
    }
    s->seq=p->seq;s->valid=p->valid;s->motion=p->motion;
    s->received=1;s->received_ms=now;
    if(!p->valid || gap(s->pan_us,s->pan_demand)>6 ||
                       gap(s->tilt_us,s->tilt_demand)>6) {
        s->stable=0;s->settled=0;
    }
}
static uint16_t step(uint16_t current,uint16_t target,uint32_t limit) {
    uint32_t delta=gap(current,target);
    if(delta>limit)delta=limit;
    return (uint16_t)(current<target?current+delta:current-delta);
}
void Gimbal_Update(GimbalState *s, uint32_t now) {
    uint32_t dt=(uint32_t)(now-s->step_ms);
    if(!s->received || (uint32_t)(now-s->received_ms)>=GIMBAL_TIMEOUT_MS) {
        s->valid=0;s->motion=0;s->lost=0;
        s->pan_demand=s->tilt_demand=1500;
    } else if(s->lost && (uint32_t)(now-s->loss_ms)>=3000u) {
        s->pan_demand=s->tilt_demand=1500;s->lost=0;
    }
    if(dt>=20u) {
        if(dt>40u)dt=40u;
        s->pan_us=step(s->pan_us,s->pan_demand,dt*600u/1000u);
        s->tilt_us=step(s->tilt_us,s->tilt_demand,dt*600u/1000u);
        s->step_ms=now;
    }
    if(s->valid && gap(s->pan_us,s->pan_demand)<=6 &&
                    gap(s->tilt_us,s->tilt_demand)<=6) {
        if(!s->stable) {s->stable=1;s->stable_ms=now;}
        s->settled=(uint32_t)(now-s->stable_ms)>=GIMBAL_SETTLE_MS;
    } else {s->stable=0;s->settled=0;}
}
uint8_t Gimbal_MotionAllowed(const GimbalState *s,uint32_t now,uint32_t visual_seq) {
    return s->received && s->valid && s->motion && s->settled &&
           s->seq==visual_seq && (uint32_t)(now-s->received_ms)<GIMBAL_TIMEOUT_MS;
}
uint8_t Gimbal_RangeReady(const GimbalState *s,uint32_t sample_ms,uint32_t now) {
    /* At least 35ms after settling: discard an ECHO triggered before settle.
       Unsigned ordering remains valid across the millisecond counter wrap. */
    uint32_t since=(uint32_t)(sample_ms-s->stable_ms);
    return s->valid && s->settled &&
           since<0x80000000u && since>=GIMBAL_SETTLE_MS+35u &&
           (uint32_t)(now-sample_ms)<250u;
}
