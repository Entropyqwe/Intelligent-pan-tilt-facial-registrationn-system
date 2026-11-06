#include "steering_protocol.h"
#include <stddef.h>
static int hex(char c) {
    if(c>='0' && c<='9') return c-'0';
    if(c>='A' && c<='F') return c-'A'+10;
    if(c>='a' && c<='f') return c-'a'+10;
    return -1;
}
/* Bounds each integer before multiply; reject whitespace, +, empty, overflow. */
static int number(const char **p, int64_t *value, char end) {
    uint64_t n=0; int neg=0, digits=0;
    if(**p=='-') {neg=1; ++*p;}
    while(**p>='0' && **p<='9') {
        if(++digits>10) return 0;
        n=n*10+(uint32_t)(**p-'0'); if(n>4294967295ULL) return 0;
        ++*p;
    }
    if(!digits || **p!=end) return 0;
    *value=neg?-(int64_t)n:(int64_t)n;
    ++*p;
    return 1;
}
int Steer_Parse(const char *line, SteerPacket *out) {
    const char *p, *star; uint8_t check=0; int h,l,i;
    int64_t v[7]; SteerPacket t;
    if(!line || !out || line[0]!='@') return 0;
    p=line+1;
    for(i=0; i<119 && p[i] && p[i]!='*'; ++i) check^=(uint8_t)p[i];
    if(i==119 || p[i]!='*') return 0;
    star=p+i;
    /* Short circuit checks protect all accesses for truncated strings. */
    if(!star[1] || !star[2]) return 0;
    h=hex(star[1]); l=hex(star[2]);
    if(h<0 || l<0 || (uint8_t)((h<<4)|l)!=check) return 0;
    p=star+3;
    if(*p=='\r') ++p;
    if(*p=='\n') ++p;
    if(*p!='\0') return 0;
    p=line+1;
    if(p[0]!='S' || p[1]!='1' || p[2]!=',') return 0;
    p+=3;
    for(i=0;i<7;++i) if(!number(&p,&v[i],i==6?'*':',')) return 0;
    if(p!=star+1 || v[0]<1 || v[0]>4294967295LL || v[1]<0 || v[1]>1 ||
       v[2]<0 || v[2]>2147483647 || v[3]<-89000 || v[3]>89000 ||
       v[4]<STEER_MIN_US || v[4]>STEER_MAX_US || v[5]<0 || v[5]>999999 ||
       v[6]<0 || v[6]>2) return 0;
    if(v[1]) { if(!v[2] || !v[6] || v[5]>400) return 0; }
    else if(v[3] || v[4]!=STEER_CENTER_US || v[6]) return 0;
    t.seq=(uint32_t)v[0]; t.valid=(uint8_t)v[1]; t.target_id=(uint32_t)v[2];
    t.bearing_mdeg=(int32_t)v[3]; t.servo_us=(uint16_t)v[4];
    t.age_ms=(uint32_t)v[5]; t.source=(uint8_t)v[6];
    *out=t; return 1;
}
void Steer_Accept(SteerState *s, const SteerPacket *p, uint32_t now) {
    /* Distinguish healthy invalid visual heartbeats from a dead serial link.
     * Check link timeout before accepting a packet so stale demand cannot revive.
     */
    (void)Steer_Demand(s,now);
    if(p->valid) {
        s->demand=p->servo_us; s->holding=0;
    } else if(s->tracking) {
        s->loss_ms=now; s->holding=1;
    }
    s->last_ms=now; s->received=1; s->tracking=p->valid;
    /* Repeated invalid packets must never restart the 3s hold timer. */
    if(!p->valid && !s->holding)s->demand=STEER_CENTER_US;
}
uint16_t Steer_Demand(SteerState *s, uint32_t now) {
    if(!s->received || (uint32_t)(now-s->last_ms)>=STEER_TIMEOUT_MS) {
        s->tracking=0; s->holding=0; s->demand=STEER_CENTER_US;
    } else if(s->holding && (uint32_t)(now-s->loss_ms)>=STEER_LOSS_HOLD_MS) {
        s->holding=0; s->demand=STEER_CENTER_US;
    }
    return s->demand;
}
