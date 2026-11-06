#include "oled_status.h"
#include "steering_protocol.h"
#include "vehicle_control.h"
#include <stdio.h>
#include <string.h>
static void chinese(uint16_t row[16], unsigned x, unsigned glyph) {
    row[x]=(uint16_t)(OLED_CELL_CN | (glyph << 1));
    row[x+1]=(uint16_t)(OLED_CELL_CN | (glyph << 1) | 1u);
}
static void text(uint16_t row[16], unsigned x, const char *value) {
    while(*value && x<16) row[x++]=(uint8_t)*value++;
}
void OledStatus_Build(const OledStatus *s, uint16_t cells[4][16]) {
    unsigned r,c,x; char number[16]; uint32_t age, magnitude, tenths;
    uint8_t live, target, range;
    for(r=0;r<4;++r) for(c=0;c<16;++c) cells[r][c]=' ';
    age=(uint32_t)(s->now_ms-s->last_rx_ms);
    live=(uint8_t)(s->received && age<STEER_TIMEOUT_MS);
    target=(uint8_t)(live && s->visual_valid &&
        s->source_age_ms<STEER_TIMEOUT_MS &&
        age<STEER_TIMEOUT_MS-s->source_age_ms);
    range=(uint8_t)(s->distance_valid && s->distance_mm>=20u &&
        s->distance_mm<=5000u &&
        (uint32_t)(s->now_ms-s->distance_ms)<VEHICLE_DISTANCE_TIMEOUT_MS);
    chinese(cells[0],0,CN_JIE); chinese(cells[0],2,CN_SHOU);
    cells[0][4]=':';
    sprintf(number,"%lu",(unsigned long)s->accepted_s1);text(cells[0],5,number);
    chinese(cells[1],0,CN_PIAN);chinese(cells[1],2,CN_JIAO);cells[1][4]=':';
    if(target && s->bearing_mdeg>=-89000 && s->bearing_mdeg<=89000) {
        magnitude=(uint32_t)(s->bearing_mdeg<0?-s->bearing_mdeg:s->bearing_mdeg);
        tenths=(magnitude+50u)/100u;
        sprintf(number,"%c%lu.%lu",s->bearing_mdeg<0?'-':'+',
                (unsigned long)(tenths/10u),(unsigned long)(tenths%10u));
        text(cells[1],5,number);x=5u+(unsigned)strlen(number);
        chinese(cells[1],x,CN_DU);
    } else {chinese(cells[1],5,CN_WU);chinese(cells[1],7,CN_XIAO);}
    chinese(cells[2],0,CN_JU);chinese(cells[2],2,CN_LI);cells[2][4]=':';
    if(range) {
        sprintf(number,"%lu.%lu",(unsigned long)(s->distance_mm/10u),
                (unsigned long)(s->distance_mm%10u));
        text(cells[2],5,number);x=5u+(unsigned)strlen(number);
        chinese(cells[2],x,CN_LI_CM);chinese(cells[2],x+2,CN_MI);
    } else {chinese(cells[2],5,CN_WU);chinese(cells[2],7,CN_XIAO);}
    chinese(cells[3],0,CN_ZHUANG);chinese(cells[3],2,CN_TAI);cells[3][4]=':';
    if(!s->received) {chinese(cells[3],5,CN_DAI);chinese(cells[3],7,CN_JI);}
    else if(!live) {chinese(cells[3],5,CN_DUAN);chinese(cells[3],7,CN_LIAN);}
    else if(target) {chinese(cells[3],5,CN_GEN);chinese(cells[3],7,CN_ZONG);}
    else {chinese(cells[3],5,CN_DIU);chinese(cells[3],7,CN_SHI);}
}
