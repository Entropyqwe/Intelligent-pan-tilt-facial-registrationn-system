#include "oled_status.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static uint16_t c[4][16];
static void ascii(unsigned r,unsigned col,const char *s) {
    while(*s) assert(c[r][col++]==(unsigned char)*s++);
}
static void glyph(unsigned r,unsigned col,unsigned g) {
    assert(c[r][col]==(OLED_CELL_CN|(g<<1)));
    assert(c[r][col+1]==(OLED_CELL_CN|(g<<1)|1));
}
static void bounds(void) {
    unsigned r,x;for(r=0;r<4;++r)for(x=0;x<16;++x)
        assert((c[r][x]&OLED_CELL_CN)?((c[r][x]&0x7FFFu)<42):
            (c[r][x]>=' ' && c[r][x]<='~'));
}
int main(int argc,char **argv) {
    OledStatus s={0};unsigned i;
    OledStatus_Build(&s,c);glyph(1,5,CN_WU);glyph(2,5,CN_WU);glyph(3,5,CN_DAI);bounds();
    s.received=1;s.visual_valid=1;s.now_ms=1000;s.last_rx_ms=980;
    s.source_age_ms=20;s.accepted_s1=123;s.bearing_mdeg=-12540;
    s.distance_valid=1;s.distance_ms=970;s.distance_mm=865;
    OledStatus_Build(&s,c);ascii(0,5,"123");ascii(1,5,"-12.5");
    ascii(2,5,"86.5");glyph(3,5,CN_GEN);bounds();
    if(argc>1) {FILE *f=fopen(argv[1],"wb");assert(f);assert(fwrite(c,1,sizeof(c),f)==sizeof(c));fclose(f);}
    s.accepted_s1=0xFFFFFFFFu;s.bearing_mdeg=89000;s.distance_mm=5000;
    OledStatus_Build(&s,c);ascii(0,5,"4294967295");ascii(1,5,"+89.0");ascii(2,5,"500.0");bounds();
    s.bearing_mdeg=-49;OledStatus_Build(&s,c);ascii(1,5,"-0.0");bounds();
    s.bearing_mdeg=-89000;OledStatus_Build(&s,c);ascii(1,5,"-89.0");bounds();
    s.now_ms=1220;OledStatus_Build(&s,c);glyph(2,5,CN_WU);bounds();
    s.now_ms=1460;s.source_age_ms=20;OledStatus_Build(&s,c);glyph(1,5,CN_WU);bounds();
    s.now_ms=1480;OledStatus_Build(&s,c);glyph(3,5,CN_DUAN);bounds();
    s.now_ms=1000;s.visual_valid=0;OledStatus_Build(&s,c);glyph(1,5,CN_WU);glyph(3,5,CN_DIU);
    s.visual_valid=1;s.source_age_ms=500;OledStatus_Build(&s,c);glyph(1,5,CN_WU);
    s.source_age_ms=0;s.now_ms=20;s.last_rx_ms=0xFFFFFFF0u;
    s.distance_ms=0xFFFFFFF0u;OledStatus_Build(&s,c);glyph(3,5,CN_GEN);bounds();
    for(i=0;i<=5001;++i) {s.distance_mm=i;OledStatus_Build(&s,c);bounds();}
    puts("OLED: startup, accepted count, signed bearing, range, invalid/stale data, wraparound and all 0..5001mm bounds PASS");
    return 0;
}
