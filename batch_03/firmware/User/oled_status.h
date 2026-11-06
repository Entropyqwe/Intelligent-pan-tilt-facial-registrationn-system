#ifndef OLED_STATUS_H
#define OLED_STATUS_H
#include <stdint.h>
/* A token is one 8x16 cell: ASCII or one half of a 16x16 Chinese glyph. */
#define OLED_CELL_CN 0x8000u
#define CN_JIE 0u
#define CN_SHOU 1u
#define CN_PIAN 2u
#define CN_JIAO 3u
#define CN_DU 4u
#define CN_JU 5u
#define CN_LI 6u
#define CN_LI_CM 7u
#define CN_MI 8u
#define CN_WU 9u
#define CN_XIAO 10u
#define CN_ZHUANG 11u
#define CN_TAI 12u
#define CN_GEN 13u
#define CN_ZONG 14u
#define CN_DIU 15u
#define CN_SHI 16u
#define CN_DUAN 17u
#define CN_LIAN 18u
#define CN_DAI 19u
#define CN_JI 20u
typedef struct {
    uint32_t accepted_s1, now_ms, last_rx_ms, source_age_ms;
    int32_t bearing_mdeg;
    uint32_t distance_mm, distance_ms;
    uint8_t received, visual_valid, distance_valid;
} OledStatus;
void OledStatus_Build(const OledStatus *s, uint16_t cells[4][16]);
#endif
