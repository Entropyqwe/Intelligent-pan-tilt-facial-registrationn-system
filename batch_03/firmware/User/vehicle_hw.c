#include "vehicle_hw.h"
#include "stm32f10x.h"

/* Recommended mapping; verify board silk and your carrier before wiring:
 * PA6/PA7 TIM3_CH1/CH2 -> TB6612 PWMA/PWMB
 * PB12/PB13 -> AIN1/AIN2; PB14/PB15 -> BIN1/BIN2; PB10 -> STBY
 * PB6 -> HC-SR04 TRIG; PB7 -> HC-SR04 ECHO through a 3.3V-safe interface.
 */
#define MOTOR_TIMER_PERIOD 999u
#define ULTRASONIC_PERIOD_MS 60u
#define ULTRASONIC_TIMEOUT_US 30000u

static uint32_t last_trigger_ms;
static volatile uint8_t echo_state;
static volatile uint16_t echo_start;
static volatile uint16_t echo_width;
static volatile uint8_t echo_ready;

static void echo_capture_polarity(uint16_t polarity) {
    /* TIM4_CH2 uses CCER.CC2P (bit 5) for edge polarity. The SPL version
     * shipped with this project has no TIM_ICPolarityConfig() helper. */
    if(polarity==TIM_ICPolarity_Falling) TIM4->CCER |= TIM_CCER_CC2P;
    else TIM4->CCER &= (uint16_t)~TIM_CCER_CC2P;
}

static void motor_dir(uint8_t a1, uint8_t a2, uint8_t b1, uint8_t b2) {
    if(a1) GPIO_SetBits(GPIOB,GPIO_Pin_12); else GPIO_ResetBits(GPIOB,GPIO_Pin_12);
    if(a2) GPIO_SetBits(GPIOB,GPIO_Pin_13); else GPIO_ResetBits(GPIOB,GPIO_Pin_13);
    if(b1) GPIO_SetBits(GPIOB,GPIO_Pin_14); else GPIO_ResetBits(GPIOB,GPIO_Pin_14);
    if(b2) GPIO_SetBits(GPIOB,GPIO_Pin_15); else GPIO_ResetBits(GPIOB,GPIO_Pin_15);
}

static void pwm(uint16_t left, uint16_t right) {
    TIM_SetCompare1(TIM3,left>MOTOR_TIMER_PERIOD?MOTOR_TIMER_PERIOD:left);
    TIM_SetCompare2(TIM3,right>MOTOR_TIMER_PERIOD?MOTOR_TIMER_PERIOD:right);
}

void VehicleHw_Init(void) {
    GPIO_InitTypeDef g;
    TIM_TimeBaseInitTypeDef t;
    TIM_OCInitTypeDef o;
    TIM_ICInitTypeDef ic;
    NVIC_InitTypeDef n;
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA|RCC_APB2Periph_GPIOB,ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3|RCC_APB1Periph_TIM4,ENABLE);

    g.GPIO_Pin=GPIO_Pin_6|GPIO_Pin_7; g.GPIO_Mode=GPIO_Mode_AF_PP;
    g.GPIO_Speed=GPIO_Speed_50MHz; GPIO_Init(GPIOA,&g);
    g.GPIO_Pin=GPIO_Pin_10|GPIO_Pin_12|GPIO_Pin_13|GPIO_Pin_14|GPIO_Pin_15;
    g.GPIO_Mode=GPIO_Mode_Out_PP; GPIO_Init(GPIOB,&g);
    g.GPIO_Pin=GPIO_Pin_6; g.GPIO_Mode=GPIO_Mode_Out_PP; GPIO_Init(GPIOB,&g);
    g.GPIO_Pin=GPIO_Pin_7; g.GPIO_Mode=GPIO_Mode_IN_FLOATING; GPIO_Init(GPIOB,&g);

    TIM_TimeBaseStructInit(&t); t.TIM_Prescaler=71; t.TIM_Period=MOTOR_TIMER_PERIOD;
    TIM_TimeBaseInit(TIM3,&t);
    TIM_OCStructInit(&o); o.TIM_OCMode=TIM_OCMode_PWM1; o.TIM_OutputState=TIM_OutputState_Enable;
    o.TIM_OCPolarity=TIM_OCPolarity_High; o.TIM_Pulse=0;
    TIM_OC1Init(TIM3,&o); TIM_OC2Init(TIM3,&o);
    TIM_OC1PreloadConfig(TIM3,TIM_OCPreload_Enable); TIM_OC2PreloadConfig(TIM3,TIM_OCPreload_Enable);
    TIM_ARRPreloadConfig(TIM3,ENABLE); TIM_Cmd(TIM3,ENABLE);

    /* TIM4 free-runs at 1 MHz for HC-SR04 echo timing. PB7 is also
     * TIM4_CH2, so use hardware input capture instead of polling the pin.
     * This prevents a short ECHO pulse being missed while USART/OLED code
     * is running. */
    TIM_TimeBaseStructInit(&t); t.TIM_Prescaler=71; t.TIM_Period=65535;
    TIM_TimeBaseInit(TIM4,&t);
    TIM_ICStructInit(&ic);
    ic.TIM_Channel=TIM_Channel_2;
    ic.TIM_ICPolarity=TIM_ICPolarity_Rising;
    ic.TIM_ICSelection=TIM_ICSelection_DirectTI;
    ic.TIM_ICPrescaler=TIM_ICPSC_DIV1;
    ic.TIM_ICFilter=2;
    TIM_ICInit(TIM4,&ic);
    TIM_ITConfig(TIM4,TIM_IT_CC2,ENABLE);
    n.NVIC_IRQChannel=TIM4_IRQn; n.NVIC_IRQChannelPreemptionPriority=2;
    n.NVIC_IRQChannelSubPriority=0; n.NVIC_IRQChannelCmd=ENABLE; NVIC_Init(&n);
    TIM_Cmd(TIM4,ENABLE);
    GPIO_ResetBits(GPIOB,GPIO_Pin_10|GPIO_Pin_6);
    motor_dir(0,0,0,0); pwm(0,0);
    echo_state=0; echo_ready=0; echo_width=0; last_trigger_ms=0;
}

void TIM4_IRQHandler(void) {
    if(TIM_GetITStatus(TIM4,TIM_IT_CC2)!=RESET) {
        uint16_t captured=TIM_GetCapture2(TIM4);
        TIM_ClearITPendingBit(TIM4,TIM_IT_CC2);
        if(echo_state==1) {
            echo_start=captured;
            echo_state=2;
            echo_capture_polarity(TIM_ICPolarity_Falling);
        } else if(echo_state==2) {
            echo_width=(uint16_t)(captured-echo_start);
            echo_ready=1;
            echo_state=0;
            echo_capture_polarity(TIM_ICPolarity_Rising);
        }
    }
}

void VehicleHw_UltrasonicTask(uint32_t now_ms, VehicleState *state) {
    uint16_t now, width;
    if(echo_ready) {
        __disable_irq(); width=echo_width; echo_ready=0; __enable_irq();
        state->distance_mm=(uint32_t)width*343u/2000u;
        state->distance_valid=(width>=150u && width<=ULTRASONIC_TIMEOUT_US);
        state->distance_ms=now_ms;
    }
    if(echo_state==0) {
        if((uint32_t)(now_ms-last_trigger_ms)<ULTRASONIC_PERIOD_MS)return;
        last_trigger_ms=now_ms; TIM_SetCounter(TIM4,0);
        GPIO_SetBits(GPIOB,GPIO_Pin_6);
        /* HC-SR04 requires a HIGH trigger pulse of at least 10 us.
         * TIM4 is already configured as a 1 MHz free-running timer, so use
         * it for the pulse width instead of a compiler/clock-dependent NOP
         * loop. The previous 120-NOP pulse was compiler-dependent and can be
         * shorter than the required 10 us, leaving ECHO permanently invalid. */
        while(TIM_GetCounter(TIM4) < 12u) { __NOP(); }
        GPIO_ResetBits(GPIOB,GPIO_Pin_6);
        echo_state=1;
        return;
    }
    now=TIM_GetCounter(TIM4);
    if(echo_state!=0 && now>=ULTRASONIC_TIMEOUT_US) {
        state->distance_valid=0;
        state->distance_ms=now_ms;
        echo_state=0;
        echo_ready=0;
        echo_capture_polarity(TIM_ICPolarity_Rising);
    }
}

void VehicleHw_ApplyMotors(const VehicleState *state) {
    if(!Vehicle_MotionAllowed(state)) {
        GPIO_ResetBits(GPIOB,GPIO_Pin_10); motor_dir(0,0,0,0); pwm(0,0); return;
    }
    GPIO_SetBits(GPIOB,GPIO_Pin_10);
    motor_dir(1,0,1,0);
    pwm(state->left_duty,state->right_duty);
}
