"""Rebuild with local ARM Compiler 5; no download or flash operation."""
from pathlib import Path
import os, subprocess, sys
P=Path(__file__).resolve().parent
T=Path(os.environ.get('ARMCC_BIN',r'D:\ARM\ARMCC\bin'))
env=os.environ.copy()
env.setdefault('ARM_TOOL_VARIANT','mdk_lite')
env.setdefault('ARM_PRODUCT_PATH',r'D:\ARM\sw\mappings')
B=P/'Objects'; B.mkdir(exist_ok=True)
log=[]
def run(args):
    r=subprocess.run([str(x) for x in args],cwd=P,env=env,capture_output=True,text=True,errors='replace')
    log.append(' '.join(str(x) for x in args)+'\n'+r.stdout+r.stderr)
    (B/'build_log.txt').write_text('\n'.join(log),encoding='utf-8')
    print(r.stdout+r.stderr,end='')
    if r.returncode: raise SystemExit(r.returncode)
sources=['User/main.c','User/oled_status.c','User/gimbal_control.c','User/steering_protocol.c','User/vehicle_control.c','User/vehicle_hw.c','User/stm32f10x_it.c',
         'Hardware/OLED.c','Start/system_stm32f10x.c','Library/stm32f10x_rcc.c',
         'Library/stm32f10x_gpio.c','Library/stm32f10x_tim.c','Library/stm32f10x_usart.c',
         'Library/stm32f10x_iwdg.c','Library/misc.c']
objs=[]
for name in sources:
    source=P/name; obj=B/(source.stem+'.o'); objs.append(obj)
    run([T/'armcc.exe','--cpu','Cortex-M3','--c99','-O2','--split_sections',
         '-DSTM32F10X_MD','-DUSE_STDPERIPH_DRIVER',
         '-IStart','-ILibrary','-IUser','-IHardware','-c',source,'-o',obj])
startup=B/'startup.o'; objs.append(startup)
run([T/'armasm.exe','--cpu','Cortex-M3',P/'Start/startup_stm32f10x_md.s','-o',startup])
axf=B/'FaceLock_GIMBAL.axf'
run([T/'armlink.exe','--cpu','Cortex-M3','--ro-base','0x08000000','--rw-base','0x20000000',
     '--first','startup.o(RESET)','--entry','Reset_Handler','--strict','--info','sizes',
     '--map','--list',B/'FaceLock_GIMBAL.map','--output',axf,*objs])
run([T/'fromelf.exe','--i32combined','--output',B/'FaceLock_GIMBAL.hex',axf])
run([T/'fromelf.exe','--bin','--output',B/'FaceLock_GIMBAL.bin',axf])
print('Built',B/'FaceLock_GIMBAL.hex')
