"""Small three-axis bench test. G1 motion=0 throughout; requires G2 firmware.
Disconnect motor VM first. Does not fake distance or arm motor motion.
"""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gimbal_control import checked_frame
from steering import encode_s1


def main():
    import serial
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--axis',choices=('pan','tilt','front'),required=True)
    p.add_argument('--delta',type=int,default=50)
    a=p.parse_args()
    if not 10<=a.delta<=100:p.error('首次测试幅度仅允许10..100微秒')
    print('请先断开电机VM；本测试不允许电机启动。',flush=True)
    seq=0;buf=bytearray();has_g2=False
    with serial.Serial(a.port,115200,timeout=.02,write_timeout=.2) as link:
        def receive():
            nonlocal buf,has_g2
            buf.extend(link.read(link.in_waiting or 1))
            if len(buf)>8192:buf=buf[-256:]
            while b'\n' in buf:
                raw,_,buf=buf.partition(b'\n');line=raw.decode('ascii',errors='replace')
                print('RX',line,flush=True)
                if line.startswith('#G2,'):
                    try:
                        values=list(map(int,line.split(',')[1:]))
                        if len(values)==8 and 1000<=values[2]<=2000 and 1300<=values[3]<=1700:has_g2=True
                    except ValueError:pass
        def send(valid=0,pulse=1500):
            nonlocal seq
            seq=seq%4294967295+1
            pan=pulse if a.axis=='pan' and valid else 1500
            tilt=pulse if a.axis=='tilt' and valid else 1500
            front=pulse if a.axis=='front' and valid else 1500
            fields=dict(valid=valid,target_id=1,bearing_mdeg=0,servo_us=front,age_ms=0,source=2 if valid else 0)
            link.write(checked_frame('G1',(seq,valid,pan,tilt,0,0))+encode_s1(seq,fields))
        try:
            until=time.monotonic()+3
            while time.monotonic()<until and not has_g2:
                send();receive();time.sleep(.04)
            if not has_g2:raise RuntimeError('未收到G2，拒绝发有效S1；先烧录新版FaceLock_GIMBAL.hex')
            for pulse in (1500,1500-a.delta,1500+a.delta,1500):
                print(f'轴={a.axis} / 指令={pulse}us / 电机许可=0',flush=True)
                until=time.monotonic()+2
                while time.monotonic()<until:send(1,pulse);receive();time.sleep(.04)
        finally:
            send();time.sleep(.05)

if __name__=='__main__':main()
