"""Synthetic S1 tests. Only use with raised wheels, propulsion disconnected."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from steering import encode_s1
import serial
p=argparse.ArgumentParser()
p.add_argument('--port',required=True)
p.add_argument('--pulse',type=int,default=1500)
p.add_argument('--seconds',type=float,default=3.)
p.add_argument('--invalid',action='store_true')
a=p.parse_args()
if not 1300<=a.pulse<=1700 or not 0<a.seconds<=30: p.error('pulse 1300..1700, duration (0,30]')
with serial.Serial(a.port,115200,timeout=.05,write_timeout=.2) as s:
    s.reset_input_buffer()
    try:
        end=time.monotonic()+a.seconds
        seq=0
        while time.monotonic()<end:
            seq+=1
            f=dict(valid=0 if a.invalid else 1,target_id=1,bearing_mdeg=0,
                   servo_us=1500 if a.invalid else a.pulse,age_ms=0,source=0 if a.invalid else 1)
            packet=encode_s1(seq,f); s.write(packet)
            print('TX',packet.decode().strip())
            data=s.read(s.in_waiting or 1)
            if data: print(data.decode('ascii',errors='replace'),end='')
            time.sleep(.1)
    finally:
        seq+=1
        s.write(encode_s1(seq,dict(valid=0,target_id=1,bearing_mdeg=0,servo_us=1500,age_ms=0,source=0)))
        s.flush()
print('Stopped; invalid packet sent. MCU timeout also returns to center.')
