"""Bounded bench verification; motor VM must already be disconnected."""
import argparse
import json
from pathlib import Path
import sys
import time
import serial

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from steering import encode_s1

p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--mode',choices=['servo','ramp'],default='servo');a=p.parse_args()
events=[];buffer=bytearray();seq=0

def record(kind,**data):
    row=dict(t=time.monotonic(),kind=kind,**data);events.append(row)
    if kind!='TX':print(json.dumps(row),flush=True)

def send(link,valid,pulse):
    global seq
    seq+=1
    link.write(encode_s1(seq,dict(valid=int(valid),target_id=1,bearing_mdeg=0,servo_us=pulse,age_ms=0,source=2 if valid else 0)))
    record('TX',seq=seq,valid=valid,pulse=pulse)

def read(link):
    global buffer
    buffer.extend(link.read(link.in_waiting or 1))
    while b'\n' in buffer:
        raw,_,buffer=buffer.partition(b'\n');line=raw.decode('ascii',errors='replace').strip()
        if line.startswith(('#A1,','#D1,','#M1,','#FW,')):record('RX',line=line)

with serial.Serial(a.port,115200,timeout=.01,write_timeout=.2,exclusive=True) as link:
    try:
        stages=[('CENTER',True,1500,1.),('LEFT',True,1400,1.5),('RIGHT',True,1600,1.5),
                ('VISUAL_LOSS',False,1500,3.5)] if a.mode=='servo' else [
                ('RAMP',True,1500,5.),('INVALID_STOP',False,1500,1.)]
        for name,valid,pulse,seconds in stages:
            record('STAGE',name=name)
            until=time.monotonic()+seconds;last=0
            while time.monotonic()<until:
                now=time.monotonic()
                if now-last>=.05:send(link,valid,pulse);last=now
                read(link)
    finally:
        for _ in range(3):send(link,False,1500);time.sleep(.03)
        link.flush()
path=ROOT/'logs'/f'control_bench_{a.mode}_{time.time_ns()}.json'
path.write_text(json.dumps(events,indent=2),encoding='utf-8');print('SAVED',path,flush=True)
