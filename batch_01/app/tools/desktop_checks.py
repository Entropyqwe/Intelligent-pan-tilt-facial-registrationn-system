"""Independent checks. No firmware changes and no new wire protocol."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
running = True


def stop(*_):
    global running
    running = False


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop)


def camera_check():
    import cv2
    from main import LatestCamera
    camera = LatestCamera(0, 640, 480, 30, camera_profile='quality')
    started = time.monotonic()
    count = seq = 0
    window = time.monotonic()
    fps = 0.
    try:
        while running:
            frame, stamp, number = camera.latest(seq)
            if camera.failed:
                raise RuntimeError('摄像头掉线，重新接入后再试。')
            if frame is None:
                if time.monotonic() - started > 8 and not count:
                    raise RuntimeError('8秒未收到相机图像。')
                time.sleep(.005)
                continue
            seq = number
            count += 1
            now = time.monotonic()
            if now-window >= 1:
                fps = count/(now-window)
                count = 0
                window = now
                print(f'相机实际交付帧率={fps:.2f} FPS；画面={frame.shape[1]}x{frame.shape[0]}；图像年龄={(now-stamp)*1000:.1f}ms', flush=True)
            view = frame.copy()
            cv2.putText(view, f'640x480 COLOR | capture {fps:.1f} FPS | Q: exit', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1, cv2.LINE_AA)
            cv2.imshow('Camera only - no serial', view)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'),27) or cv2.getWindowProperty('Camera only - no serial',cv2.WND_PROP_VISIBLE)<1:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


def serial_check(action, port):
    import serial
    from steering import encode_s1
    counts = Counter()
    sequence = 0
    received = bytearray()
    distance = None
    last_distance = 0.
    def read(link):
        nonlocal received, distance, last_distance
        received.extend(link.read(link.in_waiting or 1))
        if len(received)>8192:
            raise RuntimeError('串口没有有效换行帧，请检查波特率和接线。')
        while b'\n' in received:
            raw, _, received = received.partition(b'\n')
            line = raw.decode('ascii', errors='replace').strip()
            print('RX', line, flush=True)
            if line.startswith('#A1,'):
                counts['A1'] += 1
            if line.startswith('#M1,'):
                fields=line.split(',')
                try:
                    if len(fields)==8:
                        print(f'主控输出：舵机{int(fields[3])}us，电机{int(fields[4])/10:.1f}% / 目标{int(fields[6])/10:.0f}%，状态={fields[7]}（非实测转速）',flush=True)
                        counts['M1'] += 1
                except ValueError:counts['MALFORMED'] += 1
            if line.startswith('#D1,'):
                fields = line.split(',')
                try:
                    if len(fields)!=4: continue
                    mm = int(fields[1]); valid=int(fields[2])==1 and mm>0
                    distance = mm if valid else None
                    last_distance = time.monotonic()
                    counts['D1'] += 1
                    counts['D1_VALID' if valid else 'D1_INVALID'] += 1
                    print(f'测距={mm}mm；有效={valid}；主控状态={fields[3]}', flush=True)
                except ValueError:
                    counts['MALFORMED'] += 1
    def send(link, valid, pulse=1500):
        nonlocal sequence
        sequence += 1
        packet = encode_s1(sequence,dict(valid=int(valid),target_id=1,bearing_mdeg=0,
                                        servo_us=pulse,age_ms=0,source=1 if valid else 0))
        link.write(packet)
        print('TX',packet.decode().strip(),flush=True)
    with serial.Serial(port,115200,timeout=.05,write_timeout=.2,exclusive=True) as link:
        print(f'串口={port}，115200 8N1；停止可点击面板停止按钮或按Ctrl+C。',flush=True)
        if action=='distance':
            print('只监听，不发送任何控制。',flush=True)
            deadline=time.monotonic()+30
            while running and time.monotonic()<deadline: read(link)
        else:
            try:
                send(link,False)
                if action=='motor':
                    print('先观察2秒测距；仅有效且至少350mm才进入5秒测试。',flush=True)
                    until=time.monotonic()+2
                    while running and time.monotonic()<until: read(link)
                    if not running:return
                    if distance is None or distance<350 or time.monotonic()-last_distance>.6:
                        raise RuntimeError('拒绝启动：超声波无效、距离小于350mm或反馈已过期。')
                    stages=[(1500,5.)]
                else:
                    stages=[(1500,1.5),(1400,2.),(1600,2.),(1500,1.5)]
                for pulse,seconds in stages:
                    print(f'阶段：舵机命令={pulse}us，持续={seconds}s。ACK仅证明收到指令，不代表轴已转动。',flush=True)
                    until=time.monotonic()+seconds;last_tx=0
                    while running and time.monotonic()<until:
                        read(link)
                        now=time.monotonic()
                        if action=='motor' and (distance is None or distance<=300 or now-last_distance>.6):
                            raise RuntimeError('结束电机测试：距离过近或测距无效/过期。')
                        if now-last_tx>=.1:
                            send(link,True,pulse);last_tx=now
                    if not running:break
            finally:
                for _ in range(3):
                    try:
                        send(link,False);link.flush();time.sleep(.05)
                    except serial.SerialException:break
        print('统计：',dict(counts),flush=True)
        if not counts['D1']:
            print('未收到#D1：核对车辆固件、PA9/PA10交叉连接、共地和串口。',flush=True)
        elif not counts['D1_VALID']:
            print('串口有测距反馈但无有效回波：核对HC-SR04供电、PB6/PB7及ECHO分压。',flush=True)


def npu_check():
    import cv2
    import numpy as np
    from yolo_npu import YoloNpuVision
    cv2.setNumThreads(2)
    model=Path('/opt/yolov5/model/yolov5.nb')
    print('模型：',model,'SHA256：',hashlib.sha256(model.read_bytes()).hexdigest(),flush=True)
    detector=YoloNpuVision(str(model),.35)
    try:
        frame=np.zeros((480,640,3),np.uint8)
        for _ in range(3):detector.process(frame)
        start=time.monotonic();count=0
        for _ in range(30):
            if not running:break
            detector.process(frame);count+=1
        elapsed=time.monotonic()-start
        print(json.dumps(dict(frames=count,seconds=elapsed,inference_pipeline_fps=count/elapsed,
                              last_timing=detector.last_timing),ensure_ascii=False,indent=2),flush=True)
        print('该结果验证NPU执行和数据处理，不代表摄像头实时帧率，也不证明人体检测准确度。',flush=True)
    finally:detector.close()


def status_check():
    print('项目目录：',ROOT,'\nPython：',sys.executable,flush=True)
    for name in ['.venv','models']:
        p=ROOT/name
        print(name,'真实路径：',p.resolve(),'符号链接：',p.is_symlink(),flush=True)
    for name in ['body_main.py','main.py','steering.py','steering_config.json','yolo_npu.py']:
        print('SHA256',name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),flush=True)
    for cmd in [['lsusb'],['ls','-l','/dev/video0'],['bash','-c','ls -l /dev/serial/by-id/* /dev/ttyUSB* 2>/dev/null'],
                [sys.executable,'tools/check_npu.py']]:
        print('\n$', ' '.join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT)
    print('\n配置：',(ROOT/'steering_config.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['camera','distance','servo','motor','npu','status']);p.add_argument('--port');a=p.parse_args()
    if a.action=='camera':camera_check()
    elif a.action=='npu':npu_check()
    elif a.action=='status':status_check()
    else:
        if not a.port:p.error('serial check requires --port')
        serial_check(a.action,a.port)
