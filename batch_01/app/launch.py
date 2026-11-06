"""Single supported entry for this standalone package (Linux desktop or CLI)."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
LABELS={
    'full':'完整视觉＋人脸姓名＋语音＋云台＋测距（串口开启）',
    'preview':'人体＋人脸姓名＋语音预览（无串口）',
    'body-only':'仅人体视觉（关闭人脸与语音，无串口）',
    'enroll':'登记人脸姓名（无串口）',
    'list':'查看已登记姓名',
    'speech-test':'测试中文欢迎语音（无摄像头、无串口）',
    'camera':'仅相机预览与采集帧率',
    'distance':'仅超声波遥测监听（不发送控制）',
    'pan':'水平舵机小幅测试（电机VM断开）',
    'tilt':'俯仰舵机小幅测试（电机VM断开）',
    'front':'前轮舵机小幅测试（电机VM断开）',
    'exposure':'曝光预览、质量检查与固定曝光保存',
    'fixed-preview':'已保存固定曝光＋人脸预览（无串口）',
}


def command_for(mode,port='',name='',follow_face=0,backend='npu',extra=()):
    py=sys.executable
    if mode in ('full','preview','body-only','fixed-preview'):
        profile='saved-fixed' if mode=='fixed-preview' else 'quality'
        if profile=='saved-fixed':
            from camera_exposure import load_profile
            load_profile(0)
        cmd=[py,'-u',str(ROOT/'body_main.py'),'--person-backend',backend,
             '--gimbal-config',str(ROOT/'gimbal_config.json'),
             '--steering-config',str(ROOT/'steering_gimbal_config.json'),
             '--frame-rotation','180','--width','640','--height','480',
             '--threads','2','--faces','off' if mode=='body-only' else 'on',
             '--speech','off' if mode=='body-only' else 'on','--face-hz','2',
             '--camera-profile',profile]
        if mode=='full':cmd+=['--port',port]
        if follow_face:cmd+=['--follow-face',str(follow_face)]
        return cmd+list(extra)
    if mode in ('enroll','list'):
        cmd=[py,'-u',str(ROOT/'tools/enroll_named_face.py')]
        return cmd+(['--list'] if mode=='list' else ['--name',name])
    if mode in ('pan','tilt','front'):
        return [py,'-u',str(ROOT/'tools/gimbal_check.py'),'--port',port,'--axis',mode]
    if mode=='distance':return [py,'-u',str(ROOT/'tools/distance_monitor.py'),'--port',port]
    if mode=='camera':return [py,'-u',str(ROOT/'tools/desktop_checks.py'),'camera']
    if mode=='exposure':return [py,'-u',str(ROOT/'tools/exposure_tuner.py')]
    raise ValueError('Unsupported action: '+mode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=tuple(LABELS),default='preview')
    p.add_argument('--port',default='')
    p.add_argument('--name',default='测试人员')
    p.add_argument('--follow-face',type=int,default=0)
    p.add_argument('--backend',choices=('npu','cpu'),default='npu')
    a,extra=p.parse_known_args()
    if os.name!='posix':p.error('该控制入口供香橙派Linux使用；Windows用于源码与离线测试')
    import fcntl
    from desktop_control import pick_port,ticks
    import json
    (ROOT/'logs').mkdir(exist_ok=True)
    os.chdir(ROOT)
    # Match the inherited launcher lock, so these two menus cannot own devices together.
    with (ROOT/'logs/hardware_session.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('已有程序运行，请先关闭原窗口或点击停止')
        if a.mode=='speech-test':
            from welcome_speech import SpeechPlayer
            player=SpeechPlayer()
            try:
                if not player.enqueue(a.name):raise RuntimeError(player.status)
                player.jobs.join()
                print(player.status,flush=True)
            finally:player.close()
            return
        if a.mode in ('full','distance','pan','tilt','front'):
            a.port=a.port or pick_port()
        devices=[]
        if a.mode in ('full','preview','body-only','enroll','camera','exposure','fixed-preview'):
            devices.append('/dev/video0')
        if a.port:devices.append(a.port)
        for device in devices:
            if not Path(device).exists():raise RuntimeError('设备不存在：'+device)
            if subprocess.run(['fuser',device],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
                raise RuntimeError('设备已被其他程序占用：'+device)
        if a.mode in ('pan','tilt','front'):print('测试会转动舵机；电机VM须保持断开。',flush=True)
        if a.mode=='full':print('完整模式开启串口；当前未标定配置禁止电机行进。首次联调保持VM断开。',flush=True)
        command=command_for(a.mode,a.port,a.name,a.follow_face,a.backend,extra)
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log=ROOT/'logs'/f'session_{a.mode}_{stamp}.log'
        env=os.environ.copy()
        env.update(PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                   QT_AUTO_SCREEN_SCALE_FACTOR='0',QT_SCALE_FACTOR='1',QT_FONT_DPI='96')
        print('项目：',REPO,'\n模式：',LABELS[a.mode],'\n日志：',log,flush=True)
        with log.open('w',encoding='utf-8') as out:
            out.write(json.dumps(dict(mode=a.mode,command=command),ensure_ascii=False)+'\n');out.flush()
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',start_new_session=True)
            (ROOT/'logs/desktop_session.json').write_text(json.dumps(dict(child_pid=child.pid,
                start_ticks=ticks(child.pid),action=a.mode,log=str(log))),encoding='utf-8')
            def forward(*_):
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGINT)
                    except ProcessLookupError:pass
            signal.signal(signal.SIGINT,forward);signal.signal(signal.SIGTERM,forward)
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);out.write(line);out.flush()
                code=child.wait()
            finally:
                forward()
                try:child.wait(timeout=18)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            if code:raise SystemExit(f'程序退出码 {code}，详见 {log}')


if __name__=='__main__':main()
