#!/usr/bin/env python3
"""Chinese desktop entry point; hardware control remains in the existing project."""
import argparse
import datetime
try:
    import fcntl
except ImportError:
    fcntl=None  # Command construction is inspectable on Windows; GUI runs on Linux.
import glob
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'logs/desktop_session.json'
LABELS = {
    'gimbal_full':'云台实验版 · 视觉＋两轴云台＋前轮＋超声波（需新版固件）',
    'gimbal_fixed':'云台完整跟随 · 已测试保存的固定曝光',
    'gimbal_pan':'云台台架 · 仅水平轴小幅测试（VM断开）',
    'gimbal_tilt':'云台台架 · 仅俯仰轴小幅测试（VM断开）',
    'gimbal_front':'云台台架 · 仅前轮小幅测试（VM断开）',
    'full': '自动曝光 · 完整视觉跟随（开启串口）',
    'vision': '自动曝光 · 仅人体视觉与中文UI（不连接串口）',
    'exposure': '曝光调试 · 预览、测试、保存当前固定曝光（无串口）',
    'fixed_vision': '已保存固定曝光 · 仅人体视觉与中文UI（无串口）',
    'fixed_full': '已保存固定曝光 · 完整视觉跟随（开启串口）',
    'camera': '仅摄像头预览与实测帧率',
    'distance': '仅串口与超声波监听（不发送控制）',
    'servo': '舵机左右与回中验证（必须断开电机VM）',
    'motor': '单电机短时验证（架空车轮，最长5秒）',
    'npu': 'NPU独立推理验证（不打开摄像头和串口）',
    'status': '设备、环境与版本检查',
    'stop': '停止由本面板启动的程序',
    'logs': '打开运行日志文件夹',
    'help': '打开启动与排错说明',
}


def dialog(kind, text):
    return subprocess.run(['zenity', '--' + kind, '--title=人体跟随控制中心',
                           '--width=640', '--text=' + text]).returncode == 0


def pick_port():
    ports = sorted(glob.glob('/dev/serial/by-id/*'))
    preferred = [p for p in ports if '1a86' in p or 'CH340' in p or 'CH341' in p]
    ports = preferred or ports or sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    if not ports:
        raise RuntimeError('未发现USB-TTL。将TTL的USB接到香橙派，再检查数据线与集线器。')
    # Resolve duplicate names referring to the same device.
    unique = {str(Path(p).resolve()): p for p in ports}
    if len(unique) != 1:
        raise RuntimeError('发现多个串口，请先移除无关串口设备，避免选错主控板。')
    return next(iter(unique.values()))


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def stop_session():
    try:
        state = json.loads(STATE.read_text())
        pid = int(state['child_pid'])
        if ticks(pid) != state['start_ticks'] or os.getpgid(pid) != pid:
            return '原会话已经结束，无需停止。'
        if Path(f'/proc/{pid}/cwd').resolve() != ROOT:
            return '进程目录不匹配，未发送停止信号。'
        os.killpg(pid, signal.SIGINT)
        for _ in range(50):
            time.sleep(.1)
            if not Path(f'/proc/{pid}').exists():
                return '已请求正常退出；视觉程序会发送无效控制，主控超时保护继续生效。'
        os.killpg(pid, signal.SIGTERM)
        return '已发送终止信号。请确认电机已停止；串口中断后由STM32超时保护停车。'
    except (FileNotFoundError, ProcessLookupError):
        return '没有本面板正在运行的会话。'


def command_for(action):
    py = str(ROOT / '.venv/bin/python')
    if action=='gimbal_fixed':
        from camera_exposure import load_profile
        load_profile(0)
        return ['bash',str(ROOT/'run_gimbal_tracking.sh'),'--camera-profile','saved-fixed']
    if action=='gimbal_full':
        return ['bash',str(ROOT/'run_gimbal_tracking.sh')]
    if action.startswith('gimbal_'):
        return [py,'-u','tools/gimbal_check.py','--port',pick_port(),'--axis',action.split('_')[1]]
    if action=='exposure':return [py,'-u','tools/exposure_tuner.py']
    if action in ('fixed_vision','fixed_full'):
        from camera_exposure import load_profile
        load_profile(0)  # Fail before launching control if no usable saved file.
    profile='saved-fixed' if action in ('fixed_vision','fixed_full') else 'quality'
    if action in ('full','fixed_full'):
        return ['bash', str(ROOT / 'run_human_tracking.sh'),'--camera-profile',profile]
    if action in ('vision','fixed_vision'):
        return [py, '-u', 'body_main.py', '--person-backend', 'npu', '--lock-mode', 'auto',
                '--single-person-slot', 'on', '--threads', '2', '--width', '640', '--height', '480',
                '--camera-fps', '30','--camera-profile',profile]
    cmd = [py, '-u', 'tools/desktop_checks.py', action]
    if action in ('distance', 'servo', 'motor'):
        cmd += ['--port', pick_port()]
    return cmd


def run_action(action):
    if fcntl is None:raise RuntimeError('图形启动面板在香橙派 Linux 使用')
    os.chdir(ROOT)
    (ROOT / 'logs').mkdir(exist_ok=True)
    with (ROOT / 'logs/hardware_session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有面板程序运行。请先点击“停止”，或关闭原程序。', flush=True)
            return 2
        devices = []
        if action in ('full', 'vision', 'camera','exposure','fixed_vision','fixed_full','gimbal_full','gimbal_fixed'):
            if not Path('/dev/video0').exists():
                raise RuntimeError('未发现 /dev/video0，请接入摄像头。')
            devices.append('/dev/video0')
        if action in ('full', 'fixed_full', 'distance', 'servo', 'motor') or action.startswith('gimbal_'):
            devices.append(pick_port())
        for device in devices:
            if subprocess.run(['fuser', device], capture_output=True).returncode == 0:
                raise RuntimeError(f'{device} 已被其他程序占用，请先退出原程序。')
        command = command_for(action)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log = ROOT / 'logs' / f'desktop_{action}_{stamp}.log'
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   QT_AUTO_SCREEN_SCALE_FACTOR='0', QT_SCALE_FACTOR='1', QT_FONT_DPI='96')
        print('项目目录：', ROOT, '\n模式：', LABELS[action], '\n日志：', log, flush=True)
        with log.open('w', encoding='utf-8', buffering=1) as f:
            f.write(json.dumps(dict(time=stamp, action=action, cwd=str(ROOT), command=command),
                               ensure_ascii=False) + '\n')
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, errors='replace',
                                     start_new_session=True)
            state = dict(child_pid=child.pid, start_ticks=ticks(child.pid), action=action, log=str(log))
            STATE.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            def forward(sig, _):
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, forward)
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    f.write(line)
                code = child.wait()
                f.write(f'\n退出码：{code}\n')
                print(f'\n程序结束，退出码：{code}。日志已保存。', flush=True)
                return code
            finally:
                if child.poll() is None:
                    forward(signal.SIGINT, None)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                if STATE.exists() and json.loads(STATE.read_text()).get('child_pid') == child.pid:
                    STATE.unlink()


def menu():
    while True:
        args = ['zenity', '--list', '--title=云台实验版控制中心', '--width=900', '--height=680',
                '--text=当前：两轴云台新版；分析与显示默认旋转180°。自动曝光优先，设备测试互斥。',
                '--column=功能代码', '--column=操作说明', '--hide-column=1', '--print-column=1',
                '--ok-label=启动', '--cancel-label=关闭']
        for key, label in LABELS.items():
            # Fixed-camera control utilities stay in source for regression
            # comparison, but must not be mistaken for the gimbal entry.
            if key in ('full','fixed_full','servo','motor'):continue
            args += [key, label]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            return 0
        action = result.stdout.strip()
        if action == 'stop':
            dialog('info', stop_session())
        elif action == 'logs':
            subprocess.Popen(['xdg-open', str(ROOT / 'logs')])
        elif action == 'help':
            subprocess.Popen(['xdg-open', str(ROOT / 'docs/云台版桌面启动说明.md')])
        elif action in LABELS:
            if action.startswith('gimbal_'):
                if not dialog('question','云台实验版会驱动舵机。请断开电机VM，检查三轴舵臂空间。未标定时电机被锁止；上电前先把舵臂卸下回中。继续？'):
                    continue
            elif action in ('full','fixed_full', 'servo', 'motor'):
                text = {
                    'full': '完整跟随会发送有效S1；视觉和测距满足条件时，STM32会驱动电机。首次验证请断开电机VM。确认场地和接线已准备好后启动。',
                    'fixed_full': '使用你已测试保存的固定曝光，并开启原S1控制。视觉和测距满足条件时可能驱动电机。先确认当前光照下图像清楚，首次验证断开电机VM。',
                    'servo': '测试会发送有效S1，依次1500→1400→1600→1500微秒。现有固件可能同时允许电机运行，必须先断开TB6612的电机VM供电。已断开后继续。',
                    'motor': '先架空车轮并固定车辆，电机供电必须符合额定电压。程序确认距离至少350mm后，最多发送5秒有效S1；距离过近、测距失效立即结束。CONTROL_20261103_R1固件约1秒升到30%PWM；旧固件不具备缓启动。继续测试？'
                }[action]
                if not dialog('question', text):
                    continue
            command = shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), '--run', action])
            subprocess.Popen(['xfce4-terminal', '--disable-server', '--hold',
                              '--title=' + LABELS[action], '--command=' + command], cwd=ROOT)


if __name__ == '__main__':
    # Legacy helpers remain importable for compatibility tests. A direct launch
    # must open the standalone package's current face/gimbal menu.
    from control_center import main as standalone_menu
    standalone_menu()
