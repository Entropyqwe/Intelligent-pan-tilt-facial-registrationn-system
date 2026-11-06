"""Offline asynchronous Mandarin greeting; no shell and no network service."""
import queue
import shutil
import subprocess
import threading
import time
from named_faces import checked_name


class WelcomeGate:
    def __init__(self, cooldown=60., departure=8.):
        self.cooldown, self.departure = cooldown, departure
        self.last_seen = {}
        self.last_spoken = {}

    def update(self, identities, now, enqueue):
        for pid, name in identities.items():
            previous = self.last_seen.get(pid)
            self.last_seen[pid] = now
            returning = previous is None or now-previous >= self.departure
            due = pid not in self.last_spoken or now-self.last_spoken[pid] >= self.cooldown
            if returning and due and enqueue(name):
                self.last_spoken[pid] = now
            elif returning and due:
                # Failed/busy audio can retry while the person remains visible.
                self.last_seen.pop(pid, None)


class SpeechPlayer:
    def __init__(self, enabled=True, voice='cmn', executable=None, runner=None):
        self.executable = executable or shutil.which('espeak-ng')
        self.enabled = enabled
        self.voice = voice
        self.runner = runner or subprocess.run
        self.status = '语音关闭' if not enabled else ('等待欢迎' if self.executable else '缺少 espeak-ng')
        self.jobs = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.active = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='welcome-audio', daemon=True)
        self.thread.start()

    def enqueue(self, name):
        if not self.enabled or not self.executable or self.stop.is_set():
            return False
        name = checked_name(name)
        with self.lock:
            if self.active == name:
                return False
        try:
            self.jobs.put_nowait(name)
            return True
        except queue.Full:
            self.status = '语音队列繁忙'
            return False

    def _run(self):
        while not self.stop.is_set():
            try:
                name = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            with self.lock:
                self.active = name
            try:
                self.status = f'播报：{name}，欢迎光临'
                result = self.runner([self.executable, '-v', self.voice, '-s', '155', '--stdin'],
                    input=f'{name}，欢迎光临。\n', text=True, encoding='utf-8',
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12, check=False)
                if result.returncode:
                    self.status = '音频失败，请检查声卡/音量'
                    print(self.status+': '+str(result.stderr)[-200:], flush=True)
                else:
                    self.status = f'已欢迎：{name}'
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.status = '音频不可用'
                print(f'语音异常：{exc}', flush=True)
            finally:
                with self.lock:
                    self.active = None
                self.jobs.task_done()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=13.)
