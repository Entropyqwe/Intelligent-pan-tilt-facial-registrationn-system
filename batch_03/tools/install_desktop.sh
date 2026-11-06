#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
py="$repo/app/.venv/bin/python"
test -x "$py" || { echo '缺少虚拟环境，请先运行setup_orangepi.sh'; exit 1; }
"$py" - "$repo" <<'PY'
import os
from pathlib import Path
import subprocess
import sys
root=Path(sys.argv[1])
def quoted(value):
    return '"'+str(value).replace('\\','\\\\').replace('"','\\"').replace('`','\\`').replace('$','\\$').replace('%','%%')+'"'
try:
    result=subprocess.run(['xdg-user-dir','DESKTOP'],capture_output=True,text=True,check=True)
    desktop=Path(result.stdout.strip())
    if not desktop.is_absolute():raise ValueError('invalid desktop path')
except (OSError,ValueError,subprocess.CalledProcessError):
    desktop=Path.home()/'Desktop'
desktop.mkdir(exist_ok=True)
target=desktop/'FaceLock_Gimbal_Face.desktop'
target.write_text('[Desktop Entry]\nType=Application\nVersion=1.0\nName=人脸欢迎与云台控制中心\n'
    +'Comment=独立工程：人体、人脸、语音、超声波与三路舵机\n'
    +'Exec='+quoted(root/'app/.venv/bin/python')+' '+quoted(root/'app/control_center.py')+'\n'
    +'Terminal=false\nIcon=camera-web\nCategories=Utility;\n',encoding='utf-8')
target.chmod(0o755)
print('桌面入口：',target)
print('若桌面要求“允许启动/信任”，右键此图标确认一次。')
PY
