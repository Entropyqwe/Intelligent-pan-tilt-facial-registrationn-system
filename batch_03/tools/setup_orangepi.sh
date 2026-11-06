#!/usr/bin/env bash
# Run as the desktop user. sudo only installs OS dependencies/group membership.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
if [ "$(id -u)" -eq 0 ]; then echo '请以日常桌面用户运行，不要 sudo 整个脚本。'; exit 1; fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-tk build-essential \
  libgl1 libglib2.0-0 fonts-noto-cjk v4l-utils alsa-utils espeak-ng psmisc xterm
python3 -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,13), "需要Python 3.10～3.12"'
python3 -m venv "$repo/app/.venv"
"$repo/app/.venv/bin/python" -m pip install -r "$repo/requirements.txt"
offline="$repo/../offline_models_DO_NOT_UPLOAD"
if [ -d "$offline" ]; then
  "$repo/app/.venv/bin/python" "$repo/tools/models.py" --offline "$offline"
else
  "$repo/app/.venv/bin/python" "$repo/tools/models.py"
fi
sudo usermod -aG video,dialout,audio "$(id -un)"
if [ -c /dev/vipcore ] && [ -f /usr/include/vip_lite.h ]; then
  ( cd "$repo/app/native"; bash build.sh )
else
  echo '未发现VIPLite设备/头文件。人体NPU模式还需官方SDK、兼容的YOLOv5 .nb模型。'
  echo '可先用 bash run.sh --mode preview --backend cpu 验证人脸/语音/UI；不保证CPU帧率。'
fi
bash "$repo/tools/install_desktop.sh"
echo '安装结束；新增设备组权限须注销桌面后重新登录。默认自动曝光。'
