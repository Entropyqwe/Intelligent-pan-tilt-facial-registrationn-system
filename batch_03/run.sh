#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")" && pwd)"
test -x "$repo/app/.venv/bin/python" || { echo '请先执行 bash tools/setup_orangepi.sh'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export QT_AUTO_SCREEN_SCALE_FACTOR=0 QT_SCALE_FACTOR=1 QT_FONT_DPI=96
if [ "$#" -eq 0 ]; then
  exec "$repo/app/.venv/bin/python" "$repo/app/control_center.py"
fi
exec "$repo/app/.venv/bin/python" "$repo/app/launch.py" "$@"
