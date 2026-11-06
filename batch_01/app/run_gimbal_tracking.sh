#!/usr/bin/env bash
# Compatibility entry. All new launches use launch.py and the same device lock.
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec .venv/bin/python launch.py --mode full --steering-config steering_gimbal_config.json "$@"
