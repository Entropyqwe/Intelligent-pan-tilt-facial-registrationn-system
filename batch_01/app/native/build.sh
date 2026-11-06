#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -f /usr/include/vip_lite.h || { echo 'Missing official target VIPLite header'; exit 1; }
cc -O3 -std=c99 -Wall -Wextra -Werror -fPIC -shared vip_bridge.c -o libfacelock_vip.so -lNBGlinker -lVIPhal -lm
echo "Built $PWD/libfacelock_vip.so with this board's SDK headers"
