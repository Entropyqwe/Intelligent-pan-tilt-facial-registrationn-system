"""Read-only A733 NPU inventory. Does not load drivers or execute inference."""
from pathlib import Path
import glob
import os
import platform
import shutil
import subprocess


def read(path):
    try:
        return Path(path).read_text(errors='replace').strip().replace('\x00', '')
    except OSError:
        return '(unavailable)'


print('=== System ===')
print('Model:', read('/proc/device-tree/model'))
print('Kernel:', platform.release(), 'Architecture:', platform.machine())
print('=== NPU device ===')
devices = sorted(set(glob.glob('/dev/*vip*') + glob.glob('/dev/*npu*') + glob.glob('/dev/galcore')))
for name in devices:
    print(name, 'readable=', os.access(name, os.R_OK), 'writable=', os.access(name, os.W_OK))
if not devices:
    print('No matching device node. This does not prove the hardware is absent.')
print('=== Loaded drivers ===')
modules = read('/proc/modules').splitlines()
hits = [line for line in modules if any(x in line.lower() for x in ('vip', 'npu', 'galcore'))]
print('\n'.join(hits) if hits else 'No matching loadable module; driver may be built in or absent.')
print('=== Runtime tools ===')
for name in ('vpm_run', 'vip_test', 'viplite_test', 'nbg_unify', 'pegasus'):
    print(name+':', shutil.which(name) or '(not on PATH)')
print('=== Dynamic linker cache ===')
ldconfig = shutil.which('ldconfig') or '/sbin/ldconfig'
try:
    proc = subprocess.run([ldconfig, '-p'], capture_output=True, text=True, timeout=10)
    hits = [line for line in proc.stdout.splitlines()
            if any(x in line.lower() for x in ('viplite','vipuser','viphal','nbglinker','tim-vx','openvx'))]
    print('\n'.join(hits) if hits else 'No matching runtime in linker cache; it may exist elsewhere.')
except (OSError, subprocess.TimeoutExpired) as exc:
    print('Could not read linker cache:', exc)
print('=== Likely SDK library paths ===')
patterns = [base+pattern for base in ('/lib/', '/usr/lib/', '/usr/lib/aarch64-linux-gnu/', '/usr/local/lib/')
            for pattern in ('libVIP*.so*','libNBG*.so*','libOpenVX*.so*')]
paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
print('\n'.join(paths) if paths else 'No matching library in the checked paths.')
print('Inventory only: usable acceleration still requires a model inference test.')
