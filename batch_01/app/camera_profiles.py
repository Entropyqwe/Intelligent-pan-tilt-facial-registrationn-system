"""Measured profiles for the project's 0c45:64ab camera; no universal presets."""
from pathlib import Path
import subprocess
import sys

COMMON=dict(contrast=32,gain=4,backlight_compensation=0,
            power_line_frequency=1,white_balance_automatic=1)
PROFILES={
    'quality':dict(COMMON,auto_exposure=3,exposure_dynamic_framerate=1,
                   brightness=128,gamma=120),
    # Speed-only experimental profile: observed purple chroma noise. Must not
    # silently replace quality mode or be treated as a tracking acceptance.
    'fast-experimental':dict(COMMON,auto_exposure=1,exposure_dynamic_framerate=0,
                            exposure_time_absolute=10,brightness=64,gamma=100),
}


def usb_identity(index):
    device=Path(f'/sys/class/video4linux/video{index}/device').resolve()
    for parent in (device,*device.parents):
        vendor=parent/'idVendor';product=parent/'idProduct'
        if vendor.is_file() and product.is_file():
            return vendor.read_text().strip()+':'+product.read_text().strip()
    return None


def apply_profile(index,profile):
    if profile=='unchanged':return False
    if profile=='saved-fixed':
        from camera_exposure import apply_saved
        record=apply_saved(index)
        print('Camera profile: saved-fixed / '+record['saved_at'],file=sys.stderr,flush=True)
        return True
    if profile not in PROFILES:raise ValueError('Unknown camera profile')
    if sys.platform!='linux' or usb_identity(index)!='0c45:64ab':
        print('Camera profile skipped: preset requires USB 0c45:64ab on Linux',file=sys.stderr,flush=True)
        return False
    device=f'/dev/video{index}'
    for key,value in PROFILES[profile].items():
        subprocess.run(['v4l2-ctl','-d',device,f'--set-ctrl={key}={value}'],
                       check=True,capture_output=True,text=True,timeout=3)
    print(f'Camera profile: {profile}'+
          (' (experimental: visible chroma noise; tracking not accepted)' if profile=='fast-experimental' else ''),
          file=sys.stderr,flush=True)
    return True
