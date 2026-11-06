"""Restore the observed camera's default image controls, with explicit rollback."""
import argparse
import json
from pathlib import Path
import subprocess

DEFAULTS=dict(brightness=128,contrast=32,gain=4,backlight_compensation=0,
              auto_exposure=3,exposure_dynamic_framerate=1)


def read(device,names):
    r=subprocess.run(['v4l2-ctl','-d',device,'--get-ctrl='+','.join(names)],
                     check=True,text=True,capture_output=True)
    values={}
    for line in r.stdout.splitlines():
        key,sep,value=line.partition(':')
        if sep and key.strip() in names:values[key.strip()]=int(value.strip().split()[0])
    if set(values)!=set(names):raise RuntimeError('Camera does not report all required controls')
    return values


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('show','defaults','restore'))
    ap.add_argument('--device',default='/dev/video0')
    ap.add_argument('--state',type=Path,default=Path(__file__).resolve().parents[1]/'logs/camera_image_before.json')
    a=ap.parse_args()
    before=read(a.device,DEFAULTS)
    if a.action=='show':print(json.dumps(before,indent=2));return
    if a.action=='defaults':
        if not a.state.exists():
            a.state.parent.mkdir(parents=True,exist_ok=True)
            a.state.write_text(json.dumps(dict(device=a.device,controls=before),indent=2),encoding='utf-8')
        else:
            saved=json.loads(a.state.read_text(encoding='utf-8'))
            if saved['device']!=a.device:raise SystemExit('Saved settings belong to a different device')
        desired=DEFAULTS
    else:
        saved=json.loads(a.state.read_text(encoding='utf-8'))
        if saved['device']!=a.device:raise SystemExit('Saved settings belong to a different device')
        desired=saved['controls']
        if set(desired)!=set(DEFAULTS):raise SystemExit('Unexpected controls in saved state')
    # These defaults were read from this exact UVC camera. Other models need their own controls.
    for name,value in desired.items():
        subprocess.run(['v4l2-ctl','-d',a.device,f'--set-ctrl={name}={int(value)}'],check=True)
    print(json.dumps(read(a.device,DEFAULTS),indent=2))
    print('Wait for automatic exposure to settle; measure frame rate and face quality again.')


if __name__=='__main__':main()
