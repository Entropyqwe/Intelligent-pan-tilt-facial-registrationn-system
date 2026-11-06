"""Reversible V4L2 / CPU settings. Run explicitly on the board; no driver install."""
import argparse
import json
from pathlib import Path
import subprocess


def camera_get(device):
    result=subprocess.run(['v4l2-ctl','-d',device,'--get-ctrl=exposure_dynamic_framerate'],
                          check=True,capture_output=True,text=True)
    return int(result.stdout.rsplit(':',1)[1].strip())


def camera_set(device,value):
    subprocess.run(['v4l2-ctl','-d',device,f'--set-ctrl=exposure_dynamic_framerate={int(value)}'],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('show','apply','restore'))
    p.add_argument('--device',default='/dev/video0')
    p.add_argument('--cpu-performance',action='store_true',help='Optional; requires sudo, higher power consumption')
    p.add_argument('--fixed-frame-rate',action='store_true',help='Explicitly disable dynamic frame rate; may darken low-light images')
    p.add_argument('--state',type=Path,default=Path(__file__).resolve().parents[1]/'logs/board_settings_before.json')
    args=p.parse_args()
    policies=sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    current=dict(device=args.device,exposure_dynamic_framerate=camera_get(args.device),
                 governors={str(path):path.read_text().strip() for path in policies})
    if args.action=='show':
        print(json.dumps(current,indent=2));return
    if args.action=='apply':
        if not args.state.exists():
            args.state.parent.mkdir(parents=True,exist_ok=True)
            args.state.write_text(json.dumps(current,indent=2),encoding='utf-8')
        else:
            saved=json.loads(args.state.read_text(encoding='utf-8'))
            if saved['device']!=args.device:raise SystemExit('State belongs to a different camera; use another --state')
        camera_set(args.device,0 if args.fixed_frame_rate else 1)
        if args.cpu_performance:
            for path in policies:path.write_text('performance')
        print('Applied. Settings are not installed as a boot service. Original state:',args.state)
    else:
        saved=json.loads(args.state.read_text(encoding='utf-8'))
        for name,value in saved['governors'].items():
            path=Path(name)
            if path not in policies:raise SystemExit('CPU policy layout changed; restore manually')
            if value not in path.with_name('scaling_available_governors').read_text().split():
                raise SystemExit('Saved governor is no longer supported')
        camera_set(saved['device'],saved['exposure_dynamic_framerate'])
        for name,value in saved['governors'].items():Path(name).write_text(value)
        print('Restored:',args.state)


if __name__=='__main__':main()
