"""Same capture mode: at forward distance D, measure width W filling whole image."""
import argparse, json, math
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--distance-m',type=float,required=True)
p.add_argument('--visible-width-m',type=float,required=True)
p.add_argument('--save',default='steering_config.json')
a=p.parse_args()
if not all(math.isfinite(x) and x>0 for x in (a.distance_m,a.visible_width_m)):
    p.error('positive finite distances required')
fov=math.degrees(2*math.atan(a.visible_width_m/(2*a.distance_m)))
if not 5<fov<170: p.error('FOV outside supported range')
path=Path(a.save); c=json.loads(path.read_text(encoding='utf-8-sig'))
c['horizontal_fov_deg']=fov
c['camera_calibration_file']=None
path.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
print(f'Estimated horizontal FOV: {fov:.3f} deg; saved {path}')
print('Approximation only: lens distortion and principal point not calibrated.')
