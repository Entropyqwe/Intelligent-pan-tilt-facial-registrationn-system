"""Calibrate from 15+ checkerboard images, same resolution/focus as tracking."""
import argparse, glob, json
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--images',required=True,help='quoted glob, e.g. calibration/*.jpg')
p.add_argument('--cols',type=int,default=9,help='inner corners, not square count')
p.add_argument('--rows',type=int,default=6)
p.add_argument('--square-mm',type=float,default=25.)
p.add_argument('--output',default='camera_calibration.json')
a=p.parse_args()
if a.cols<3 or a.rows<3 or a.square_mm<=0: p.error('invalid board dimensions')
obj=np.zeros((a.cols*a.rows,3),np.float32)
obj[:,:2]=np.mgrid[0:a.cols,0:a.rows].T.reshape(-1,2)*a.square_mm
objects=[]; pixels=[]; size=None
for name in sorted(glob.glob(a.images)):
    im=cv2.imread(name)
    if im is None: continue
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    if size is None: size=g.shape[::-1]
    if g.shape[::-1]!=size: raise SystemExit('Mixed image resolutions')
    ok,corners=cv2.findChessboardCorners(g,(a.cols,a.rows))
    if not ok: print('Skip (no corners)',name); continue
    corners=cv2.cornerSubPix(g,corners,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.001))
    objects.append(obj.copy()); pixels.append(corners)
if len(objects)<15: raise SystemExit(f'Need >=15 good images, found {len(objects)}')
rms,k,d,_,_=cv2.calibrateCamera(objects,pixels,size,None,None)
result=dict(image_width=size[0],image_height=size[1],camera_matrix=k.tolist(),dist_coeffs=d.reshape(-1).tolist(),rms_pixels=rms,image_count=len(objects))
Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
print('RMS reprojection error (pixels):',rms,'saved',a.output)
print('Inspect coverage and independent test angles; RMS alone is not acceptance.')
