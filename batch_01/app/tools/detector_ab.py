"""Photometric A/B on same frames; counts measure yield, not precision/recall."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from yolo_npu import YoloNpuVision

def main():
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cv2.setNumThreads(2);net=YoloNpuVision(threshold=.1)
 clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
 tables={g:np.round((np.arange(256)/255.)**g*255).astype(np.uint8) for g in (1.3,.7)}
 variants=('original','clahe','gamma1.3','gamma0.7')
 cap=cv2.VideoCapture(a.video);frames=[];i=0
 try:
  while True:
   ok,f=cap.read()
   if not ok:break
   if i%5==0:frames.append((i,f))
   i+=1
  results={v:[] for v in variants};tiles=[]
  for index,(i,frame) in enumerate(frames):
   row=[]
   for name in variants:
    start=time.monotonic()
    if name=='clahe':
     lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB);lab[:,:,0]=clahe.apply(lab[:,:,0]);image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    elif name.startswith('gamma'):image=cv2.LUT(frame,tables[float(name[5:])])
    else:image=frame
    pre=(time.monotonic()-start)*1000;bodies=net.process(image)
    best=max((b['score'] for b in bodies),default=0.)
    results[name].append(dict(frame=i,best=best,high_count=sum(b['score']>=.35 for b in bodies),pre_ms=pre))
    if index%8==0 and len(tiles)<40:
     tile=image.copy()
     for body in bodies:
      if body['score']<.35:continue
      x,y,w,h=map(int,body['box']);cv2.rectangle(tile,(x,y),(x+w,y+h),(0,255,0),2)
     tile=cv2.resize(tile,(320,240));cv2.rectangle(tile,(0,0),(320,23),(0,0,0),-1)
     cv2.putText(tile,f'{i} {name} {best:.2f}',(3,17),0,.45,(255,255,255),1);row.append(tile)
   if row:tiles.extend(row)
  summary={v:dict(frames=len(r),high_fraction=np.mean([x['best']>=.35 for x in r]),mean_best=np.mean([x['best'] for x in r]),mean_pre_ms=np.mean([x['pre_ms'] for x in r])) for v,r in results.items()}
  a.output.write_text(json.dumps(dict(summary=summary,frames=results),indent=2),encoding='utf8')
  if tiles:
   sheet=np.vstack([np.hstack(tiles[j:j+4]) for j in range(0,len(tiles),4)])
   cv2.imwrite(str(a.output.with_suffix('.jpg')),sheet)
  print(json.dumps(summary,indent=2),flush=True)
 finally:cap.release();net.close()
if __name__=='__main__':main()
