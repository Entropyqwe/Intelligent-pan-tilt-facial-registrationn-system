"""Explicit local enrollment. Camera only: never imports/open serial control."""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from named_faces import NamedGallery
from vision import FaceVision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name')
    p.add_argument('--list',action='store_true')
    p.add_argument('--delete',type=int)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--gallery',type=Path,default=ROOT/'data/named_faces.json')
    p.add_argument('--frame-rotation',type=int,choices=(0,180),default=180)
    p.add_argument('--no-show',action='store_true')
    a=p.parse_args()
    gallery=NamedGallery(a.gallery)
    if a.list:
        for pid,row in sorted(gallery.people.items()):
            print(f"ID={pid} 姓名={row['name']} 样本={len(row['samples'])}")
        if not gallery.people:print('尚未登记人员。')
        return
    if a.delete is not None:
        gallery.delete(a.delete);print('已删除人脸ID',a.delete);return
    if not a.name:p.error('请用 --name 姓名 显式登记')
    from named_faces import checked_name
    checked_name(a.name)
    from main import LatestCamera
    from body_hud import write, GREEN, AMBER
    detector=FaceVision(ROOT,threads=2,max_faces=5,reject_overflow=True)
    # Enrollment rejects extra faces even when they would be too small for SFace.
    camera=LatestCamera(a.camera,640,480,30,camera_profile='quality',frame_rotation=a.frame_rotation)
    samples=[];seq=0;started=time.monotonic();last=0.;message='请只让待登记的一张正脸入镜'
    try:
        while time.monotonic()-started<60 and len(samples)<8:
            if camera.failed:raise RuntimeError('摄像头掉线')
            frame,stamp,number=camera.latest(seq)
            if frame is None:time.sleep(.01);continue
            seq=number
            if time.monotonic()-stamp>.4:continue
            faces=detector.process(frame,gallery)
            # YuNet raw count prevents a second small face being silently ignored.
            size=detector.detector_size
            detector.detector.setInputSize((size,size))
            _,raw=detector.detector.detect(cv2.resize(frame,(size,size)))
            total=0 if raw is None else len(raw)
            if total==1 and len(faces)==1:
                face=faces[0];x,y,w,h=face['box']
                gray=cv2.cvtColor(face['aligned'],cv2.COLOR_BGR2GRAY)
                usable=(min(w,h)>=75 and 45<=float(gray.mean())<=215 and
                        float(cv2.Laplacian(gray,cv2.CV_64F).var())>=35 and
                        (not samples or max(float(face['feature']@v) for v in samples)>=.55))
                if usable and time.monotonic()-last>=.4:
                    samples.append(face['feature']);last=time.monotonic()
                    message=f'采样 {len(samples)}/8：请缓慢小幅转头'
                elif not usable:
                    message='脸太小/暗/亮/模糊/身份不连续，请调整；Q取消'
                cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
            else:
                message='需要且只允许一张清楚人脸；请其他人退出画面'
            if not a.no_show:
                write(frame,message,12,30,.52,GREEN if len(faces)==1 else AMBER)
                cv2.imshow('Named face enrollment / Q cancel',frame)
                if cv2.waitKey(1)&255 in (ord('q'),27):
                    print('登记已取消，未写入。');return
                if cv2.getWindowProperty('Named face enrollment / Q cancel',cv2.WND_PROP_VISIBLE)<1:return
            print(message,flush=True)
            time.sleep(.05)
        if len(samples)<8:raise RuntimeError('60秒内未采满8个合格样本，未写入姓名库')
        pid=gallery.add(a.name,samples)
        print(f'登记成功：ID={pid} / {a.name} / 样本={len(samples)}；未保存人脸照片。',flush=True)
    finally:
        camera.close()
        if not a.no_show:cv2.destroyAllWindows()


if __name__=='__main__':main()
