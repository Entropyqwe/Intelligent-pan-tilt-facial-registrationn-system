"""Short-lived optical flow for HEAD DISPLAY only, never identity/control.

Fresh YuNet/SFace results replace this cache. Failed flow clears a head box;
it never creates a control observation or refreshes an identity timestamp.
"""
import cv2
import numpy as np


class DisplayFaces:
    def __init__(self,max_age=.25):
        self.max_age=max_age
        self.gray=None
        self.faces=[]
        self.checked=0.

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(cv2.resize(frame,(320,240)),cv2.COLOR_BGR2GRAY)

    def update(self,frame,faces,captured,face_checked):
        gray=self._gray(frame)
        if face_checked:
            self.faces=[dict(f) for f in faces]
            self.gray=gray
            self.checked=captured
            return self.faces
        if self.gray is None or not 0<captured-self.checked<=self.max_age:
            self.faces=[]
            self.gray=gray
            return []
        height,width=frame.shape[:2]
        sx,sy=320/width,240/height
        moved=[]
        for face in self.faces:
            x,y,w,h=face['box']
            mask=np.zeros_like(gray)
            x1,y1=max(0,int(x*sx)),max(0,int(y*sy))
            x2,y2=min(320,int((x+w)*sx)),min(240,int((y+h)*sy))
            mask[y1:y2,x1:x2]=255
            points=cv2.goodFeaturesToTrack(self.gray,maxCorners=15,qualityLevel=.02,minDistance=3,mask=mask)
            if points is None or len(points)<4:continue
            new,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,winSize=(15,15),maxLevel=2)
            if new is None:continue
            back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,new,None,winSize=(15,15),maxLevel=2)
            if back is None:continue
            keep=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.)
            if keep.sum()<4:continue
            dx,dy=np.median((new-points)[keep,0],axis=0)
            dx,dy=float(dx)/sx,float(dy)/sy
            if abs(dx)>w*.3 or abs(dy)>h*.3:continue
            x,y=x+dx,y+dy
            if x<0 or y<0 or x+w>width or y+h>height:continue
            updated=dict(face,box=(x,y,w,h))
            moved.append(updated)
        self.faces=moved
        self.gray=gray
        return self.faces
