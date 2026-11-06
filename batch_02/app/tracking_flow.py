"""Short-lived, forward/backward checked KLT assistance for body association.

This is image continuity, not identity or a detector. Its prediction never
becomes a control observation. All coordinates exposed here are source pixels.
"""
import cv2
import numpy as np


class BodyFlow:
    def __init__(self,max_age=.7):
        self.gray=None;self.timestamp=0.;self.states={};self.max_age=max_age

    @staticmethod
    def _small(frame):
        h,w=frame.shape[:2]
        small=cv2.resize(frame,(320,max(1,round(h*320/w))),interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),320/w

    def predict(self,frame,captured):
        gray,self.scale=self._small(frame);self.current=gray;result={}
        if self.gray is None or not 0<captured-self.timestamp<=.25:
            self.states={};return result
        for key,state in self.states.items():
            if captured-state['trusted']>self.max_age:continue
            points=state['points']
            if points is None or len(points)<8:continue
            nxt,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,
                winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,15,.03))
            if nxt is None:continue
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None,
                winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,15,.03))
            if back is None:continue
            good=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(points-back,axis=2).ravel()<1.)
            before=points[good].reshape(-1,2);after=nxt[good].reshape(-1,2)
            if len(before)<8 or len(before)<.55*len(points):continue
            shift=after-before;median=np.median(shift,axis=0)
            inlier=np.linalg.norm(shift-median,axis=1)<max(2.,.2*np.linalg.norm(median))
            before=before[inlier];after=after[inlier]
            if len(after)<8 or len(after)<.5*len(points):continue
            x,y,w,h=state['box'];centers=before/self.scale
            quadrants=(centers[:,0]>x+w/2).astype(int)+2*(centers[:,1]>y+h/2)
            # A single patch on the background is insufficient evidence.
            if len(np.unique(quadrants))<3:continue
            dx,dy=np.median(after-before,axis=0)/self.scale
            if np.hypot(dx,dy)>max(35.,h*.3):continue
            box=(x+float(dx),y+float(dy),w,h)
            state['box']=box;state['points']=after.reshape(-1,1,2).astype(np.float32)
            result[key]=dict(box=box,points=after/self.scale,quality=len(after)/len(points))
        return result

    def commit(self,tracks,captured):
        states={}
        for track in list(tracks)[:8]:
            key=id(track);old=self.states.get(key)
            # Refresh from real, unique high-score observations only.
            if track.control_ready:
                x,y,w,h=track.box;s=self.scale;ih,iw=self.current.shape
                mask=np.zeros_like(self.current)
                x1,x2=max(0,int((x+.12*w)*s)),min(iw,int((x+.88*w)*s))
                y1,y2=max(0,int((y+.12*h)*s)),min(ih,int((y+.92*h)*s))
                if x2>x1 and y2>y1:mask[y1:y2,x1:x2]=255
                points=cv2.goodFeaturesToTrack(self.current,maxCorners=48,qualityLevel=.01,
                                               minDistance=4,mask=mask,blockSize=5)
                if points is not None and len(points)>=8:
                    states[key]=dict(points=points,box=track.box,trusted=captured)
            elif old is not None and captured-old['trusted']<=self.max_age:
                states[key]=old
        self.states=states;self.gray=self.current;self.timestamp=captured
