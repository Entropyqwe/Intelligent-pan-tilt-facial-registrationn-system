"""Lightweight launch-telemetry-inspired interface; no logos or extra ID box.

All detections stay in the original image coordinate system. The camera view
is translated, never stretched; mouse picks are translated back by to_image.
"""
from collections import deque
import cv2
import numpy as np

BG=(14,12,11)
PANEL=(25,22,19)
TEXT=(239,238,231)
MUTED=(151,143,131)
LINE=(69,61,51)
CYAN=(239,211,114)
GREEN=(165,236,170)
AMBER=(106,189,249)
RED=(97,101,244)
FONT=cv2.FONT_HERSHEY_SIMPLEX


def text(image,value,point,size=.45,color=TEXT,weight=1):
    cv2.putText(image,str(value),point,FONT,size,color,weight,cv2.LINE_AA)


def corners(image,box,color,thickness=1):
    x,y,w,h=map(int,box)
    length=max(6,min(18,w//4,h//4))
    for sx,sy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(image,(sx,sy),(sx+dx*length,sy),color,thickness,cv2.LINE_AA)
        cv2.line(image,(sx,sy),(sx,sy+dy*length),color,thickness,cv2.LINE_AA)


class TelemetryHUD:
    def __init__(self):
        self.origin=(20,88)
        self.history=deque(maxlen=100)
        self.template=None
        self.view_size=None

    def to_image(self,x,y):
        if self.view_size is None:return None
        x-=self.origin[0];y-=self.origin[1]
        w,h=self.view_size
        return (x,y) if 0<=x<w and 0<=y<h else None

    def _layout(self,width,height):
        if self.view_size==(width,height):return
        self.view_size=(width,height)
        total_width=width+280
        total_height=max(height+166,640)
        self.template=np.full((total_height,total_width,3),BG,np.uint8)
        canvas=self.template
        text(canvas,'FACELOCK / FLIGHT DECK',(20,29),.64,TEXT,1)
        text(canvas,'VISION GUIDANCE  /  ORANGE PI ZERO 3W',(20,53),.36,MUTED)
        cv2.line(canvas,(20,68),(total_width-20,68),LINE,1)
        x=width+44
        cv2.line(canvas,(x-13,88),(x-13,total_height-75),LINE,1)
        text(canvas,'LIVE TELEMETRY',(x,105),.42,CYAN)
        text(canvas,'FRAME RATE',(x,138),.34,MUTED)
        text(canvas,'TARGET / IDENTITY',(x,220),.34,MUTED)
        text(canvas,'HORIZONTAL BEARING',(x,299),.34,MUTED)
        text(canvas,'PROCESSING',(x,384),.34,MUTED)
        text(canvas,'SERIAL CONTROL',(x,461),.34,MUTED)
        cv2.line(canvas,(20,total_height-66),(total_width-20,total_height-66),LINE,1)
        text(canvas,'N REGISTER  /  CLICK OR 1-9 SELECT  /  A AUTO  /  U UNLOCK  /  Q EXIT',
             (20,total_height-43),.34,MUTED)

    def render(self,frame,faces,bodies,target,snapshot,fields,stats,status,
               bearing=None,backend='CPU',port='',elapsed=0.,face_age=0.,
               serial_error=False,auto_enabled=False):
        height,width=frame.shape[:2]
        self._layout(width,height)
        canvas=self.template.copy()
        ox,oy=self.origin
        view=canvas[oy:oy+height,ox:ox+width]
        view[:]=frame
        valid=bool(fields['valid'])
        for body in bodies:
            x,y,w,h=body['box']
            active=valid and getattr(target,'box',None)==body['box']
            color=GREEN if active else CYAN
            corners(view,(x,y,w,h),color,2 if active else 1)
            # The body has no identity label or separate marking rectangle.
        for face in faces:
            x,y,w,h=map(int,face['box'])
            pid=face['person_id']
            active=pid>0 and pid==target.target_id
            color=GREEN if active else AMBER
            cv2.rectangle(view,(x,y),(x+w,y+h),color,1,cv2.LINE_AA)
            # Identity text belongs to the head box; no third geometry/ID box.
            label=f'ID {pid:02d}' if pid else 'UNKNOWN'
            label+=f" | {face['similarity']:.2f}"
            scale=max(.22,min(.40,(w-6)/max(cv2.getTextSize(label,FONT,1.,1)[0][0],1)))
            text(view,label,(x+3,min(height-3,y+13)),scale,color)
        # Tiny neutral reticle rather than the former full-height blue line.
        cx,cy=width//2,height//2
        for dx,dy in [(-7,0),(4,0),(0,-7),(0,4)]:
            cv2.line(view,(cx+dx,cy+dy),(cx+dx+(3 if dy==0 else 0),cy+dy+(3 if dx==0 else 0)),TEXT,1)
        if valid and snapshot.observation:
            o=snapshot.observation
            px=int(o.x+o.w/2); py=int(o.y+o.h/2)
            cv2.circle(view,(px,py),3,GREEN,1,cv2.LINE_AA)
        mode=((getattr(target,'mode','LOCKED') if valid else 'VERIFY / LOST')
              if target.target_id else 'STANDBY')
        x=width+44
        text(canvas,f"{stats['fps']:.1f}",(x,179),1.05,TEXT,1)
        text(canvas,'FPS / PROCESSED',(x+103,176),.29,MUTED)
        text(canvas,f"CAM {stats['camera_fps']:.1f}  FACE {stats['face_fps']:.1f}",(x,198),.34,MUTED)
        text(canvas,f'{target.target_id:02d}' if target.target_id else '--',(x,254),.85,TEXT)
        text(canvas,mode,(x,276),.39,GREEN if valid else AMBER)
        text(canvas,'AUTO' if auto_enabled else 'MANUAL',(x+126,252),.31,MUTED)
        label=f'{bearing:+.1f}' if bearing is not None and valid else '--.-'
        text(canvas,label,(x,339),.95,TEXT)
        text(canvas,'DEG',(x+144,337),.34,MUTED)
        baseline=360
        cv2.line(canvas,(x,baseline),(x+210,baseline),LINE,1)
        for pos in (0,52,105,157,210):
            cv2.line(canvas,(x+pos,baseline-3),(x+pos,baseline+3),MUTED,1)
        if bearing is not None and valid:
            pos=int(x+105+np.clip(bearing,-45,45)/45*105)
            cv2.circle(canvas,(pos,baseline),3,GREEN,-1,cv2.LINE_AA)
        text(canvas,'NPU / YOLOv5' if 'NPU' in backend else 'CPU / NanoDet',(x,407),.42,CYAN)
        text(canvas,f"FACE {stats['face_ms']:.1f} ms  BODY {stats['body_ms']:.1f} ms",(x,428),.31,MUTED)
        identity_age=f"{fields.get('face_age_ms',0)}" if target.target_id and snapshot.face_confirmed else '--'
        text(canvas,f"AGE {stats['age_ms']:.0f} ms  ID AGE {identity_age} ms",(x,446),.30,MUTED)
        link='OUTPUT ERROR' if serial_error else ('S1 / 115200 / 10Hz' if port else 'PREVIEW / NO SERIAL')
        text(canvas,link,(x,486),.36,RED if serial_error else (GREEN if port else MUTED))
        text(canvas,'TX STATUS / NOT SERVO FEEDBACK',(x,505),.27,MUTED)
        self.history.append(stats['fps'])
        if len(self.history)>1:
            points=np.array([(x+i*210/(len(self.history)-1),552-min(60,v*1.5))
                             for i,v in enumerate(self.history)],np.int32)
            cv2.polylines(canvas,[points],False,CYAN,1,cv2.LINE_AA)
        text(canvas,'20 FPS MIN / 24 FPS TARGET',(x,570),.29,MUTED)
        total_height,total_width=canvas.shape[:2]
        text(canvas,f'T+{int(elapsed)//60:02d}:{int(elapsed)%60:02d}',(total_width-136,30),.51,TEXT)
        text(canvas,'LIVE / '+('TRACK' if valid else 'ACQUIRE'),(total_width-136,53),.32,GREEN if valid else AMBER)
        text(canvas,status[:105],(20,total_height-20),.36,TEXT)
        return canvas
