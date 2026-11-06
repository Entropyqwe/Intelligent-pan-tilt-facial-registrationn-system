"""Face names rendered in the existing native-pixel Chinese HUD."""
import cv2
from body_hud import write, GREEN, AMBER, BG


def overlay_faces(canvas, frame, tracks, labels, error, speech_status, follow_id=0):
    # Face boxes are projected from the recognition-frame body to the current
    # continuous body, not drawn at stale image coordinates. They are a label,
    # never an extra measurement for vehicle steering.
    h,w=frame.shape[:2]
    view=canvas[86:566,16:656]
    sx,sy=640/w,480/h
    for tid,row in labels.items():
        color=GREEN if row['verified'] else AMBER
        box=row['box']
        track=next((t for t in tracks if t['person_id']==tid),None)
        if track is None:continue
        if box:
            x,y,bw,bh=box
            x1=max(0,min(639,round(x*sx)));y1=max(0,min(479,round(y*sy)))
            x2=max(0,min(639,round((x+bw)*sx)));y2=max(0,min(479,round((y+bh)*sy)))
            cv2.rectangle(view,(x1,y1),(x2,y2),color,1)
            px,py=max(3,min(340,x1)),max(45,min(470,y2+20))
        else:
            x,y,_,_=track['box'];px,py=max(3,min(340,round(x*sx))),max(45,min(470,round(y*sy)+45))
        text='人脸：'+row['name']
        cv2.rectangle(view,(px-2,py-19),(min(639,px+len(text)*18),py+4),BG,-1)
        write(view,text,px,py,.52,color,bold=True)
    canvas[38:68,16:944]=BG
    status='人脸停用：请检查模型和日志' if error else ('人脸低频识别 / '+speech_status)
    if follow_id:status+=f' / 仅跟随人脸ID {follow_id}'
    write(canvas,status[:65],16,59,.45,AMBER if error else GREEN)
