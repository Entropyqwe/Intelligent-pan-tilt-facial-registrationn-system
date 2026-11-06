"""Native 960x660 HUD. Render text at final size; never downscale the canvas."""
import cv2
import time
import numpy as np
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BG=(18,18,16);TEXT=(245,245,240);MUTED=(182,182,170)
GREEN=(110,235,140);CYAN=(235,210,100);AMBER=(80,190,255)


@lru_cache(maxsize=32)
def font(size,bold):
    suffix='-Bold' if bold else ''
    # Noto TTC index 2 is Simplified Chinese; do not use a Latin-only fallback.
    paths=[(f'/usr/share/fonts/opentype/noto/NotoSansCJK{"-Bold" if bold else "-Regular"}.ttc',2),
           (f'C:/Windows/Fonts/{"msyhbd" if bold else "msyh"}.ttc',0),
           ('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',0)]
    for path,index in paths:
        if Path(path).exists():return ImageFont.truetype(path,size,index=index)
    raise RuntimeError('缺少中文字体，请安装 fonts-noto-cjk 后重新启动')


@lru_cache(maxsize=512)
def character(value,pixels,bold):
    face=font(pixels,bold)
    box=face.getbbox(value,anchor='ls')
    mask=Image.new('L',(max(1,box[2]-box[0]),max(1,box[3]-box[1])),0)
    ImageDraw.Draw(mask).text((-box[0],-box[1]),value,font=face,anchor='ls',fill=255)
    return np.asarray(mask),box,float(face.getlength(value))


@lru_cache(maxsize=256)
def glyph(value,pixels,bold):
    # Telemetry changes almost every frame. Cache the small glyph alphabet,
    # then compose masks; do not rerun FreeType for each new floating number.
    positioned=[];cursor=0.
    for char in value:
        mask,box,advance=character(char,pixels,bold)
        positioned.append((round(cursor)+box[0],box[1],mask));cursor+=advance
    if not positioned:return np.zeros((1,1),np.uint8),(0,0,1,1)
    left=min(x for x,y,m in positioned);top=min(y for x,y,m in positioned)
    right=max(x+m.shape[1] for x,y,m in positioned)
    bottom=max(y+m.shape[0] for x,y,m in positioned)
    result=np.zeros((bottom-top,right-left),np.uint8)
    for x,y,mask in positioned:
        roi=result[y-top:y-top+mask.shape[0],x-left:x-left+mask.shape[1]]
        np.maximum(roi,mask,out=roi)
    return result,(left,top,right,bottom)


def write(image,value,x,y,size=.55,color=TEXT,bold=False):
    mask,box=glyph(str(value),max(15,round(size*32)),bold)
    x+=box[0];y+=box[1]
    h,w=mask.shape;ih,iw=image.shape[:2]
    x1,y1=max(0,x),max(0,y);x2,y2=min(iw,x+w),min(ih,y+h)
    if x2<=x1 or y2<=y1:return
    alpha=mask[y1-y:y2-y,x1-x:x2-x,None].astype(np.uint16)
    roi=image[y1:y2,x1:x2]
    roi[:]=((roi.astype(np.uint16)*(255-alpha)+np.asarray(color,np.uint16)*alpha+127)//255).astype(np.uint8)


class BodyHUD:
    width=960
    height=660
    origin=(16,86)
    view_size=(640,480)

    def __init__(self):
        self.source_size=(640,480)
        self.panel_cache=None
        self.panel_key=None
        self.panel_updated=0.
        self.template=np.full((self.height,self.width,3),BG,np.uint8)
        write(self.template,'人体跟踪 · 转向协同',16,34,.8,bold=True)
        write(self.template,'NPU 人体检测 / 衣着与运动连续性跟踪',16,59,.47,MUTED)
        cv2.line(self.template,(16,70),(944,70),(65,65,58),1)
        for value,y in (('实际处理帧率',112),('当前目标编号',226),('相对车身偏转角度',326),('超声波回波距离',410),('计算耗时 / 通信',505)):
            write(self.template,value,680,y,.48,CYAN)
        write(self.template,'点击人体框选择目标   A：自动/手动   U：解除锁定   Q：退出',16,606,.51)
        write(self.template,'单人编号 01 为跟随槽位；重新入镜会重选，不代表同一身份。',16,634,.48,MUTED)

    def to_image(self,x,y):
        x-=self.origin[0];y-=self.origin[1]
        return (x*self.source_size[0]/640,y*self.source_size[1]/480) if 0<=x<640 and 0<=y<480 else None

    def render(self,frame,tracks,target,stats,bearing,auto,port,elapsed,distance=None,serial_error=None,motor=None):
        height,width=frame.shape[:2];self.source_size=(width,height)
        canvas=self.template.copy();ox,oy=self.origin
        view=canvas[oy:oy+480,ox:ox+640]
        view[:]=frame if (width,height)==(640,480) else cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
        sx,sy=640/width,480/height
        for track in tracks:
            x,y,w,h=track['box'];x,y,w,h=int(x*sx),int(y*sy),int(w*sx),int(h*sy)
            selected=track['person_id']==target.target_id
            color=GREEN if selected and target.locked else (CYAN if track['confirmed'] else AMBER)
            cv2.rectangle(view,(x,y),(x+w,y+h),color,2,cv2.LINE_AA)
            label=(f"人体 {track['person_id']:02d}" if track['person_id']>0 else '人体候选')
            if not track['confirmed']:label+=' / 确认中'
            # Text belongs to the body box; no face/registration geometry.
            px=max(2,min(640-200,x+4));py=max(20,min(476,y+22))
            # OpenCV Hershey fonts cannot draw Chinese; use a dark label plate.
            mask,box=glyph(label,max(15,round(.55*32)),True)
            cv2.rectangle(view,(px-2,py+box[1]-3),(min(639,px+box[2]+3),min(479,py+box[3]+3)),(8,8,8),-1)
            write(view,label,px,py,.55,color,bold=True)
        cv2.line(view,(314,240),(326,240),TEXT,1)
        cv2.line(view,(320,234),(320,246),TEXT,1)
        # Update expensive text at 10Hz while camera and detections stay live.
        # Stop/invalid/target changes bypass the timer immediately.
        d=distance or dict(mm=0,valid=False,mode='NO_DATA')
        m=motor or dict(valid=False)
        panel_key=(target.target_id,target.mode,target.locked,auto,bool(port),bool(serial_error),
                   d.get('valid'),d.get('mode'),m.get('valid'),m.get('mode'))
        panel_now=time.monotonic()
        if self.panel_cache is not None and panel_key==self.panel_key and panel_now-self.panel_updated<.1:
            canvas[86:,670:]=self.panel_cache
        else:
            write(canvas,f"{stats['fps']:.1f}",680,163,1.35,bold=True)
            write(canvas,f"相机 {stats['camera_fps']:.1f} 帧/秒",680,191,.55)
            write(canvas,f"{target.target_id:02d}" if target.target_id else '--',680,269,1.1,bold=True)
            mode={'STANDBY':'待选目标','VERIFY':'确认中','LOST':'目标丢失','BODY':'跟踪中'}.get(target.mode,'未知状态')
            write(canvas,f"{mode} / {'自动' if auto else '手动'}",680,298,.55,GREEN if target.locked else AMBER)
            write(canvas,f'{bearing:+.1f} 度' if bearing is not None else '--.- 度',680,371,.95,bold=True)
            d=distance or dict(mm=0,valid=False,mode='无测距')
            if d.get('valid'):
                write(canvas,f"{d.get('mm',0)/10:.1f} cm",680,454,.88,bold=True)
            else:
                write(canvas,'--.- cm',680,454,.88,bold=True)
                age_ms=int(d.get('age_ms',999999))
                if age_ms >= 900000:
                    message='未收到 D1 距离帧'
                elif d.get('mode')=='STALE' or age_ms >= 500:
                    message='D1 数据超时'
                else:
                    message=f"ECHO 无效 / {d.get('mode','UNKNOWN')}"
                write(canvas,message,680,480,.45,AMBER)
            labels={'TRACK_READY':'前进','START_WAIT':'等待稳定/≥35cm','STOP_NEAR':'距离≤30cm停车',
                    'STOP_VISUAL':'视觉无效停车','STOP_SENSOR':'测距无效停车','FAULT':'故障停车','SEARCH':'等待','STOP_AIM':'云台/目标未对准'}
            state=(motor or {}).get('mode') if (motor or {}).get('valid') else d.get('mode','NO_DATA')
            if d.get('valid'):
                write(canvas,labels.get(state,'等待主控状态'),680,480,.45,GREEN if state=='TRACK_READY' else AMBER)
            if (motor or {}).get('valid'):
                write(canvas,f"舵机输出 {motor['applied_us']}μs",680,528,.50)
                write(canvas,f"电机PWM {motor['left_pwm']/10:.1f}% / 目标{motor['target_pwm']/10:.0f}%",680,551,.43)
                write(canvas,'主控命令值 / 非实测转速',680,572,.40,MUTED)
            else:
                write(canvas,f"检测 {stats['body_ms']:.1f} / 匹配 {stats['appearance_ms']:.1f} ms",680,531,.45)
                write(canvas,f"延迟 {stats['age_ms']:.0f}ms / 暂无M1反馈",680,555,.43)
            if serial_error:
                write(canvas,'串口断开 / 正在重连',680,594,.40,AMBER)
            else:
                write(canvas,'串口 S1 / 主控遥测' if port else '仅预览 / 串口未启用',680,594,.40,GREEN if port else AMBER)
            self.panel_cache=canvas[86:,670:].copy()
            self.panel_key=panel_key
            self.panel_updated=panel_now
        g=stats.get('gimbal')
        if g:
            canvas[608:660,:]=BG
            pan,tilt=g.get('pan_deg',0.),g.get('tilt_deg',0.)
            write(canvas,f'云台估计：水平 {pan:+.1f}° / 俯仰 {tilt:+.1f}°   '+g.get('reason','等待'),16,631,.49,AMBER)
            wheel=g.get('wheel_deg')
            qualifier='已对准的回波候选' if g.get('range_candidate') else '回波尚未对准/验证'
            if g.get('pan_at_limit'):
                pulse=g.get('pan_command_us',g.get('pan_us'))
                pan_state='水平到限位'+(f' {int(pulse)}μs' if pulse is not None else '')
            else:
                pan_state={'HOLD':'水平稳像保持','TRACK':'水平跟随中'}.get(g.get('pan_state'),'水平待确认')
            if g.get('tilt_at_limit'):
                tilt_state=f"俯仰到限位{g.get('tilt_command_us')}μs"
            else:
                tilt_state={'HOLD':'上半身构图保持','TRACK':'俯仰构图调整'}.get(g.get('tilt_state'),'俯仰待确认')
            write(canvas,pan_state+'；'+tilt_state+'；'+('前轮 '+f'{wheel:+.1f}°' if wheel is not None else '前轮待标定')+'；'+qualifier+'（未验身份）',16,654,.43,MUTED)
        write(canvas,f'T+{int(elapsed)//60:02d}:{int(elapsed)%60:02d}',814,30,.6)
        return canvas
