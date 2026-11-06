"""Chinese click-to-tune exposure window. Never opens a serial device."""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import LatestCamera
from body_hud import write,BG,TEXT,MUTED,GREEN,AMBER,CYAN
from camera_profiles import apply_profile
from camera_exposure import (EDITABLE,SAVED_PROFILE,control_info,capture_controls,set_control,
                             freeze_auto,apply_saved,save_profile,same_controls,apply_controls)
from exposure_quality import measure,average,rejection

NAMES=dict(exposure_time_absolute='曝光值',brightness='亮度',gain='增益',gamma='伽马',
           contrast='对比度',saturation='饱和度')


def run(index=0,max_seconds=0):
    cv2.setNumThreads(2)
    window='Exposure tuning / no serial'
    actions=deque();buttons=[];stopping=False
    camera=None;controls={};info={};tested=None;testing=None
    message='默认自动曝光。可先读取当前曝光，再转为固定并微调。'
    seq=0;history=deque();last_read=0.;settle_until=time.monotonic()+1
    startup=time.monotonic();screenshot=None
    auto_samples=deque(maxlen=12);auto_reference=None;last_quality=0.;last_reconnect=0
    def stop(*_):
        nonlocal stopping
        stopping=True
    def mouse(event,x,y,*_):
        if event==cv2.EVENT_LBUTTONDOWN:
            for x1,y1,x2,y2,key in buttons:
                if x1<=x<=x2 and y1<=y<=y2:actions.append(key);break
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    try:
        camera=LatestCamera(index,640,480,30,camera_profile='quality')
        info=control_info(index);controls=capture_controls(index)
        cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE|cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowTitle(window,'曝光调试 · 不发送舵机或电机控制')
        cv2.setMouseCallback(window,mouse)
        while not stopping:
            if camera.failed:raise RuntimeError('摄像头重连失败，请接回后重新打开调试。')
            frame,stamp,number=camera.latest(seq)
            now=time.monotonic()
            if frame is None:
                # Keep close/quit responsive during USB reconnect.
                if cv2.waitKey(1)&255 in (27,ord('q')) or (screenshot is not None and cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1):break
                if max_seconds and now-startup>=max_seconds:break
                time.sleep(.005)
                continue
            seq=number;history.append((stamp,number))
            while len(history)>2 and history[1][0]<stamp-2:history.popleft()
            fps=(history[-1][1]-history[0][1])/max(.001,history[-1][0]-history[0][0])
            if now-last_read>1:
                controls=capture_controls(index);last_read=now
            if camera.reconnect_count!=last_reconnect:
                last_reconnect=camera.reconnect_count;tested=testing=None
                auto_samples.clear();auto_reference=None
                message='相机已重连并恢复自动曝光，请重新取得基准并测试。'
            if controls.get('auto_exposure')!=1 and now-startup>=2 and now-last_quality>=.25:
                auto_samples.append(measure(frame));auto_reference=frame.copy();last_quality=now
            if actions:
                action=actions.popleft()
                try:
                    if action=='quit':break
                    if action=='test':
                        controls=capture_controls(index)
                        if controls['auto_exposure']!=1:raise ValueError('先点击“读取自动并固定”，再测试固定曝光。')
                        if len(auto_samples)<5:raise ValueError('缺少自动曝光基准，先恢复自动并等待约4秒。')
                        tested=None
                        # Verify the values can be replayed before saving.
                        apply_controls(index,controls)
                        replay=capture_controls(index)
                        if not same_controls(controls,replay):raise ValueError('相机参数重载不一致，不能测试保存。')
                        testing=dict(controls=controls.copy(),start=max(now,settle_until)+1,frames=0,
                                     first=None,last=None,first_seq=None,last_seq=None,quality=[],
                                     baseline=average(list(auto_samples)))
                        message='重载参数后测试5秒：检查曝光、紫色色偏及噪点。不合格会恢复自动。'
                    elif action=='save':
                        if not tested:raise ValueError('请先测试当前参数；修改后需要重新测试。')
                        reason=rejection(tested['stats'],tested['stats']['auto_baseline'])
                        if reason:raise ValueError('画质不合格，禁止保存：'+reason)
                        approved=subprocess.run(['zenity','--question','--title=确认固定曝光画质',
                          '--width=580','--text=请对照主画面与左下角自动曝光基准：人物细节、色彩清楚，没有明显紫色噪点、过曝或拖影吗？\n\n只有确认画质可用才保存。帧率不是唯一标准。',
                          '--ok-label=画质可用，保存','--cancel-label=画质不合格，不保存']).returncode==0
                        if not approved:message='未保存。请继续调整，或使用自动曝光。';continue
                        save_profile(index,tested['controls'],tested['stats'],quality_confirmed=True)
                        message='已保存！关闭调试后，可选择“已保存固定曝光”启动。'
                        print('SAVED_FIXED_EXPOSURE',SAVED_PROFILE,flush=True)
                    else:
                        testing=None;tested=None
                        if action=='auto':apply_profile(index,'quality');message='已恢复自动曝光；不会覆盖保存的固定档。'
                        elif action=='freeze':
                            if len(auto_samples)<5:raise ValueError('请先恢复自动曝光并等待约4秒取得画质基准。')
                            freeze_auto(index);message='仅切换手动并保留内部状态；测试时会验证参数能否重载。'
                        elif action=='load':apply_saved(index);message='已载入保存数据；可直接退出后用固定档启动，或重新微调测试。'
                        elif action in EDITABLE:
                            spec=info[action]
                            if action=='exposure_time_absolute' and controls['auto_exposure']!=1:
                                raise ValueError('先读取自动曝光并转为固定，再编辑曝光值。')
                            result=subprocess.run(['zenity','--entry','--title='+NAMES[action],
                                '--text='+f"请输入整数，范围 {spec['min']}～{spec['max']}，步长 {spec.get('step',1)}。",'--entry-text='+str(controls[action])],
                                capture_output=True,text=True)
                            if result.returncode==0:
                                value=int(result.stdout.strip());step=max(1,spec.get('step',1))
                                if not spec['min']<=value<=spec['max'] or (value-spec['min'])%step:
                                    raise ValueError('数值不符合相机范围或步长。')
                                set_control(index,action,value);message=NAMES[action]+'已更新，请重新测试后保存。'
                        controls=capture_controls(index);settle_until=time.monotonic()+1
                        history.clear()
                except Exception as exc:message='操作未完成：'+str(exc)
            if testing and stamp>=testing['start']:
                test=testing;test['frames']+=1
                if test['first'] is None:test['first']=stamp;test['first_seq']=number
                test['last']=stamp;test['last_seq']=number
                # Native pixels preserve chroma speckle that sparse sampling hides.
                if not test['quality'] or test['frames']%3==0:test['quality'].append(measure(frame))
                if stamp-test['first']>=5:
                    seconds=stamp-test['first'];current=capture_controls(index)
                    if not same_controls(test['controls'],current):message='参数发生变化，测试已作废，请重新测试。'
                    else:
                        stats=dict(seconds=seconds,frames=test['frames'],
                                   fps=(test['last_seq']-test['first_seq'])/seconds,
                                   preview_fps=(test['frames']-1)/seconds,
                                   **average(test['quality']),auto_baseline=test['baseline'],replay_verified=True)
                        reason=rejection(stats,test['baseline'])
                        if reason:
                            tested=None;apply_profile(index,'quality');controls=capture_controls(index)
                            message='固定曝光不可用：'+reason+'；已恢复自动，禁止保存。'
                        else:
                            tested=dict(controls=current,stats=stats)
                            message=f"画质筛查通过：采集{stats['fps']:.2f}FPS。对照基准确认清楚后可保存。"
                        print('EXPOSURE_TEST',json.dumps(stats),flush=True)
                    testing=None
            canvas=np.full((710,980,3),BG,np.uint8);canvas[70:550,16:656]=frame
            if controls.get('auto_exposure')==1 and auto_reference is not None:
                canvas[412:532,24:184]=cv2.resize(auto_reference,(160,120))
                write(canvas,'自动曝光基准',26,407,.43,CYAN)
            write(canvas,'曝光调试 · 彩色640×480 · 串口关闭',16,34,.8,bold=True)
            write(canvas,f"实际采集 {fps:.1f} FPS / {'自动曝光' if controls.get('auto_exposure')!=1 else '固定曝光'}",16,65,.6,CYAN)
            write(canvas,'点击数值编辑，修改后重新测试',680,96,.43,MUTED)
            buttons=[]
            def button(key,label,x,y,w=280,h=42):
                cv2.rectangle(canvas,(x,y),(x+w,y+h),(55,55,48),-1)
                write(canvas,label,x+10,y+28,.50,TEXT)
                buttons.append((x,y,x+w,y+h,key))
            for i,name in enumerate(EDITABLE):
                if name in controls:button(name,f'{NAMES[name]}：{controls[name]}',680,116+i*50)
            button('auto','恢复自动曝光',680,430)
            button('freeze','读取自动曝光并转为固定',680,480)
            button('load','载入已保存的固定曝光',680,530)
            button('test','测试当前参数（约5秒）',16,590,310)
            button('save','保存已测试的参数',342,590,310)
            button('quit','退出调试 / 恢复自动曝光',680,590)
            for line,text in enumerate((message[:58],message[58:116])):
                if text:write(canvas,text,16,659+line*23,.44,GREEN if message.startswith('已保存') else AMBER)
            write(canvas,'请观察色彩、噪点与拖影。固定档仅在单独入口使用；默认视觉程序保持自动曝光。',16,702,.44,MUTED)
            if testing:write(canvas,f"测试中 {max(0.,stamp-testing['start']):.1f}/5秒",16,551,.55,GREEN)
            if tested:write(canvas,f"已测试 {tested['stats']['fps']:.2f} FPS / 平均亮度 {tested['stats']['mean_luma']:.0f}",16,580,.48,CYAN)
            cv2.imshow(window,canvas);screenshot=canvas
            if cv2.waitKey(1)&255 in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if max_seconds and now-startup>=max_seconds:break
    finally:
        if camera:camera.close()
        cv2.destroyAllWindows()
        if screenshot is not None:
            path=ROOT/'logs/exposure_tuner_last.png';path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path),screenshot)
        try:apply_profile(index,'quality')
        except Exception as exc:print('恢复自动曝光失败；下次正常启动会重试：',exc,file=sys.stderr)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--max-seconds',type=float,default=0.)
    args=parser.parse_args()
    run(args.camera,args.max_seconds)
