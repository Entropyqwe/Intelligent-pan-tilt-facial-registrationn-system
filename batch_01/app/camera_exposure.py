"""V4L2 exposure readback, testing metadata and saved fixed camera settings."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import os

ROOT=Path(__file__).resolve().parent
SAVED_PROFILE=ROOT/'config/camera_fixed_exposure.json'
EDITABLE=('exposure_time_absolute','brightness','gain','gamma','contrast','saturation')
PERSISTED=('auto_exposure','exposure_dynamic_framerate',*EDITABLE,
           'backlight_compensation','power_line_frequency',
           'white_balance_automatic','white_balance_temperature','sharpness')


def ctl(index,*args):
    result=subprocess.run(['v4l2-ctl','-d',f'/dev/video{index}',*args],
                          capture_output=True,text=True,check=True,timeout=3)
    return result.stdout


def control_info(index):
    info={}
    for line in ctl(index,'--list-ctrls').splitlines():
        match=re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)',line)
        if not match:continue
        name,kind,details=match.groups()
        numbers={k:int(v) for k,v in re.findall(r'(min|max|step|default|value)=(-?\d+)',details)}
        if kind=='bool':numbers.update(min=0,max=1,step=1)
        if 'value' in numbers:info[name]=dict(numbers,kind=kind,inactive='inactive' in details)
    if 'auto_exposure' not in info or 'exposure_time_absolute' not in info:
        raise RuntimeError('相机没有提供所需V4L2曝光控制，不能使用此调试功能。')
    return info


def read_controls(index,names):
    names=tuple(names)
    result={}
    for line in ctl(index,'--get-ctrl='+','.join(names)).splitlines():
        match=re.match(r'\s*(\w+)\s*:\s*(-?\d+)',line)
        if match and match[1] in names:result[match[1]]=int(match[2])
    if set(result)!=set(names):raise RuntimeError('相机参数回读不完整，未保存。')
    return result


def set_control(index,name,value):
    if name not in PERSISTED:raise ValueError('Unsupported camera control')
    ctl(index,f'--set-ctrl={name}={int(value)}')


def capture_controls(index):
    available=control_info(index)
    return read_controls(index,[name for name in PERSISTED if name in available])


def apply_controls(index,controls):
    # Set mode before its dependent exposure / white-balance value.
    ordered=('auto_exposure','exposure_dynamic_framerate','white_balance_automatic')
    for name in (*ordered,*(key for key in controls if key not in ordered)):
        if name not in controls:continue
        if name=='white_balance_temperature' and controls.get('white_balance_automatic')==1:continue
        if name=='exposure_time_absolute' and controls.get('auto_exposure')!=1:continue
        set_control(index,name,controls[name])


def freeze_auto(index):
    before=capture_controls(index)
    # Read automatic exposure first; switching modes may reset the register.
    # This device can report a stale exposure value. Rewriting that value
    # visibly corrupts the image; do not pretend readback is a measured value.
    set_control(index,'auto_exposure',1)
    if 'exposure_dynamic_framerate' in before:set_control(index,'exposure_dynamic_framerate',0)
    return capture_controls(index)


def camera_identity(index):
    from camera_profiles import usb_identity
    identity=usb_identity(index)
    if not identity:raise RuntimeError('无法识别USB相机型号，不能载入固定曝光。')
    return identity


def same_controls(first,second):
    # Temperature readback may change while automatic white balance is on.
    def stable(values):
        return {k:v for k,v in values.items() if not
                (k=='white_balance_temperature' and values.get('white_balance_automatic')==1)}
    return stable(first)==stable(second)


def validate_controls(controls,info):
    if not isinstance(controls,dict) or controls.get('auto_exposure')!=1:
        raise ValueError('保存的参数必须为手动曝光。')
    if 'exposure_time_absolute' not in controls:raise ValueError('缺少固定曝光值。')
    if 'exposure_dynamic_framerate' in info and controls.get('exposure_dynamic_framerate')!=0:
        raise ValueError('固定档必须关闭动态帧率。')
    for name,value in controls.items():
        if name not in PERSISTED or name not in info or type(value) is not int:
            raise ValueError('无效或不支持的相机控制项：'+str(name))
        bounds=info[name]
        if not bounds['min']<=value<=bounds['max']:
            raise ValueError('相机参数超出范围：'+name)
        step=max(1,bounds.get('step',1))
        if (value-bounds['min'])%step:raise ValueError('相机参数步长不匹配：'+name)


def validate_test(test):
    if not isinstance(test,dict):raise ValueError('缺少曝光测试记录。')
    for name in ('seconds','fps','mean_luma','dark_fraction','clipped_fraction','magenta_fraction','chroma_noise'):
        value=test.get(name)
        if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('测试记录数值无效。')
    if test['seconds']<3 or test['fps']<=0 or type(test.get('frames')) is not int or test['frames']<3:
        raise ValueError('请先完成至少3秒的实际图像测试。')
    if not 0<=test['mean_luma']<=255 or any(not 0<=test[k]<=1 for k in ('dark_fraction','clipped_fraction')):
        raise ValueError('图像统计值无效。')
    if test['dark_fraction']>.98 or test['clipped_fraction']>.98:
        raise ValueError('画面几乎全黑或全白，不能保存为有效测试配置。')
    from exposure_quality import rejection
    baseline=test.get('auto_baseline')
    if not isinstance(baseline,dict):raise ValueError('缺少自动曝光画质基准，请重新测试。')
    for key in ('mean_luma','dark_fraction','clipped_fraction','magenta_fraction','chroma_noise'):
        value=baseline.get(key)
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('自动曝光基准无效。')
    for values in (test,baseline):
        if not 0<=values['mean_luma']<=255 or values['chroma_noise']<0 or any(not 0<=values[key]<=1 for key in ('dark_fraction','clipped_fraction','magenta_fraction')):
            raise ValueError('画质统计值超出范围。')
    reason=rejection(test,baseline)
    if reason:raise ValueError('固定曝光画质不合格：'+reason+'。已阻止保存/载入。')
    if test.get('replay_verified') is not True:raise ValueError('固定曝光参数尚未通过重载测试。')


def save_profile(index,controls,test,path=SAVED_PROFILE,*,quality_confirmed=False):
    current=capture_controls(index)
    if not same_controls(controls,current):raise ValueError('参数已变化，请重新测试后保存。')
    validate_controls(controls,control_info(index));validate_test(test)
    if quality_confirmed is not True:raise ValueError('请先对照自动与固定画面，确认细节、色彩、噪点可用。')
    record=dict(schema=2,usb_identity=camera_identity(index),resolution=[640,480],fourcc='MJPG',
                controls=controls,test=test,saved_at=datetime.now(timezone.utc).isoformat(),
                tracking_validated=False,quality_confirmed=True)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
        name=f.name
        try:json.dump(record,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
    try:os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)
    return record


def load_profile(index,path=SAVED_PROFILE):
    try:record=json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:raise RuntimeError('尚未保存固定曝光，请先打开“曝光调试”完成测试并保存。') from None
    if record.get('schema')!=2 or record.get('quality_confirmed') is not True:
        raise ValueError('旧固定档未经过画质与重载检查，已禁用；请用自动曝光，重新调试。')
    if record.get('resolution')!=[640,480] or record.get('fourcc')!='MJPG':
        raise ValueError('固定曝光文件格式不兼容。')
    if record.get('usb_identity')!=camera_identity(index):raise ValueError('相机型号已变化，请重新调试曝光。')
    validate_controls(record.get('controls'),control_info(index));validate_test(record.get('test'))
    return record


def apply_saved(index,path=SAVED_PROFILE):
    record=load_profile(index,path)
    apply_controls(index,record['controls'])
    actual=capture_controls(index)
    for key,value in record['controls'].items():
        if key=='white_balance_temperature' and actual.get('white_balance_automatic')==1:continue
        if actual.get(key)!=value:raise RuntimeError('相机未接受保存参数：'+key)
    return record
