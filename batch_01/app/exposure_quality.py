"""Conservative exposure screening, not a guarantee of tracking accuracy."""
import cv2
import numpy as np

METRICS=('mean_luma','dark_fraction','clipped_fraction','magenta_fraction','chroma_noise')


def measure(frame):
    # Use full native pixels: downsampling hides the chroma speckle in the
    # faulty manual-exposure frames. Median residual is a noise proxy only.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    chroma=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    residual=np.abs(chroma.astype(np.float32)-cv2.medianBlur(chroma,3).astype(np.float32))
    b,g,r=cv2.split(frame.astype(np.float32))
    return dict(mean_luma=float(gray.mean()),dark_fraction=float(np.mean(gray<16)),
                clipped_fraction=float(np.mean(gray>245)),
                magenta_fraction=float(np.mean((b+r)*.5-g>30)),
                chroma_noise=float(np.percentile(residual[:,:,1:].max(axis=2),90)))


def average(samples):
    if not samples:raise ValueError('没有画质样本。')
    return {key:float(np.mean([sample[key] for sample in samples])) for key in METRICS}


def rejection(stats,baseline=None):
    if not 30<=stats['mean_luma']<=210:return '画面整体过暗或过亮'
    if stats['clipped_fraction']>.25:return '过曝区域超过25%'
    if stats['dark_fraction']>.80:return '暗部区域超过80%'
    if stats['magenta_fraction']>.25:return '大面积紫色色偏'
    if stats['chroma_noise']>12:return '彩色高频噪点过强'
    if baseline:
        if stats['clipped_fraction']>baseline['clipped_fraction']+.10:return '过曝明显劣于自动曝光'
        if stats['magenta_fraction']>baseline['magenta_fraction']+.12:return '紫色色偏明显劣于自动曝光'
        if stats['chroma_noise']>max(6,baseline['chroma_noise']+4):return '噪点明显劣于自动曝光'
    return None


def verify_capture(cap,baseline=None,warm_seconds=1.):
    import time
    warm=time.monotonic()+warm_seconds
    samples=[];start=None
    while True:
        ok,frame=cap.read()
        if not ok:raise RuntimeError('固定曝光画质检查未收到图像。')
        now=time.monotonic()
        if now<warm:continue
        if start is None:start=now
        samples.append(measure(frame))
        if len(samples)>=8 and now-start>=.5:break
        if now-warm>4:break
    stats=average(samples)
    return rejection(stats,baseline),stats
