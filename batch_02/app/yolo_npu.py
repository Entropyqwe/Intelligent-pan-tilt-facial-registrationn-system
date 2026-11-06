"""Person-only decoding of the system YOLOv5 NBG on A733 VIPLite.

Keeps the existing PersonVision.process -> [{box, score}] contract. This is
object detection, not person identity recognition. The body entry loads no faces.
"""
import time
from pathlib import Path
import cv2
import numpy as np
from vip_runtime import VipNetwork

ANCHORS = np.array([[[10,13],[16,30],[33,23]],
                    [[30,61],[62,45],[59,119]],
                    [[116,90],[156,198],[373,326]]], np.float32)


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -80., 80.)))


def decode_heads(outputs, size, width, height, resized_w, resized_h, left, top,
                 threshold=.45, nms=.5):
    boxes, scores = [], []
    seen = set()
    gate = np.log(threshold/(1.-threshold))
    for raw in outputs:
        if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[1] != 3 or raw.shape[-1] != 85:
            raise ValueError(f'Unsupported YOLOv5 output layout {raw.shape}; expected [1,3,H,W,85]')
        _, _, gh, gw, _ = raw.shape
        stride = size//gw
        if gh != gw or stride not in (8,16,32) or gw*stride != size or stride in seen:
            raise ValueError('YOLO head grid does not match the pinned stride contract')
        seen.add(stride)
        head = raw[0]
        candidate = np.nonzero((head[...,4] >= gate) & (head[...,5] >= gate))
        if not len(candidate[0]):
            continue
        rows = head[candidate]
        # Preserve multiclass semantics: a candidate must rank person highest.
        confidence = sigmoid(rows[:,4]) * sigmoid(rows[:,5])
        keep = (confidence >= threshold) & (rows[:,5:].argmax(axis=1) == 0)
        if not np.any(keep):
            continue
        rows = rows[keep]
        a, gy, gx = (indices[keep] for indices in candidate)
        confidence = confidence[keep]
        xywh = sigmoid(rows[:,:4])
        xy = (xywh[:,:2]*2.-.5+np.column_stack((gx,gy)))*stride
        wh = (xywh[:,2:]*2.)**2 * ANCHORS[(8,16,32).index(stride),a]
        xy1, xy2 = xy-wh/2., xy+wh/2.
        xy1[:,0] = np.clip((xy1[:,0]-left)*width/resized_w,0,width)
        xy2[:,0] = np.clip((xy2[:,0]-left)*width/resized_w,0,width)
        xy1[:,1] = np.clip((xy1[:,1]-top)*height/resized_h,0,height)
        xy2[:,1] = np.clip((xy2[:,1]-top)*height/resized_h,0,height)
        wh = xy2-xy1
        useful = (wh[:,0]>=20) & (wh[:,1]>=40)
        boxes.extend(np.column_stack((xy1,wh))[useful].tolist())
        scores.extend(confidence[useful].tolist())
    if seen != {8,16,32}:
        raise ValueError('Expected exactly the three YOLOv5 head strides')
    if not boxes:
        return []
    indices = np.asarray(cv2.dnn.NMSBoxes(boxes,scores,threshold,nms)).reshape(-1)
    return [dict(box=tuple(boxes[int(i)]),score=float(scores[int(i)])) for i in indices]


class YoloNpuVision:
    backend_name = 'VIPLite NPU / YOLOv5'

    def __init__(self, model='/opt/yolov5/model/yolov5.nb', threshold=.45, nms=.5):
        self.network = VipNetwork(model)
        self.threshold, self.nms = threshold, nms
        try:
            info = self.network.input_info
            chw=(info.shape == (1,3,640,640) and info.format == 2
                    and info.quantization == 2 and info.zero_point == 0
                    and abs(info.scale-1/255.) <= 1e-6)
            rgb=(info.shape == (1,640,640,3) and info.format == 2 and info.quantization == 0)
            if not (chw or rgb):
                raise ValueError('Adapter requires verified YOLOv5 CHW quantized or RGB-image uint8 input')
            if (chw and (info.shape != (1,3,640,640) or info.format != 2
                    or info.quantization != 2 or info.zero_point != 0
                    or abs(info.scale-1/255.) > 1e-6)):
                raise ValueError('This adapter requires the system YOLOv5 640 RGB CHW uint8 model (scale 1/255)')
            shapes = {i.shape for i in self.network.output_info}
            expected=({(1,3,80,80,85),(1,3,40,40,85),(1,3,20,20,85)} if chw else
                      {(1,3,85,80,80),(1,3,85,40,40),(1,3,85,20,20)})
            if shapes != expected:
                raise ValueError('NBG output contract differs from system YOLOv5; use a matching decoder')
            self.size = 640
            self.rgb_input=rgb
            self.tensor = np.zeros(info.shape,np.uint8)
            self.last_timing = {}
        except BaseException:
            self.close()
            raise

    def process(self, frame):
        start = time.monotonic()
        height,width = frame.shape[:2]
        scale = min(self.size/width,self.size/height)
        # Matches the official system example: floor resize and black padding.
        rw,rh = max(1,int(width*scale)),max(1,int(height*scale))
        left,top = (self.size-rw)//2,(self.size-rh)//2
        resized = frame if (rw,rh)==(width,height) else cv2.resize(frame,(rw,rh))
        self.tensor.fill(114 if self.rgb_input else 0)
        # BGR -> planar RGB directly, without allocating an intermediate RGB image.
        if self.rgb_input:self.tensor[0,top:top+rh,left:left+rw,:]=resized[:,:,::-1]
        else:self.tensor[0,:,top:top+rh,left:left+rw] = resized[:,:,::-1].transpose(2,0,1)
        pre = time.monotonic()
        candidates = self.network.run_people(self.tensor,self.threshold)
        infer = time.monotonic()
        bodies = decode_candidates(candidates,width,height,rw,rh,left,top,self.threshold,self.nms)
        done = time.monotonic()
        self.last_timing = dict(pre_ms=1000*(pre-start), npu_and_copy_ms=1000*(infer-pre),
                                post_ms=1000*(done-infer),total_ms=1000*(done-start))
        return bodies

    def close(self):
        self.network.close()


def decode_candidates(rows,width,height,rw,rh,left,top,threshold=.45,nms=.5):
    if not len(rows):
        return []
    stride=rows[:,0]
    group=np.searchsorted(np.array([8,16,32]),stride)
    anchor=rows[:,1].astype(int)
    xywh=sigmoid(rows[:,4:8])
    xy=(xywh[:,:2]*2.-.5+rows[:,[3,2]])*stride[:,None]
    wh=(xywh[:,2:]*2.)**2*ANCHORS[group,anchor]
    xy1,xy2=xy-wh/2.,xy+wh/2.
    xy1[:,0]=np.clip((xy1[:,0]-left)*width/rw,0,width)
    xy2[:,0]=np.clip((xy2[:,0]-left)*width/rw,0,width)
    xy1[:,1]=np.clip((xy1[:,1]-top)*height/rh,0,height)
    xy2[:,1]=np.clip((xy2[:,1]-top)*height/rh,0,height)
    wh=xy2-xy1
    keep=(wh[:,0]>=20)&(wh[:,1]>=40)
    boxes=np.column_stack((xy1,wh))[keep].tolist()
    scores=rows[:,8][keep].tolist()
    if not boxes:return []
    indices=np.asarray(cv2.dnn.NMSBoxes(boxes,scores,threshold,nms)).reshape(-1)
    return [dict(box=tuple(boxes[int(i)]),score=float(scores[int(i)])) for i in indices]
