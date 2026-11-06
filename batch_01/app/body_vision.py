"""NanoDet Plus CPU person detection, adapted from OpenCV Zoo (Apache 2.0).

Reference commit: 48b10fa505aeb51fbd127fc60a5c660040e823b7.
Keep models/object_detection_nanodet_LICENSE.txt with distributions.
"""
from pathlib import Path
import cv2
import numpy as np


class PersonVision:
    def __init__(self, root, threshold=0.45, nms=0.5):
        path = Path(root) / "models" / "object_detection_nanodet_2022nov.onnx"
        if not path.is_file():
            raise RuntimeError(f"Missing person model: {path}; run verify_models.py")
        # Read bytes through pathlib so Windows Chinese drive paths work too.
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(),dtype=np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.threshold, self.nms = threshold, nms
        self.size = 416
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.output_pairs = None
        self.canvas = np.zeros((self.size, self.size, 3), np.uint8)
        self.bins = np.arange(8, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], np.float32)
        self.std = np.array([57.375, 57.12, 58.395], np.float32)
        self.strides = (8, 16, 32)  # Fixed 2022nov model exposes three feature levels.
        self.anchors = []
        for stride in self.strides:
            grid = np.arange(self.size // stride, dtype=np.float32) * stride + (stride - 1) * .5
            xx, yy = np.meshgrid(grid, grid)
            self.anchors.append(np.column_stack((xx.ravel(), yy.ravel())))

    def process(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized_w = max(1, round(width * scale))
        resized_h = max(1, round(height * scale))
        left, top = (self.size - resized_w) // 2, (self.size - resized_h) // 2
        canvas = self.canvas
        canvas.fill(0)
        resized = cv2.resize(frame, (resized_w, resized_h))
        canvas[top:top+resized_h, left:left+resized_w] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage((canvas.astype(np.float32)-self.mean)/self.std)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        if len(outputs) != 6:
            raise RuntimeError("Unexpected NanoDet output; use the pinned 2022nov model")
        if self.output_pairs is None:
            # Layer order differs across OpenCV versions. Pair by tensor shape.
            self.output_pairs = []
            for anchors in self.anchors:
                pair = []
                for channels in (80, 32):
                    matches = [i for i, a in enumerate(outputs)
                               if a.shape[-1] == channels and a.size == len(anchors)*channels]
                    if len(matches) != 1:
                        raise RuntimeError('Unexpected NanoDet tensor shapes; use the pinned 2022nov model')
                    pair.append(matches[0])
                self.output_pairs.append(pair)
        boxes, scores = [], []
        for stride, anchors, (ci, ri) in zip(self.strides, self.anchors, self.output_pairs):
            classes, regression = outputs[ci], outputs[ri]
            classes = classes.reshape(-1, 80)
            regression = regression.reshape(-1, 4, 8)
            # COCO class 0 = person. Other class logits still run in the model.
            candidates = np.flatnonzero(classes[:, 0] >= self.threshold)
            if not len(candidates):
                continue
            candidates = candidates[classes[candidates].argmax(axis=1) == 0]
            if not len(candidates):
                continue
            logits = regression[candidates]
            exponent = np.exp(logits - logits.max(axis=2, keepdims=True))
            distances = (exponent / exponent.sum(axis=2, keepdims=True)
                         @ self.bins) * stride
            points = anchors[candidates]
            xy1 = points - distances[:, :2]
            xy2 = points + distances[:, 2:]
            xy1[:, 0] = np.clip((xy1[:, 0]-left)*width/resized_w, 0, width)
            xy2[:, 0] = np.clip((xy2[:, 0]-left)*width/resized_w, 0, width)
            xy1[:, 1] = np.clip((xy1[:, 1]-top)*height/resized_h, 0, height)
            xy2[:, 1] = np.clip((xy2[:, 1]-top)*height/resized_h, 0, height)
            wh = xy2 - xy1
            usable = (wh[:, 0] >= 20) & (wh[:, 1] >= 40)
            boxes.extend(np.column_stack((xy1, wh))[usable].tolist())
            scores.extend(classes[candidates, 0][usable].tolist())
        if not boxes:
            return []
        indices = np.asarray(cv2.dnn.NMSBoxes(
            boxes, scores, self.threshold, self.nms)).reshape(-1)
        return [dict(box=tuple(boxes[int(i)]), score=float(scores[int(i)]))
                for i in indices]
