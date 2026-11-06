"""OpenCV YuNet + SFace on CPU. The two model versions are pinned."""
import cv2
import numpy as np
from core import Observation, unit, unique_identities


class FaceVision:
    def __init__(self, root, threshold=0.45, margin=0.05, max_faces=5,
                 threads=4, detector_size=320, configure_threads=True, reject_overflow=False):
        if not 1 <= threads <= 64 or detector_size not in (160, 224, 320):
            raise ValueError('threads must be 1..64; detector_size must be 160, 224 or 320')
        if configure_threads:
            cv2.setNumThreads(threads)
        self.detector_size = detector_size
        models = root / "models"
        if (tuple(int(x) for x in cv2.__version__.split('.')[:2]) < (4,8)
            or not hasattr(cv2, "FaceDetectorYN_create")
            or not hasattr(cv2, "FaceRecognizerSF_create")):
            raise RuntimeError("OpenCV >=4.8 is required; use the project environment")
        self.detector = cv2.FaceDetectorYN_create(
            str(models / "face_detection_yunet_2023mar.onnx"), "", (detector_size, detector_size),
            0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(models / "face_recognition_sface_2021dec.onnx"), "")
        self.threshold, self.margin = threshold, margin
        self.max_faces = max_faces
        self.reject_overflow = reject_overflow

    def process(self, frame, gallery):
        # Reduced detector input; scale all five landmarks back to full frame.
        height, width = frame.shape[:2]
        size = self.detector_size
        small = cv2.resize(frame, (size, size))
        self.detector.setInputSize((size, size))
        _, rows = self.detector.detect(small)
        faces = []
        if rows is None:
            return faces
        if self.reject_overflow and len(rows)>self.max_faces:
            return faces
        rows = sorted(rows, key=lambda r: r[2] * r[3], reverse=True)
        for row in rows[:self.max_faces]:
            face = row.copy()
            face[[0, 2, 4, 6, 8, 10, 12]] *= width / size
            face[[1, 3, 5, 7, 9, 11, 13]] *= height / size
            x, y, w, h = map(float, face[:4])
            # Tiny faces are detected but not used for identification/control.
            if w < 45 or h < 45:
                continue
            aligned = self.recognizer.alignCrop(frame, face)
            feature = unit(self.recognizer.feature(aligned))
            pid, similarity = gallery.match(feature, self.threshold, self.margin)
            x1, y1 = max(0., x), max(0., y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(dict(person_id=pid, similarity=similarity,
                              score=float(face[-1]), box=(x1,y1,x2-x1,y2-y1),
                              feature=feature, aligned=aligned))
        return unique_identities(faces)


def observations(faces):
    return [Observation(f["person_id"], *f["box"], f["score"], f["similarity"])
            for f in faces if f["person_id"] > 0]
