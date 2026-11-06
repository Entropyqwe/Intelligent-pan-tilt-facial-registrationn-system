"""Verify original ONNX downloads and detect Git LFS pointer files."""
from pathlib import Path
import hashlib

MODELS = {
    "object_detection_nanodet_2022nov.onnx": "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}

def main():
    for name, expected in MODELS.items():
        path = Path(__file__).resolve().parent/"models"/name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Invalid model: {path}. Download original ONNX, not a web page.")
        print("OK", name, path.stat().st_size)

if __name__ == "__main__":
    main()
