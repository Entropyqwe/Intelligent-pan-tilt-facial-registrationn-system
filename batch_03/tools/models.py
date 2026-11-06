"""Download or restore pinned OpenCV Zoo weights; reject LFS pointers/bad hashes."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

REPO=Path(__file__).resolve().parents[1]
COMMIT='48b10fa505aeb51fbd127fc60a5c660040e823b7'
MODELS={
 'face_detection_yunet_2023mar.onnx':('face_detection_yunet','8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
 'face_recognition_sface_2021dec.onnx':('face_recognition_sface','0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
 'object_detection_nanodet_2022nov.onnx':('object_detection_nanodet','4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186'),
}


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',type=Path,help='Folder containing original model files')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--faces-only',action='store_true')
    a=p.parse_args()
    target=REPO/'app/models';target.mkdir(exist_ok=True)
    for name,(folder,sha) in MODELS.items():
        if a.faces_only and name.startswith('object_'):continue
        dest=target/name
        if dest.exists() and digest(dest)==sha:
            print('OK',name);continue
        if a.verify:raise SystemExit('MISSING OR BAD SHA256: '+str(dest))
        temp=dest.with_suffix('.download')
        try:
            if a.offline:
                shutil.copyfile(a.offline/name,temp)
            else:
                url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/{folder}/{name}'
                print('Downloading',name,flush=True)
                with urllib.request.urlopen(url,timeout=90) as response,temp.open('wb') as out:
                    shutil.copyfileobj(response,out)
            if digest(temp)!=sha:raise RuntimeError('Model hash mismatch: '+name)
            temp.replace(dest)
            print('VERIFIED',name)
        finally:
            if temp.exists():temp.unlink()


if __name__=='__main__':main()
