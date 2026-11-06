"""Audit browser-upload content, firmware integrity and relative Keil paths."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[1]
FIRMWARE_SHA='6caf86c0711a6bbe533088cfacfa31bbe1e9537db4fcbe8a68f344b723c649a8'


def included(path):
    rel=path.relative_to(REPO)
    if any(p in ('.git','.venv','__pycache__','.pytest_cache') for p in rel.parts):return False
    if path.suffix.lower() in ('.onnx','.nb','.so','.pyc','.download','.wav','.mp3','.mp4','.avi','.o','.axf','.map','.bin'):return False
    if str(rel).replace('\\','/').startswith(('app/data/','app/logs/')) and path.name!='.gitkeep':return False
    if str(rel).replace('\\','/').startswith('firmware/Objects/') and path.name!='FaceLock_GIMBAL.hex':return False
    if path.name.endswith('.uvoptx') or '.uvguix.' in path.name or 'LOCAL_ONLY' in path.name:return False
    return True


def files():
    return sorted((p for p in REPO.rglob('*') if p.is_file() and included(p)),key=lambda p:p.relative_to(REPO).as_posix())


def audit():
    errors=[];selected=files()
    for path in selected:
        if path.stat().st_size>=25*1024**2:errors.append('Browser file too large: '+str(path))
    for folder in ('app/data','app/logs'):
        private=[p.name for p in (REPO/folder).iterdir() if p.name!='.gitkeep']
        # Local runtime files are excluded from upload; report their presence clearly.
        if private:print('LOCAL EXCLUDED',folder,len(private))
    firmware=REPO/'firmware'
    actual=hashlib.sha256((firmware/'Objects/FaceLock_GIMBAL.hex').read_bytes()).hexdigest()
    if actual!=FIRMWARE_SHA:errors.append('Stable HEX SHA256 changed')
    tree=ET.parse(firmware/'Project_VEHICLE.uvprojx')
    refs=[]
    for node in tree.iter('FilePath'):
        if node.text:
            value=node.text.replace('\\','/')
            path=firmware/value
            if not path.is_file():errors.append('Missing Keil source: '+value)
            refs.append(value)
    result=dict(files=len(selected),total_bytes=sum(p.stat().st_size for p in selected),
                max_file_bytes=max(p.stat().st_size for p in selected),
                firmware_sha256=actual,keil_source_references=len(refs),errors=errors)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':
    raise SystemExit(1 if audit()['errors'] else 0)
