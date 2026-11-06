"""Produce clean upload tree and <=90-file batches OUTSIDE the repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from audit_package import REPO,files,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO.parent/'browser_upload_ready')
    a=p.parse_args();out=a.output.resolve()
    if out==REPO or REPO in out.parents:raise SystemExit('Output must be outside repository')
    if out.exists():raise SystemExit('Output already exists; use another --output to avoid overwriting files')
    if audit()['errors']:raise SystemExit('Audit failed')
    selected=[f for f in files() if f.name!='SHA256SUMS.txt']
    manifest=''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.relative_to(REPO).as_posix()+'\n' for f in selected)
    (REPO/'SHA256SUMS.txt').write_text(manifest,encoding='utf-8',newline='\n')
    selected=files()
    clean=out/'repository';clean.mkdir(parents=True)
    for source in selected:
        dest=clean/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    batches=[]
    for i in range(0,len(selected),90):
        batch=out/'batches'/f'batch_{i//90+1:02d}';members=selected[i:i+90]
        for source in members:
            dest=batch/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        batches.append(dict(directory=str(batch),files=len(members),bytes=sum(s.stat().st_size for s in members)))
    archive=out/'FaceLock_Gimbal_Face_GitHub.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source in clean.rglob('*'):
            if source.is_file():z.write(source,source.relative_to(clean).as_posix())
    (out/'批次说明.json').write_text(json.dumps(batches,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'上传说明.txt').write_text('repository 是真正的仓库根目录，不含模型、姓名库、运行日志。\n'
        '网页分批：每次打开一个 batches/batch_XX，将该目录内部的文件和子目录拖入仓库根上传；不要把 batch_XX 目录本身拖入。\n'
        '每批不超过90个文件，所有文件保留相对目录。以GitHub网页显示的文件列表为准，.gitignore及.github等隐藏内容也要上传。\n'
        'ZIP仅供传输/备份，GitHub上传ZIP不会自动展开为源代码。模型留在离线目录，不上传。\n',encoding='utf-8')
    print('READY',out)
    print('FILES',len(selected),'ZIP_BYTES',archive.stat().st_size)


if __name__=='__main__':main()
