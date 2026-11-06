"""Verify a clean source tree against SHA256SUMS.txt; ignores local extra files."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    failed=[];count=0
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        sha,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        if ROOT not in path.parents:raise SystemExit('Manifest path escapes root')
        count+=1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:failed.append(name)
    print('校验文件数：',count,'；不一致：',len(failed))
    for name in failed:print(name)
    raise SystemExit(bool(failed))
