"""Host-native C regression; temp output only, never flashes or rewrites HEX."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

FIRMWARE=Path(__file__).resolve().parents[1]/'firmware'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc',default='gcc')
    a=p.parse_args()
    compiler=shutil.which(a.cc)
    if not compiler:raise SystemExit('C compiler not found: '+a.cc)
    env=os.environ.copy()
    env['PATH']=str(Path(compiler).parent)+os.pathsep+env.get('PATH','')
    with tempfile.TemporaryDirectory(prefix='facelock-native-') as tmp:
        local=Path(tmp)
        for folder in ('User','tests'):
            for file in (FIRMWARE/folder).iterdir():
                if file.suffix in ('.h','.c'):shutil.copy2(file,local/file.name)
        for name in ('gimbal_native_test','control_upgrade_test','hold3s_native_test','oled_status_test'):
            sources=[name+'.c','oled_status.c'] if name=='oled_status_test' else [name+'.c',
                'gimbal_control.c','steering_protocol.c','vehicle_control.c']
            exe=local/(name+('.exe' if os.name=='nt' else ''))
            subprocess.run([compiler,'-std=c99','-Wall','-Wextra','-Werror','-I.',*sources,'-o',str(exe)],
                           cwd=local,env=env,check=True)
            subprocess.run([str(exe)],cwd=local,env=env,check=True)
    print('4 native C suites passed; firmware untouched.')


if __name__=='__main__':main()
