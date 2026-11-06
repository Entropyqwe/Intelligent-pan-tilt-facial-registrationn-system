"""Keep errors readable for a desktop double-click launch."""
import subprocess
import sys

if __name__=='__main__':
    try:code=subprocess.call(sys.argv[1:])
    except Exception as exc:print(exc);code=1
    if code:
        print('启动未成功，请保留上述错误并查看 app/logs。')
        try:input('按回车关闭此终端……')
        except EOFError:pass
    raise SystemExit(code)
