"""Chinese Tk desktop menu for the copied face/gimbal project."""
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox,simpledialog
from launch import LABELS,ROOT,REPO


def main():
    root=tk.Tk();root.title('人体＋人脸欢迎＋两轴云台 · 独立工程')
    root.geometry('800x820')
    tk.Label(root,text='人体识别 · 姓名标记 · 中文语音欢迎',font=('',18,'bold')).pack(pady=12)
    tk.Label(root,text=str(REPO),wraplength=760).pack()
    tk.Label(root,text='默认自动曝光 / 彩色640×480 / 旋转180°\n首次验证电机VM断开；按Q退出视觉，下一次启动重新读取姓名库。').pack(pady=8)
    status=tk.StringVar(value='先登记姓名，再运行人脸预览；确认后使用完整模式。')
    def start(mode):
        command=[sys.executable,str(ROOT/'launch.py'),'--mode',mode]
        if mode=='enroll':
            name=simpledialog.askstring('登记姓名','请输入要播报的姓名（不要输入文件路径）：',parent=root)
            if not name:return
            command+=['--name',name]
        if mode in ('pan','tilt','front','full'):
            if not messagebox.askokcancel('台架联调','程序可能转动舵机，请保持电机VM断开、云台无机械干涉。继续启动？'):return
        try:
            # Keep terminal errors visible after exit. All arguments stay argv.
            runner=[sys.executable,str(ROOT/'terminal_action.py'),*command]
            subprocess.Popen(['x-terminal-emulator','-e',*runner],cwd=ROOT)
            status.set('已打开：'+LABELS[mode])
        except OSError as exc:messagebox.showerror('启动失败',str(exc))
    for mode,label in LABELS.items():
        tk.Button(root,text=label,command=lambda m=mode:start(m),height=1).pack(fill='x',padx=28,pady=2)
    def stop():
        from desktop_control import stop_session
        status.set(stop_session())
    tk.Button(root,text='停止当前视觉／验证程序',command=stop,bg='#ffd8cf').pack(fill='x',padx=28,pady=6)
    tk.Label(root,textvariable=status,wraplength=760,fg='#24517a').pack(pady=5)
    root.mainloop()


if __name__=='__main__':main()
