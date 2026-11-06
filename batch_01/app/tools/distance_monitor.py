"""Read-only STM32 telemetry monitor for the HC-SR04 link.

Use this before starting the vision program.  It never sends S1 packets and
therefore cannot move the servo or enable the motor.  Expected lines are:
  #READY,S1,HOLD3S,PA1,115200
  #D1,<millimetres>,<valid>,<mode>
"""
import argparse
import time
import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="/dev/ttyUSB0 or COMx")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    if not 0.5 <= args.seconds <= 300:
        ap.error("--seconds must be between 0.5 and 300")

    counts = {"READY": 0, "A1": 0, "D1": 0, "D1_VALID": 0, "D1_INVALID": 0}
    buf = bytearray()
    deadline = time.monotonic() + args.seconds
    with serial.Serial(args.port, 115200, timeout=0.1) as port:
        print(f"监听 {args.port}，115200 8N1；本程序只读，不发送 S1。", flush=True)
        while time.monotonic() < deadline:
            buf.extend(port.read(256))
            if len(buf) > 4096:
                del buf[:-256]
            while b"\n" in buf:
                raw, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith("#READY,"):
                    counts["READY"] += 1
                elif line.startswith("#A1,"):
                    counts["A1"] += 1
                elif line.startswith("#D1,"):
                    fields = line.split(",")
                    if len(fields) == 4:
                        counts["D1"] += 1
                        try:
                            valid = int(fields[2]) == 1 and int(fields[1]) > 0
                        except ValueError:
                            valid = False
                        counts["D1_VALID" if valid else "D1_INVALID"] += 1
                print(line, flush=True)
    print("统计：" + ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    if counts["D1"] == 0:
        print("结论：未收到 #D1；先检查 STM32 是否烧录车辆版 HEX、PA9/PA10、公共 GND 和串口号。")
    elif counts["D1_VALID"] == 0:
        print("结论：已收到 #D1，但 ECHO 无有效回波；检查 HC-SR04 供电、PB6/PB7 和分压。")
    else:
        print("结论：距离遥测链路有效，可以启动视觉程序；电机仍需在安全架空条件下验证。")


if __name__ == "__main__":
    main()
