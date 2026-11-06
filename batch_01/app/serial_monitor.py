"""Test parser. Use a receiver PC/board, not the occupied sender port."""
import argparse
import serial
from core import decode_packet

ap = argparse.ArgumentParser()
ap.add_argument("port",help="COM4 or /dev/ttyUSB1")
args = ap.parse_args()
with serial.serial_for_url(args.port,115200,timeout=.1) as port:
    buf = bytearray()
    while True:
        chunk = port.read(256)
        buf.extend(chunk)
        if len(buf)>4096:
            buf.clear()
        while b"\n" in buf:
            line,_,tail = buf.partition(b"\n")
            buf = bytearray(tail)
            try:
                print(decode_packet(line+b"\n"))
            except (ValueError,UnicodeError):
                print("Bad packet",repr(line))
