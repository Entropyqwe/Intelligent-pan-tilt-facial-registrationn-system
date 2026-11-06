import threading
import unittest
from unittest.mock import patch
from main import Output


class MotorTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.o=Output.__new__(Output)
        self.o.mutex=threading.Lock()
        self.o._serial_rx=bytearray()

    def test_fragmented_m1_and_stale_feedback(self):
        with patch('main.time.monotonic',return_value=10.):
            self.o._consume_telemetry(b'#M1,42,1600,15')
            self.assertFalse(self.o.motor['valid'])
            self.o._consume_telemetry(b'80,120,120,300,TRACK_READY\n')
            self.assertTrue(self.o.motor['valid'])
            self.assertEqual(self.o.motor['applied_us'],1580)
            self.assertEqual(self.o.motor['left_pwm'],120)
        with patch('main.time.monotonic',return_value=10.6):
            self.assertFalse(self.o.motor['valid'])

    def test_bad_m1_cannot_refresh_feedback(self):
        with patch('main.time.monotonic',return_value=10.):
            self.o._parse_telemetry('#M1,42,1600,1580,120,120,300,TRACK_READY')
        with patch('main.time.monotonic',return_value=11.):
            for line in ['#M1,43,1600,999,120,120,300,TRACK_READY',
                         '#M1,43,1600,1580,-1,120,300,TRACK_READY',
                         '#M1,43,1600,1580,120,120,1000,TRACK_READY',
                         '#M1,43,1600,1580,120,120,300,WHATEVER',
                         '#M1,43,1600,1580,120,120,300,START_WAIT,extra']:
                self.o._parse_telemetry(line)
            self.assertFalse(self.o.motor['valid'])
            self.assertEqual(self.o.motor['seq'],42)

    def test_old_firmware_and_ack_still_supported(self):
        self.o._parse_telemetry('#A1,42,1,1600,-3000')
        self.assertEqual(self.o._last_ack['seq'],42)
        self.assertFalse(self.o.motor['valid'])
