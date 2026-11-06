"""Receive-only telemetry: framing, invalid data and stale distance display."""
import threading
import unittest
from unittest.mock import patch
from main import Output


class DistanceTelemetryTests(unittest.TestCase):
    def setUp(self):
        # Avoid opening a port or starting the heartbeat for parser tests.
        self.output = Output.__new__(Output)
        self.output.mutex = threading.Lock()
        self.output._distance = dict(mm=0, valid=False, mode='NO_DATA', age_ms=999999)
        self.output._distance_received = None
        self.output._serial_rx = bytearray()

    def test_fragmented_lines_and_legacy_ack(self):
        with patch('main.time.monotonic', return_value=10.):
            self.output._consume_telemetry(b'#A1,1,1,1500,0\n#D1,4')
            self.assertFalse(self.output.distance['valid'])
            self.output._consume_telemetry(b'20,1,FOLLOW\r\n')
            self.assertEqual(self.output.distance,
                             dict(mm=420, valid=True, mode='FOLLOW', age_ms=0))

    def test_stale_at_500ms_and_new_reading_recovers(self):
        with patch('main.time.monotonic', return_value=10.):
            self.output._parse_telemetry('#D1,420,1,FOLLOW')
        with patch('main.time.monotonic', return_value=10.499):
            self.assertTrue(self.output.distance['valid'])
        with patch('main.time.monotonic', return_value=10.5):
            self.assertEqual(self.output.distance['mode'], 'STALE')
            self.assertFalse(self.output.distance['valid'])
        with patch('main.time.monotonic', return_value=11.):
            self.output._parse_telemetry('#D1,300,1,STOP')
            self.assertTrue(self.output.distance['valid'])

    def test_corrupt_lines_cannot_refresh_last_good_reading(self):
        with patch('main.time.monotonic', return_value=10.):
            self.output._parse_telemetry('#D1,420,1,FOLLOW')
        with patch('main.time.monotonic', return_value=11.):
            for data in (b'#D1,40001,1,FOLLOW\n', b'#D1,400,2,FOLLOW\n',
                         b'#D1,400,1,F\xffOLLOW\n', b'#D1,400,1,A,B\n',
                         b'#D1,400,1,\n', b'#D1,400,1,A B\n'):
                self.output._consume_telemetry(data)
            self.assertEqual(self.output.distance['mm'], 420)
            self.assertFalse(self.output.distance['valid'])

    def test_missing_newline_memory_is_bounded_and_parser_recovers(self):
        self.output._consume_telemetry(b'x'*8192)
        self.assertLessEqual(len(self.output._serial_rx), 256)
        self.output._consume_telemetry(b'\n#D1,300,1,STOP\n')
        self.assertTrue(self.output.distance['valid'])

    def test_zero_or_explicit_invalid_distance_cannot_display_as_valid(self):
        for line in ('#D1,0,1,STOP', '#D1,400,0,NO_ECHO'):
            self.output._parse_telemetry(line)
            self.assertFalse(self.output.distance['valid'])


if __name__ == '__main__':
    unittest.main()
