import unittest
from body_tracker import BodyTargetLock
from core import Snapshot, control_fields, encode_packet, decode_packet


def face(pid=1, box=(130, 30, 70, 70)):
    return dict(person_id=pid, box=box, similarity=.8, score=.98)


def body(box=(100, 10, 150, 450)):
    return dict(box=box, score=.9)


class BodyContinuityTests(unittest.TestCase):
    def setUp(self):
        self.lock = BodyTargetLock()
        self.lock.select(1)

    def acquire(self):
        for stamp in (10., 10.1, 10.2):
            snap = self.lock.update([face()], [body()], stamp, 640, 480)
        self.assertTrue(snap.valid)
        self.assertEqual(snap.source, 1)
        return snap

    def test_back_view_requires_face_binding(self):
        self.assertFalse(self.lock.update([], [body()], 9., 640, 480).valid)
        self.acquire()
        snap = self.lock.update([], [body((110, 12, 150, 450))], 10.3, 640, 480)
        self.assertTrue(snap.valid)
        self.assertEqual((self.lock.mode, snap.source, snap.observation.similarity),
                         ("BODY", 2, -1.))
        self.assertEqual(snap.face_confirmed, 10.2)

    def test_missed_body_requires_face_reacquisition(self):
        self.acquire()
        self.assertFalse(self.lock.update([], [], 10.3, 640, 480).valid)
        self.assertFalse(self.lock.update([], [body()], 10.4, 640, 480).valid)
        for stamp in (10.5, 10.6):
            self.assertFalse(self.lock.update([face()], [body()], stamp, 640, 480).valid)
        self.assertTrue(self.lock.update([face()], [body()], 10.7, 640, 480).valid)

    def test_crossing_bodies_drop_binding(self):
        self.acquire()
        bodies = [body(), body((150, 12, 150, 440))]
        self.assertFalse(self.lock.update([], bodies, 10.3, 640, 480).valid)
        self.assertFalse(self.lock.update([], [body()], 10.4, 640, 480).valid)

    def test_other_person_and_unknown_face_stop(self):
        for pid in (0, 2):
            self.lock.select(1)
            self.acquire()
            self.assertFalse(self.lock.update([face(pid)], [body()], 10.3, 640, 480).valid)

    def test_time_gap_large_motion_and_duplicate_identity_stop(self):
        for faces, bodies, stamp in [
            ([], [body()], 10.9),
            ([], [body((400, 10, 150, 450))], 10.3),
            ([face(), face(box=(420, 30, 70, 70))], [body()], 10.3),
            ([], [body()], 10.2),
        ]:
            self.lock.select(1)
            self.acquire()
            self.assertFalse(self.lock.update(faces, bodies, stamp, 640, 480).valid)

    def test_unrelated_distant_person_does_not_break_continuity(self):
        self.acquire()
        self.assertTrue(self.lock.update([], [body(), body((450, 10, 140, 440))],
                                         10.3, 640, 480).valid)

    def test_conflicting_target_elsewhere_cannot_continue_old_body(self):
        self.acquire()
        self.assertFalse(self.lock.update([face(box=(420, 30, 70, 70))],
                                         [body()], 10.3, 640, 480).valid)

    def test_f2_distinguishes_body_and_clears_stale_fields(self):
        self.acquire()
        snap = self.lock.update([], [body()], 10.3, 640, 480)
        values = control_fields(snap, now=10.4)
        packet = encode_packet(1, values, "F2")
        parsed = decode_packet(packet)
        self.assertTrue(packet.startswith(b"@F2,"))
        self.assertEqual((parsed["source"], parsed["sim"], parsed["face_age_ms"]), (2, -1000, 200))
        stale = decode_packet(encode_packet(2, control_fields(snap, now=10.9), "F2"))
        self.assertEqual((stale["valid"], stale["source"], stale["cx"]), (0, 0, 0))
        with self.assertRaises(ValueError):
            decode_packet(encode_packet(3, dict(values, source=0), "F2"))
        invalid = control_fields(Snapshot(target_id=1), now=11.)
        self.assertEqual(decode_packet(encode_packet(4, invalid, "F2"))["source"], 0)


if __name__ == "__main__":
    unittest.main()
