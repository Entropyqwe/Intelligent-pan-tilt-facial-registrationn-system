import tempfile
import time
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import (Gallery,Observation,Snapshot,TargetLock,control_fields,
                  encode_packet,decode_packet,unique_identities)


class FaceControlTests(unittest.TestCase):
    def test_id_persists_and_unknown_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = Gallery(Path(tmp)/"gallery.json")
            self.assertEqual(g.add([[1,0,0]]),1)
            self.assertEqual(g.add([[0,1,0]]),2)
            reopened = Gallery(g.path)
            self.assertEqual(reopened.match([1,.01,0])[0],1)
            self.assertEqual(reopened.match([0,1,0])[0],2)
            self.assertEqual(reopened.match([0,0,1])[0],0)
            self.assertEqual(reopened.match([1,1,0])[0],0)

    def test_lock_never_switches_to_another_identity(self):
        lock=TargetLock(confirm_frames=3)
        lock.select(1)
        a=Observation(1,10,10,100,100,.95,.7)
        b=Observation(2,10,10,100,100,.95,.7)
        self.assertFalse(lock.update([a],10.,640,480).valid)
        self.assertFalse(lock.update([a],10.1,640,480).valid)
        self.assertTrue(lock.update([a],10.2,640,480).valid)
        self.assertFalse(lock.update([b],10.3,640,480).valid)
        self.assertEqual(lock.target_id,1)
        self.assertFalse(lock.update([],10.4,640,480).valid)
        self.assertFalse(lock.update([a],10.5,640,480).valid)
        self.assertFalse(lock.update([a],10.6,640,480).valid)
        self.assertTrue(lock.update([a],10.7,640,480).valid)

    def test_occlusion_timeout_and_duplicates(self):
        a=Observation(1,0,0,100,100,.95,.7)
        lock=TargetLock(confirm_frames=1)
        lock.select(1)
        self.assertTrue(lock.update([a],1.,640,480).valid)
        self.assertFalse(lock.update([a,a],1.1,640,480).valid)
        faces=unique_identities([dict(person_id=1),dict(person_id=1),dict(person_id=2)])
        self.assertEqual([f['person_id'] for f in faces],[0,0,2])

    def test_coordinate_sign_and_stale_data(self):
        a=Observation(1,0,100,128,200,.95,.7)
        s=Snapshot(1,True,10.,a,640,480)
        d=control_fields(s,now=10.1)
        self.assertEqual((d['valid'],d['cx'],d['ex'],d['direction']),(1,100,-800,-1))
        self.assertEqual(d['cy'],417)
        stale=control_fields(s,now=10.5)
        self.assertEqual((stale['valid'],stale['ex'],stale['direction']),(0,0,0))
        self.assertEqual(stale['target_id'],1)

    def test_wire_checksum_and_partial_packet_reassembly(self):
        fields=control_fields(Snapshot(target_id=4),now=10)
        wire=encode_packet(9,fields)
        buf=bytearray()
        parsed=[]
        for chunk in [wire[:4],wire[4:10],wire[10:]]:
            buf.extend(chunk)
            if b'\n' in buf:
                parsed.append(decode_packet(bytes(buf)))
        self.assertEqual(parsed[0]['target_id'],4)
        self.assertEqual(parsed[0]['seq'],9)
        with self.assertRaises(ValueError):
            decode_packet(wire.replace(b'F1,9,',b'F1,8,'))
        # Even a correct checksum must not admit an impossible steering field.
        invalid = dict(fields, valid=1, target_id=4, ex=1001)
        with self.assertRaises(ValueError):
            decode_packet(encode_packet(10, invalid))
        with self.assertRaises(ValueError):
            decode_packet(encode_packet(11, dict(fields, cx=500)))


if __name__=='__main__':
    unittest.main()
