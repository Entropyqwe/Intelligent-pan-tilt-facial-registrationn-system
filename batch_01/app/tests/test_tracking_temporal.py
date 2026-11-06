"""Regressions for partial first crops and independent association/control."""
import unittest
from unittest.mock import patch
import numpy as np
from body_identity import BodyIdentities,descriptor,appearance_distance
from test_body_identity import scene


class TemporalTests(unittest.TestCase):
    def test_weak_local_chain_can_change_view_without_changing_public_id(self):
        tr=BodyIdentities();box=(180,60,100,330)
        f,b=scene([box],[(0,0,230)])
        # Two nearby feature changes can each be plausible while the final
        # view is too far from the initial template. Keep the short-term
        # chain separate from the immutable identity anchor.
        def feature(angle):return np.array([np.cos(angle),np.sin(angle)],np.float32)
        with patch('body_identity.descriptor',return_value=feature(0)):
            for i in range(5):out=tr.update(f,b,10+i*.1)
        pid=out[0]['person_id'];anchor=tr.tracks[pid].anchor.copy()
        weak=[dict(box=box,score=.2)]
        with patch('body_identity.descriptor',return_value=feature(.62)):
            out=tr.update(f,weak,10.5)
        self.assertFalse(out[0]['confirmed'])
        np.testing.assert_allclose(tr.tracks[pid].anchor,anchor)
        with patch('body_identity.descriptor',return_value=feature(1.0)):
            out=tr.update(f,b,10.6)
            self.assertFalse(out[0]['confirmed'])
            out=tr.update(f,b,10.7)
        self.assertEqual(out[0]['person_id'],pid)
        self.assertTrue(out[0]['confirmed'])
        self.assertEqual(tr.counters['public_ids_created'],1)

    def test_overlap_requires_two_safe_observations_before_control_resumes(self):
        tr=BodyIdentities();f,b=scene([(180,60,100,330)],[(0,0,230)])
        for i in range(5):out=tr.update(f,b,10+i*.1)
        pid=out[0]['person_id']
        crowded=b+[dict(box=(220,60,100,330),score=.8)]
        out=tr.update(f,crowded,10.5)
        self.assertFalse(any(t['confirmed'] for t in out))
        out=tr.update(f,b,10.6)
        self.assertFalse(any(t['confirmed'] for t in out))
        out=tr.update(f,b,10.7)
        self.assertEqual(out[0]['person_id'],pid)
        self.assertTrue(out[0]['confirmed'])

    def test_bottom_fragment_does_not_control_or_establish_identity(self):
        for box in [(180,375,150,104),(541,363,97,113)]:
            with self.subTest(box=box):
                tr=BodyIdentities()
                f,b=scene([box],[(0,0,230)])
                for i in range(8):out=tr.update(f,b,10+i*.1)
                self.assertFalse(any(t['confirmed'] for t in out))
                self.assertEqual(tr.counters['public_ids_created'],0)

    def test_confirmed_template_uses_current_crop(self):
        tr=BodyIdentities()
        # A candidate's first crop sees shoulder/background; later true crops
        # have enough geometric overlap and slowly changing content.
        frames=[]
        for i,color in enumerate([(210,211,213),(215,214,216),(220,221,223)]):
            f,b=scene([(150,50,150,370)],[color]);frames.append(f)
            out=tr.update(f,b,10+i*.1)
        pid=out[0]['person_id'];self.assertGreater(pid,0)
        np.testing.assert_allclose(tr.tracks[pid].anchor,descriptor(frames[-1],b[0]['box']))

    def test_low_overlap_retains_id_but_never_controls(self):
        tr=BodyIdentities()
        f,b=scene([(180,60,100,330)],[(0,0,230)])
        for i in range(5):out=tr.update(f,b,10+i*.1)
        pid=out[0]['person_id'];updates=tr.counters['feature_updates']
        # Two weak overlapping proposals should suspend control without
        # automatically deleting a unique trajectory association.
        weak=[dict(box=(180,60,100,330),score=.25),dict(box=(220,60,100,330),score=.11)]
        out=tr.update(f,weak,10.5)
        self.assertFalse(any(t['confirmed'] for t in out))
        self.assertEqual(tr.counters['feature_updates'],updates)
        for i in range(3):out=tr.update(f,b,10.6+i*.1)
        self.assertEqual(out[0]['person_id'],pid)
        self.assertTrue(out[0]['confirmed'])
        self.assertEqual(tr.counters['public_ids_created'],1)

    def test_changed_clothing_after_loss_cannot_claim_old_id(self):
        tr=BodyIdentities();box=(180,60,100,330)
        f,b=scene([box],[(0,0,230)])
        for i in range(5):out=tr.update(f,b,10+i*.1)
        pid=out[0]['person_id']
        tr.update(f,[],10.5)
        other,b=scene([box],[(230,0,0)])
        for i in range(4):
            out=tr.update(other,b,11.5+i*.1)
            self.assertFalse(any(t['person_id']==pid and t['confirmed'] for t in out))

    def test_border_sliver_cannot_create_new_identity(self):
        tr=BodyIdentities()
        f,b=scene([(583,342,55,132)],[(0,0,230)])
        for i in range(20):tr.update(f,b,10+i*.1)
        self.assertEqual(tr.counters['public_ids_created'],0)
        self.assertFalse(tr.tracks)

    def test_established_identity_can_continue_at_border(self):
        tr=BodyIdentities();pid=0
        for i in range(35):
            f,b=scene([(430+i*4,100,80,300)],[(0,0,230)])
            out=tr.update(f,b,10+i*.1)
            if i==4:pid=out[0]['person_id']
        self.assertEqual(out[0]['person_id'],pid)
        self.assertEqual(tr.counters['public_ids_created'],1)


if __name__=='__main__':unittest.main()
