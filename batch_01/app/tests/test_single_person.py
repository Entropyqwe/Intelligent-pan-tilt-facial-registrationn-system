import unittest
from unittest.mock import patch
import numpy as np
from single_person import SinglePersonPolicy
from body_identity import BodySelection, BodyIdentities


def body(x=180,score=.9):
    return dict(box=(x,60,100,330),score=score)


class SinglePersonTests(unittest.TestCase):
    def test_auto_lock_even_when_previous_id_was_lost(self):
        policy=SinglePersonPolicy();selection=BodySelection();selection.select(55)
        for i in range(6):
            tracks=policy.apply([], [body()],10+i*.1,640,480)
            policy.auto_select(selection)
            snap=selection.update(tracks,10+i*.1,640,480)
        self.assertEqual(selection.target_id,1);self.assertTrue(snap.valid)

    def test_repeated_leave_reappear_always_uses_slot_one(self):
        policy=SinglePersonPolicy();ids=set()
        for cycle in range(20):
            t=10+cycle*5
            policy.apply([],[],t,640,480)
            for i in range(6):
                tracks=policy.apply([], [body()],t+1+i*.1,640,480)
                ids.update(x['person_id'] for x in tracks if x['confirmed'])
        self.assertEqual(ids,{1})

    def test_two_people_suspend_slot_and_never_alias_one(self):
        p=SinglePersonPolicy()
        for i in range(6):p.apply([], [body()],10+i*.1,640,480)
        tracks=p.apply([dict(person_id=1,confirmed=True)], [body(50),body(400)],10.6,640,480)
        self.assertFalse(p.ready);self.assertEqual(p.count,2)
        self.assertEqual(tracks[0]['person_id'],2)
        tracks=p.apply([], [body()],10.7,640,480)
        self.assertFalse(any(t['confirmed'] for t in tracks))

    def test_duplicates_not_extra_person_but_distinct_weak_person_vetoes(self):
        p=SinglePersonPolicy()
        for i in range(6):tracks=p.apply([], [body(),body(182,.22)],10+i*.1,640,480)
        self.assertTrue(p.ready);self.assertEqual(tracks[0]['person_id'],1)
        p.apply([], [body(),body(420,.23)],10.6,640,480)
        self.assertFalse(p.ready);self.assertEqual(p.count,2)

    def test_low_confidence_and_predicted_tracks_cannot_drive(self):
        p=SinglePersonPolicy()
        for i in range(6):p.apply([], [body()],10+i*.1,640,480)
        tracks=p.apply([], [body(score=.24)],10.6,640,480)
        self.assertFalse(p.ready)
        self.assertEqual([t['person_id'] for t in tracks],[1])
        tracks=p.apply([], [],10.7,640,480)
        self.assertEqual(tracks,[])

    def test_unconfirmed_single_never_exposes_internal_replacement_number(self):
        p=SinglePersonPolicy()
        tracks=p.apply([dict(person_id=38,confirmed=False)], [body()],10,640,480)
        self.assertEqual(tracks[0]['person_id'],1)
        self.assertFalse(tracks[0]['confirmed'])

    def test_jump_requires_reconfirmation_and_fragment_cannot_lock(self):
        p=SinglePersonPolicy()
        for i in range(6):p.apply([], [body(40)],10+i*.1,640,480)
        tracks=p.apply([], [body(480)],10.6,640,480)
        self.assertFalse(any(t['confirmed'] for t in tracks))
        for i in range(8):
            tracks=p.apply([], [dict(box=(180,375,150,104),score=.9)],11+i*.1,640,480)
            self.assertFalse(any(t['confirmed'] for t in tracks))

    def test_manual_selection_not_forced_without_auto_call(self):
        p=SinglePersonPolicy();selection=BodySelection();selection.select(0)
        for i in range(6):p.apply([], [body()],10+i*.1,640,480)
        self.assertEqual(selection.target_id,0)

    def test_repeated_timestamp_cannot_confirm(self):
        p=SinglePersonPolicy()
        for _ in range(8):p.apply([], [body()],10,640,480)
        self.assertFalse(p.ready)

    def test_recent_reentry_confirms_two_current_frames_at_new_position(self):
        p=SinglePersonPolicy()
        for i in range(6):p.apply([], [body(40)],10+i*.1,640,480)
        p.apply([], [],10.6,640,480)
        tracks=p.apply([], [body(450)],11.,640,480)
        self.assertFalse(tracks[0]['confirmed'])
        tracks=p.apply([], [body(440)],11.1,640,480)
        self.assertTrue(tracks[0]['confirmed'])
        self.assertEqual(tracks[0]['box'],body(440)['box'])

    def test_competition_weak_observation_and_long_gap_revoke_fast_reentry(self):
        for event in ('multiple','weak','long'):
            p=SinglePersonPolicy()
            for i in range(6):p.apply([], [body()],10+i*.1,640,480)
            proposals=[body(40),body(450)] if event=='multiple' else [body(score=.24)] if event=='weak' else []
            p.apply([],proposals,10.6,640,480)
            t=14. if event=='long' else 10.8
            for i in range(2):tracks=p.apply([], [body()],t+i*.1,640,480)
            self.assertFalse(tracks[0]['confirmed'],event)

    def test_single_fast_path_does_not_extract_unused_appearance(self):
        p=SinglePersonPolicy();identities=BodyIdentities()
        frame=np.full((480,640,3),120,np.uint8)
        with patch('body_identity.descriptor',side_effect=AssertionError('unused appearance')):
            for i in range(8):p.update(frame,[body()],10+i*.1,identities)
        self.assertTrue(p.ready)
        self.assertEqual(p.metrics()['fast_path_frames'],8)
        self.assertEqual(identities.counters['candidates_created'],0)

    def test_multi_paths_age_expire_and_do_not_reuse_single_role_as_identity(self):
        p=SinglePersonPolicy();ids=BodyIdentities(retain_seconds=1.)
        frame=np.full((480,640,3),120,np.uint8)
        for i in range(6):tracks=p.update(frame,[body(50),body(420)],10+i*.1,ids)
        self.assertTrue(any(t['confirmed'] for t in tracks))
        self.assertTrue(all(t['person_id']!=1 for t in tracks))
        with patch('body_identity.descriptor',side_effect=AssertionError('unused appearance')):
            for i in range(20):p.update(frame,[body()],10.6+i*.1,ids)
        self.assertEqual(ids.tracks,{})
        self.assertGreater(ids.counters['expired'],0)
        tracks=p.update(frame,[body(50),body(420)],13.,ids)
        self.assertFalse(any(t['confirmed'] for t in tracks))
