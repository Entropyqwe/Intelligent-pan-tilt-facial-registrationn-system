import itertools
import unittest
import numpy as np
from body_identity import BodyIdentities,BodySelection
from tracking_motion import assign,BoxMotion
from core import control_fields
from test_body_identity import scene


class AssignmentTests(unittest.TestCase):
    def test_global_assignment_beats_greedy(self):
        result=assign([[.1,.2],[.11,np.inf]],.8)
        self.assertEqual(set(result),{(0,1),(1,0)})

    def test_forbidden_edges_and_unmatched_costs(self):
        self.assertEqual(assign([[1.,np.inf]],.8),[])
        self.assertEqual(assign(np.zeros((0,3)),.8),[])
        self.assertEqual(assign([[.1],[.2]],.8),[(0,0)])

    def test_small_random_costs_against_exhaustive_search(self):
        rng=np.random.default_rng(30)
        for _ in range(60):
            cost=rng.uniform(size=(3,3));limit=.6
            result=assign(cost,limit)
            candidates=[]
            for choices in itertools.product(range(-1,3),repeat=3):
                columns=[j for j in choices if j>=0]
                if len(columns)!=len(set(columns)):continue
                pairs=[(i,j) for i,j in enumerate(choices) if j>=0]
                if any(cost[i,j]>limit for i,j in pairs):continue
                candidates.append((-len(pairs),sum(cost[i,j] for i,j in pairs)))
            wanted=min(candidates)
            self.assertEqual(-len(result),wanted[0])
            self.assertAlmostEqual(sum(cost[i,j] for i,j in result),wanted[1])


class TrackingTests(unittest.TestCase):
    def run_person(self,tracker,box,t,score=.9,color=(0,0,230)):
        frame,bodies=scene([box],[color]);bodies[0]['score']=score
        return tracker.update(frame,bodies,t)

    def confirmed(self,tracker,box=(180,60,100,330)):
        tracks=[]
        for i in range(5):tracks=self.run_person(tracker,box,10+i*.1)
        return tracks[0]['person_id']

    def test_false_candidate_does_not_consume_public_id(self):
        tr=BodyIdentities()
        for i in range(40):
            self.run_person(tr,(100,60,90,300),10+i*2)
            tr.update(np.zeros((480,640,3),np.uint8),[],10+i*2+.8)
        self.assertEqual(tr.next_id,1)
        self.assertEqual(tr.metrics()['public_ids_created'],0)

    def test_partial_head_sliver_cannot_establish_body_identity(self):
        tr=BodyIdentities()
        for i in range(10):self.run_person(tr,(230,370,240,100),10+i*.1)
        self.assertEqual(tr.metrics()['public_ids_created'],0)
        self.assertFalse(tr.tracks)

    def test_low_score_keeps_id_but_cannot_steer_or_update_template(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        updates=tr.counters['feature_updates']
        for i in range(8):
            tracks=self.run_person(tr,(180+i*2,60,100,330),10.5+i*.1,score=.2)
            self.assertEqual(tracks[0]['person_id'],pid)
            self.assertFalse(tracks[0]['confirmed'])
        self.assertEqual(tr.counters['feature_updates'],updates)
        tracks=self.run_person(tr,(196,60,100,330),11.3)
        self.assertFalse(tracks[0]['confirmed'])
        tracks=self.run_person(tr,(198,60,100,330),11.4)
        self.assertTrue(tracks[0]['confirmed'])

    def test_gap_recovery_keeps_id_and_needs_real_confirmations(self):
        for gap in (.2,.5,1.):
            tr=BodyIdentities();selection=BodySelection();pid=self.confirmed(tr);selection.select(pid)
            blank=np.zeros((480,640,3),np.uint8)
            self.assertEqual(tr.update(blank,[],10.5),[])
            self.assertFalse(selection.update([],10.5,640,480).valid)
            tracks=self.run_person(tr,(185,60,100,330),10.5+gap)
            self.assertEqual(tracks[0]['person_id'],pid)
            self.assertFalse(selection.update(tracks,10.5+gap,640,480).valid)
            tracks=self.run_person(tr,(190,60,100,330),10.6+gap)
            self.assertTrue(selection.update(tracks,10.6+gap,640,480).valid)
            self.assertEqual(tr.metrics()['public_ids_created'],1)

    def test_motion_uses_seconds_and_retains_velocity(self):
        motion=BoxMotion((100,60,80,300))
        for i in range(1,20):
            motion.predict(.1);motion.update((100+i*5,60,80,300))
        x=motion.box[0];motion.predict(.4)
        self.assertAlmostEqual(motion.box[0]-x,20,delta=6.)
        self.assertTrue(np.all(np.linalg.eigvalsh(motion.cov)>0))

    def test_blind_motion_drift_cannot_block_last_observed_identity(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        motion=tr.tracks[pid].motion
        motion.mean[4:]=[800.,0.,-400.,0.]
        blank=np.zeros((480,640,3),np.uint8)
        for stamp in (10.6,10.9,11.2,11.5):tr.update(blank,[],stamp)
        tracks=self.run_person(tr,(185,60,100,330),11.6)
        self.assertEqual(tracks[0]['person_id'],pid)
        self.assertFalse(tracks[0]['confirmed'])
        tracks=self.run_person(tr,(186,60,100,330),11.7)
        self.assertEqual(tracks[0]['person_id'],pid)
        self.assertTrue(tracks[0]['confirmed'])

    def test_crossing_and_overlap_freeze_control_and_appearance(self):
        tr=BodyIdentities();selection=BodySelection()
        for i in range(5):
            f,b=scene([(80,70,100,300),(420,70,100,300)],[(0,0,230),(230,0,0)])
            tracks=tr.update(f,b,10+i*.1)
        selection.select(tracks[0]['person_id']);updates=tr.counters['feature_updates']
        f,b=scene([(200,70,100,300),(220,70,100,300)],[(0,0,230),(230,0,0)])
        tracks=tr.update(f,b,10.5)
        self.assertFalse(selection.update(tracks,10.5,640,480).valid)
        self.assertEqual(tr.counters['feature_updates'],updates)

    def test_nested_weak_proposal_does_not_veto_high_person(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        f,b=scene([(180,60,100,330)],[(0,0,230)])
        b.append(dict(box=(183,65,98,320),score=.15))
        tracks=tr.update(f,b,10.5)
        self.assertEqual(len(tracks),1)
        self.assertEqual(tracks[0]['person_id'],pid);self.assertTrue(tracks[0]['confirmed'])

    def test_giant_weak_proposal_cannot_veto_unique_high_observation(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        f,b=scene([(180,60,100,330)],[(0,0,230)])
        b.append(dict(box=(2,4,636,472),score=.12))
        tracks=tr.update(f,b,10.5)
        self.assertEqual(tracks[0]['person_id'],pid);self.assertTrue(tracks[0]['confirmed'])

    def test_cropped_detection_does_not_update_appearance(self):
        tr=BodyIdentities();self.confirmed(tr,box=(180,150,100,330))
        before=tr.counters['feature_updates']
        for i in range(4):self.run_person(tr,(180,150,100,330),10.5+i*.1)
        self.assertEqual(tr.counters['feature_updates'],before)

    def test_long_loss_expires_id_and_never_switches_selection(self):
        tr=BodyIdentities();selection=BodySelection();pid=self.confirmed(tr);selection.select(pid)
        for i in range(5):tracks=self.run_person(tr,(180,60,100,330),25+i*.1)
        self.assertNotEqual(tracks[0]['person_id'],pid)
        snap=selection.update(tracks,25.4,640,480)
        self.assertEqual(snap.target_id,pid);self.assertFalse(snap.valid)
        self.assertEqual(control_fields(snap,now=25.4)['valid'],0)

    def test_decreasing_timestamps_invalidate_current_observation(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        self.assertEqual(self.run_person(tr,(180,60,100,330),10.3),[])
        tracks=self.run_person(tr,(180,60,100,330),10.5)
        self.assertEqual(tracks[0]['person_id'],pid);self.assertFalse(tracks[0]['confirmed'])

    def test_repeated_misses_cannot_create_dozens_of_ids(self):
        tr=BodyIdentities();pid=self.confirmed(tr)
        for i in range(120):
            t=10.5+i*.1
            if i%7==0:
                tr.update(np.zeros((480,640,3),np.uint8),[],t)
            else:
                tracks=self.run_person(tr,(180+int(35*np.sin(i*.08)),60,100,330),t)
                self.assertEqual(tracks[0]['person_id'],pid)
        self.assertEqual(tr.metrics()['public_ids_created'],1)
        self.assertLessEqual(len(tr.tracks),tr.max_tracks)


if __name__=='__main__':unittest.main()
