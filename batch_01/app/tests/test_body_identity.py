import unittest
import numpy as np
from core import control_fields
from body_identity import BodyIdentities,BodySelection,descriptor
from body_hud import BodyHUD


def scene(boxes,colors):
    image=np.zeros((480,640,3),np.uint8)
    bodies=[]
    for box,color in zip(boxes,colors):
        x,y,w,h=box;image[y:y+h,x:x+w]=color
        bodies.append(dict(box=tuple(box),score=.9))
    return image,bodies


class BodyAppearanceTests(unittest.TestCase):
    def test_color_identity_survives_detection_order_and_short_gap(self):
        tracker=BodyIdentities()
        boxes=[(60,80,100,300),(420,80,100,300)]
        frame,bodies=scene(boxes,[(0,0,240),(240,0,0)])
        for i in range(3):tracks=tracker.update(frame,bodies,10+i*.04)
        red,blue=[t['person_id'] for t in tracks]
        self.assertTrue(all(t['confirmed'] for t in tracks))
        tracks=tracker.update(frame,list(reversed(bodies)),10.12)
        self.assertEqual([t['person_id'] for t in tracks],[blue,red])
        tracker.update(frame,[],10.16)
        tracks=tracker.update(frame,bodies,10.2)
        self.assertEqual([t['person_id'] for t in tracks],[red,blue])
        self.assertFalse(any(t['confirmed'] for t in tracks))

    def test_ambiguous_similar_reappearance_does_not_claim_old_id(self):
        tracker=BodyIdentities()
        frame,bodies=scene([(20,70,90,300),(510,70,90,300)],[(0,0,230),(0,0,230)])
        for i in range(3):tracker.update(frame,bodies,10+i*.04)
        tracker.update(frame,[],10.2)
        frame,bodies=scene([(260,70,90,300)],[(0,0,230)])
        # A new tentative candidate may be displayed, but neither old identity
        # can be asserted and no control-ready observation may be emitted.
        recovered=tracker.update(frame,bodies,11.)
        self.assertFalse(any(t['confirmed'] or t['person_id']>0 for t in recovered))

    def test_overlap_miss_and_nonmonotonic_input_invalidate(self):
        tracker=BodyIdentities();selection=BodySelection()
        frame,bodies=scene([(200,70,130,330)],[(0,0,230)])
        for i in range(3):tracks=tracker.update(frame,bodies,10+i*.04)
        selection.select(tracks[0]['person_id'])
        snap=selection.update(tracks,10.08,640,480)
        self.assertTrue(snap.valid);self.assertEqual(snap.source,2)
        self.assertEqual(snap.face_confirmed,0.)
        self.assertEqual(control_fields(snap,now=10.1)['valid'],1)
        occluded=tracker.update(frame,bodies+[dict(box=(230,80,130,330),score=.9)],10.12)
        self.assertFalse(selection.update(occluded,10.12,640,480).valid)
        self.assertEqual(tracker.update(frame,bodies,10.1),[])

    def test_selected_id_never_switches_on_loss(self):
        selection=BodySelection();selection.select(1)
        snap=selection.update([dict(person_id=2,confirmed=True)],10.,640,480)
        self.assertEqual(snap.target_id,1);self.assertFalse(snap.valid)

    def test_similar_people_cannot_teleport_within_same_camera_interval(self):
        tracker=BodyIdentities()
        f,b=scene([(20,70,90,300)],[(0,0,230)])
        for i in range(3):old=tracker.update(f,b,10+i*.04)[0]['person_id']
        f,b=scene([(510,70,90,300)],[(0,0,230)])
        new=tracker.update(f,b,10.12)
        self.assertNotEqual(new[0]['person_id'],old)
        self.assertFalse(new[0]['confirmed'])

    def test_achromatic_shirt_noise_does_not_change_color_identity(self):
        from body_identity import appearance_distance
        f,b=scene([(120,40,160,400)],[(220,221,223)])
        q=f.copy();q[40:440,120:280]=(223,220,222)
        self.assertLess(appearance_distance(descriptor(f,b[0]['box']),descriptor(q,b[0]['box'])),.1)

    def test_head_excluded_and_resolution_mapping(self):
        frame,bodies=scene([(100,40,160,400)],[(10,20,230)])
        before=descriptor(frame,bodies[0]['box'])
        frame[40:110,100:260]=(230,0,0)
        np.testing.assert_allclose(before,descriptor(frame,bodies[0]['box']))
        hud=BodyHUD();stats=dict(fps=21,camera_fps=30,body_ms=30,appearance_ms=2,age_ms=40)
        canvas=hud.render(frame,[],BodySelection(),stats,None,True,'',0)
        self.assertEqual(canvas.shape,(660,960,3))
        self.assertEqual(hud.to_image(16,86),(0,0))
        self.assertIsNone(hud.to_image(656,86))


if __name__=='__main__':unittest.main()
