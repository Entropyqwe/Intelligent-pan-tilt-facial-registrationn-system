"""One-person follow slot, separate from multi-person identity association.

Slot 1 means the sole currently observed person, never biometric identity.
Only consecutive real high-score detections establish/recover this slot.
Multiple independent proposals immediately suspend it. No predicted box drives.
"""
import math
from body_tracker import iou, intersection


class SinglePersonPolicy:
    def __init__(self, high=.35, confirm_seconds=.35, confirm_frames=3):
        self.high=high
        self.confirm_seconds=confirm_seconds
        self.confirm_frames=confirm_frames
        self.last_time=None
        self.last_box=None
        self.started=None
        self.streak=0
        self.ready=False
        self.count=0
        self.single_frames=0
        self.multiple_frames=0
        self.last_confirmed=None
        self.recovering=False
        self.fast_frames=0
        self.full_frames=0

    def reset(self, keep_recovery=False):
        self.last_box=None
        self.started=None
        self.streak=0
        self.ready=False
        self.recovering=False
        if not keep_recovery:self.last_confirmed=None

    @staticmethod
    def multi_ids(tracks):
        return [dict(t,person_id=t['person_id']+1 if t['person_id']>0 else t['person_id'])
                for t in tracks]

    def update(self, frame, bodies, captured, identities):
        """Only extract appearance and assign identities in multi-person frames.

        Empty updates still age old multi-person trajectories. They cannot
        observe the sole person's crop or silently attach it to an old ID.
        """
        height,width=frame.shape[:2]
        tracks=self.apply([],bodies,captured,width,height)
        if self.count>1:
            self.full_frames+=1
            return self.multi_ids(identities.update(frame,bodies,captured))
        self.fast_frames+=1
        if identities.tracks:
            identities.update(frame,[],captured)
        return tracks

    def apply(self, tracks, bodies, captured, width, height):
        # Reserve 1 for the sole-person slot. Multi-person IDs never alias it.
        result=self.multi_ids(tracks)
        if self.last_time is not None and captured<=self.last_time:
            self.reset();self.count=0
            return result
        gap=0. if self.last_time is None else captured-self.last_time
        self.last_time=captured
        proposals=[]
        for body in bodies:
            x,y,w,h=body['box'];score=float(body['score'])
            if not all(math.isfinite(v) for v in (x,y,w,h,score)):
                continue
            if score<max(.2,self.high*.6) or score>1:
                continue
            x1,y1=max(0.,x),max(0.,y)
            x2,y2=min(float(width),x+w),min(float(height),y+h)
            if x2-x1<30 or y2-y1<90:
                continue
            proposals.append(dict(box=(x1,y1,x2-x1,y2-y1),score=score))
        distinct=[]
        for body in sorted(proposals,key=lambda b:b['score'],reverse=True):
            a=body['box']
            # Suppress only near-identical nested proposals, not crossing people.
            duplicate=False
            for other in distinct:
                b=other['box']
                smaller=min(a[2]*a[3],b[2]*b[3])
                center=math.hypot(a[0]+a[2]/2-b[0]-b[2]/2,
                                  a[1]+a[3]/2-b[1]-b[3]/2)
                if (iou(a,b)>.65 and intersection(a,b)/max(1.,smaller)>.85
                        and center<.12*min(a[3],b[3])):
                    duplicate=True;break
            if not duplicate:distinct.append(body)
        self.count=len(distinct)
        if self.count!=1:
            if self.count>1:self.multiple_frames+=1
            self.reset(keep_recovery=self.count==0)
            return result if self.count>1 else []
        body=distinct[0];box=body['box'];x,y,w,h=box
        # Do not turn a head fragment or border sliver into valid steering.
        safe=(body['score']>=self.high and h/w>=.6
              and not (y+h>=height-8 and h<height*.4)
              and not ((x<6 or x+w>width-6) and w/h<.6))
        continuous=False
        if safe and self.last_box is not None and gap<=.45:
            a=self.last_box
            center=math.hypot(x+w/2-a[0]-a[2]/2,y+h/2-a[1]-a[3]/2)
            continuous=(center/max(h,a[3],40)<.4
                        and .5<=w/a[2]<=2 and .5<=h/a[3]<=2)
        if not safe:
            self.reset()
            return [dict(person_id=1,box=box,score=body['score'],similarity=0.,
                         confirmed=False,established=False,observed=True,
                         state='TENTATIVE',label='SINGLE SLOT / UNSAFE OBSERVATION')]
        if not continuous:
            # A recent established sole-person slot needs two new real high
            # observations after absence. Multiple/unsafe observations revoke
            # that privilege; abrupt jumps without absence use full confirmation.
            self.recovering=(self.last_confirmed is not None
                             and 0<captured-self.last_confirmed<3.
                             and (self.last_box is None or gap>.45))
            if not self.recovering:self.last_confirmed=None
            self.started=captured;self.streak=0
        self.streak+=1;self.last_box=box
        frames=2 if self.recovering else self.confirm_frames
        seconds=.08 if self.recovering else self.confirm_seconds
        self.ready=(self.streak>=frames and captured-self.started>=seconds)
        if self.ready:self.last_confirmed=captured
        self.single_frames+=1
        # Use exactly this real observation; no Kalman prediction or old crop.
        return [dict(person_id=1,box=box,
                     score=body['score'],similarity=1.,confirmed=self.ready,
                     established=self.ready,observed=True,state='TRACKED' if self.ready else 'TENTATIVE',
                     label='SINGLE PERSON SLOT / NOT IDENTITY')]

    def auto_select(self, selector):
        if self.ready and selector.target_id!=1:
            selector.select(1)

    def metrics(self):
        return dict(visible_person_proposals=self.count,slot_ready=self.ready,
                    single_frames=self.single_frames,multiple_frames=self.multiple_frames,
                    fast_path_frames=self.fast_frames,multi_association_frames=self.full_frames,
                    recovering=self.recovering,
                    meaning='slot 1 is the sole person, not persistent identity')
