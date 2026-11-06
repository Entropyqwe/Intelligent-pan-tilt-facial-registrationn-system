"""Bounded ByteTrack-style association plus conservative appearance assistance.

Session IDs are not biometric identities. Predicted/lost/low-score tracks may
retain an ID, but only an unambiguous real high-score observation drives S1.
"""
from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np
from core import Observation, Snapshot
from body_tracker import iou, intersection
from tracking_motion import BoxMotion, assign


def descriptor(frame, box):
    x,y,w,h=box
    height,width=frame.shape[:2]
    # Exclude the upper 18% (head) and edge background from appearance features.
    x1,x2=max(0,int(x+.15*w)),min(width,int(x+.85*w))
    y1,y2=max(0,int(y+.18*h)),min(height,int(y+.95*h))
    if x2-x1<12 or y2-y1<24:return None
    patch=cv2.resize(frame[y1:y2,x1:x2],(32,80),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    # Hue is undefined near white/gray; sensor noise must not change a white shirt's ID.
    hsv[:,:,0][hsv[:,:,1]<40]=0
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    features=[]
    for start,end in ((0,28),(28,53),(53,80)):
        band=hsv[start:end]
        hs=cv2.calcHist([band],[0,1],None,[12,4],[0,180,0,256]).ravel()
        val=cv2.calcHist([band],[2],None,[8],[0,256]).ravel()
        hs/=max(float(hs.sum()),1);val/=max(float(val.sum()),1)
        # Hellinger features tolerate modest brightness changes, not new clothes.
        features.extend((np.sqrt(hs)*np.sqrt(.8/3),np.sqrt(val)*np.sqrt(.1/3)))
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitude,angle=cv2.cartToPolar(dx,dy)
    hist=np.bincount((angle.ravel()*8/(2*np.pi)).astype(int)%8,
                     weights=magnitude.ravel(),minlength=8).astype(np.float32)
    hist/=max(float(hist.sum()),1)
    features.append(np.sqrt(hist)*np.sqrt(.1))
    vector=np.concatenate(features).astype(np.float32)
    return vector/max(float(np.linalg.norm(vector)),1e-9)


def appearance_distance(a,b):
    # Scalar clamp avoids several NumPy dispatches for every tiny feature pair.
    return float(max(0.,1.-min(1.,max(0.,float(a@b))))**.5)


@dataclass
class Track:
    person_id:int
    box:tuple
    feature:np.ndarray
    score:float
    last_seen:float
    streak:int=1
    similarity:float=1.
    confirmed:bool=False
    motion:object=None
    anchor:object=None
    gallery:object=None
    hits:int=1
    state:str='TENTATIVE'
    visible:bool=False
    high_streak:int=1
    control_ready:bool=False
    feature_updates:int=0
    embedding:object=None
    embedding_anchor:object=None
    embedding_gallery:object=None
    last_embedding:float=0.
    recent_feature:object=None
    recent_time:float=0.
    high_feature:object=None
    high_time:float=0.


class BodyIdentities:
    def __init__(self,confirm_frames=3,max_gap=.7,retain_seconds=8.,
                 max_distance=.65,reid_distance=.28,ambiguity_margin=.07,
                 high_threshold=.35,low_threshold=.1,recover_frames=2,
                 max_tracks=32,max_detections=16,reid=None,reid_interval=.25):
        self.confirm_frames=confirm_frames
        self.max_gap=max_gap
        self.retain_seconds=retain_seconds
        self.max_distance=max_distance
        self.reid_distance=reid_distance
        self.margin=ambiguity_margin
        self.high_threshold=high_threshold;self.low_threshold=low_threshold
        self.recover_frames=recover_frames;self.max_tracks=max_tracks
        self.max_detections=max_detections
        self.reid=reid;self.reid_interval=reid_interval
        self.tracks={}
        self.next_id=1
        self.next_candidate=-1
        self.last_time=0.
        self.counters=dict(public_ids_created=0,candidates_created=0,recovered=0,
                           low_matches=0,ambiguous_frames=0,expired=0,feature_updates=0)

    @staticmethod
    def isolated(index,bodies,high_threshold=.35):
        a=bodies[index]['box']
        for j,other in enumerate(bodies):
            if j==index:continue
            # ByteTrack's low-score pass must not veto a unique high-score
            # observation with a giant weak proposal. High/high competition
            # remains unsafe; low-score matches never produce valid control.
            if bodies[index]['score']>=high_threshold and other['score']<high_threshold:continue
            b=other['box']
            if iou(a,b)>.15 or intersection(a,b)/max(1.,min(a[2]*a[3],b[2]*b[3]))>.25:
                return False
        return True

    @staticmethod
    def _distance(track,feature):
        if feature is None:return 1.
        # A bounded gallery tolerates front/side views without unlimited drift.
        return min(appearance_distance(track.feature,feature),
                   *(appearance_distance(sample,feature) for sample in track.gallery))

    @staticmethod
    def _near(a,b,limit):
        center=((a[0]+a[2]/2-b[0]-b[2]/2)**2+
                (a[1]+a[3]/2-b[1]-b[3]/2)**2)**.5
        return center/max(a[3],b[3],40.)<limit

    def _cost(self,track,body,feature,captured,stage):
        gap=captured-track.last_seen
        # A long blind extrapolation can shrink width to zero or move the
        # prediction off-screen. Recovery uses the last real box for gating;
        # covariance alone cannot establish identity after an observation gap.
        a=track.box if stage=='recover' else track.motion.box
        b=body['box']
        overlap=iou(a,b);last_overlap=iou(track.box,b)
        distance=self._distance(track,feature)
        # The first candidate can cover only a shoulder or include background.
        # A frozen long-term shirt template must not veto a geometrically
        # continuous observation. Keep a separate short-lived feature; it is
        # never used to assert identity after a long gap.
        if (stage!='recover' and captured-track.recent_time<=self.max_gap
                and last_overlap>=.35 and track.recent_feature is not None
                and feature is not None):
            distance=min(distance,appearance_distance(track.recent_feature,feature))
            overlap=max(overlap,.9*last_overlap)
        embedding=body.get('embedding')
        deep=None
        if embedding is not None and track.embedding is not None:
            deep=min(appearance_distance(sample,embedding) for sample in
                     (track.embedding,*track.embedding_gallery))
            if deep>.5:return np.inf
        if not .3<=b[2]/max(a[2],1)<=3. or not .35<=b[3]/max(a[3],1)<=2.8:return np.inf
        # The covariance cannot justify arbitrary jumps after a long gap.
        radius=.65 if stage=='low' else min(1.25,.6+.2*gap)
        if not self._near(a,b,radius):return np.inf
        if stage=='recover':
            if self.reid is not None:
                if deep is None or deep>.4:return np.inf
            else:
                # Last real high-score crop is separate from the immutable
                # anchor. It may restore a nearby short-term track, but only
                # with the same strict appearance threshold as the gallery.
                if feature is not None and track.high_feature is not None:
                    distance=min(distance,appearance_distance(track.high_feature,feature))
                if (feature is not None and track.recent_feature is not None
                        and captured-track.recent_time<=2.
                        and last_overlap>=.5):
                    distance=min(distance,appearance_distance(track.recent_feature,feature))
                if feature is None or distance>self.reid_distance:return np.inf
            if not self._near(track.box,b,1.25):return np.inf
        elif stage=='low':
            if gap>self.max_gap or overlap<.15:return np.inf
            if feature is not None and distance>self.max_distance:return np.inf
        elif distance>self.max_distance and (deep is None or deep>.35):
            return np.inf
        motion=0. if stage=='recover' else track.motion.distance(b)
        if motion>36. and overlap<.1:return np.inf
        return .55*(1-overlap)+.3*(deep if deep is not None else distance)+.15*min(motion/36.,1.)

    def _associate(self,pool,indices,bodies,features,captured,stage,blocked):
        if not pool or not indices:return []
        costs=np.array([[self._cost(t,bodies[j],features[j],captured,stage)
                         for j in indices] for t in pool])
        # Reject close row AND column alternatives before global allocation.
        # Otherwise the assignment can look unique while two people are tied.
        bad_rows=set();bad_cols=set()
        for row in range(len(pool)):
            valid=np.flatnonzero(np.isfinite(costs[row]))
            order=valid[np.argsort(costs[row,valid])]
            if len(order)>1 and costs[row,order[1]]-costs[row,order[0]]<self.margin:
                bad_rows.add(row);bad_cols.update(order.tolist())
        for col in range(len(indices)):
            valid=np.flatnonzero(np.isfinite(costs[:,col]))
            order=valid[np.argsort(costs[valid,col])]
            if len(order)>1 and costs[order[1],col]-costs[order[0],col]<self.margin:
                bad_cols.add(col);bad_rows.update(order.tolist())
        for row in bad_rows:
            blocked.update(indices[j] for j in np.flatnonzero(np.isfinite(costs[row])))
            costs[row,:]=np.inf
        for col in bad_cols:blocked.add(indices[col]);costs[:,col]=np.inf
        return [(pool[i],indices[j]) for i,j in assign(costs,.85)]

    def _observe(self,track,body,feature,captured,high):
        was_lost=track.state=='LOST'
        gap=captured-track.last_seen
        continuous=track.visible and gap<=self.max_gap
        safe_high=high and body.get('control_safe',True)
        track.high_streak=track.high_streak+1 if continuous and safe_high else int(safe_high)
        track.streak=track.streak+1 if continuous else 1
        track.hits+=1
        distance=self._distance(track,feature)
        old_box=track.box
        if (was_lost and gap>self.max_gap
                or max(body['box'][2]/max(old_box[2],1),old_box[2]/max(body['box'][2],1),
                       body['box'][3]/max(old_box[3],1),old_box[3]/max(body['box'][3],1))>1.5):
            track.motion=BoxMotion(body['box'])
        else:track.motion.update(body['box'])
        track.box=tuple(body['box']);track.last_seen=captured;track.score=body['score']
        track.similarity=max(0.,1-distance);track.visible=True
        if track.person_id<0 and high and track.high_streak>=self.confirm_frames:
            track.person_id=self.next_id;self.next_id+=1
            self.counters['public_ids_created']+=1
            # Confirmation represents the current person crop, rather than the
            # possibly incomplete first tentative proposal.
            if feature is not None:
                track.feature=feature.copy();track.anchor=feature.copy()
                track.gallery=deque([feature.copy()],maxlen=4)
        # Initial confirmation stays recorded through gaps; control readiness
        # requires current evidence and two high observations after a loss.
        track.confirmed=track.person_id>0
        track.control_ready=bool(track.confirmed and high and
                                 track.high_streak>=self.recover_frames
                                 and body.get('control_safe',True))
        track.state='TRACKED' if track.confirmed else 'TENTATIVE'
        if was_lost and track.confirmed:self.counters['recovered']+=1
        if not high:self.counters['low_matches']+=1
        if high and feature is not None and body.get('control_safe',True):
            track.recent_feature=feature.copy();track.recent_time=captured
            track.high_feature=feature.copy()
            track.high_time=captured
        elif feature is not None and track.recent_feature is not None:
            # Low scores extend a local observation chain only. No public ID,
            # long-term template or valid steering is created from this crop.
            overlap=iou(old_box,body['box'])
            same_scale=(.7<=body['box'][2]/max(old_box[2],1)<=1.4 and
                        .7<=body['box'][3]/max(old_box[3],1)<=1.4)
            local_low=(not high and captured-track.high_time<=1.
                       and overlap>=.65 and same_scale)
            limit=.5 if local_low else .35
            if overlap>=.5 and appearance_distance(track.recent_feature,feature)<limit:
                track.recent_feature=feature.copy();track.recent_time=captured
        embedding=body.get('embedding')
        if (high and embedding is not None and track.embedding is not None
                and continuous and track.control_ready and body.get('appearance_quality',True)
                and body['score']>=.55
                and appearance_distance(track.embedding_anchor,embedding)<.45):
            track.embedding=.95*track.embedding+.05*embedding
            track.embedding/=max(float(np.linalg.norm(track.embedding)),1e-9)
            track.embedding_gallery.append(embedding.copy())
        if embedding is not None:track.last_embedding=captured
        if (high and continuous and track.control_ready and gap<=self.max_gap
                and feature is not None and body.get('appearance_quality',True) and body['score']>=.55
                and distance<.4 and appearance_distance(track.anchor,feature)<.55):
            alpha=.96 if body['score']<.7 else .92
            track.feature=alpha*track.feature+(1-alpha)*feature
            track.feature/=max(float(np.linalg.norm(track.feature)),1e-9)
            track.feature_updates+=1;self.counters['feature_updates']+=1
            if track.feature_updates%10==0:track.gallery.append(feature.copy())

    def update(self,frame,bodies,captured):
        if not np.isfinite(captured) or captured<=self.last_time:
            for track in self.tracks.values():
                track.visible=False;track.control_ready=False;track.high_streak=0
            return []
        dt=captured-self.last_time if self.last_time else 0.
        self.last_time=captured
        survivors={}
        for key,track in self.tracks.items():
            lifetime=self.retain_seconds if track.confirmed else self.max_gap
            if captured-track.last_seen<=lifetime:
                track.motion.predict(dt)
                gap=captured-track.last_seen
                if gap>self.max_gap:
                    # Stop unconstrained size/velocity drift while temporarily
                    # lost. This state is association-only, never a control box.
                    track.motion.mean[4:]=0.
                    track.motion.mean[2:4]=track.box[2:4]
                survivors[key]=track
            else:self.counters['expired']+=1
        self.tracks=survivors
        height,width=frame.shape[:2];clean=[]
        for body in bodies:
            box=tuple(body['box']);score=float(body['score'])
            if len(box)!=4 or not np.all(np.isfinite(box)) or not np.isfinite(score):continue
            x,y,w,h=box
            if score<self.low_threshold or score>1 or w<20 or h<40:continue
            x1,y1=max(0.,x),max(0.,y);x2,y2=min(float(width),x+w),min(float(height),y+h)
            if x2-x1<20 or y2-y1<40:continue
            clean.append(dict(box=(x1,y1,x2-x1,y2-y1),score=score,
                              appearance_quality=x1>2 and y1>2 and x2<width-2 and y2<height-2))
        # Bound per-frame CPU work even if the detector sees a crowd.
        ordered=sorted(clean,key=lambda d:d['score'],reverse=True)
        bodies=[]
        for candidate in ordered:
            # At a lower detector threshold, nested weak boxes can be a second
            # proposal for one person. Never let them veto a good high-score
            # box or spawn a candidate. Equal/high-score overlap stays unsafe.
            box=candidate['box']
            duplicate=any(candidate['score']<self.high_threshold<=other['score']
                          and iou(box,other['box'])>.35
                          and intersection(box,other['box'])/max(1.,min(box[2]*box[3],other['box'][2]*other['box'][3]))>.8
                          for other in bodies)
            if not duplicate:bodies.append(candidate)
            if len(bodies)>=self.max_detections:break
        features=[descriptor(frame,b['box']) for b in bodies]
        unsafe={i for i in range(len(bodies)) if not self.isolated(i,bodies,self.high_threshold)}
        for i,body in enumerate(bodies):
            x,y,w,h=body['box']
            # A small fragment entering the bottom border is no longer a usable
            # body observation. Preserve its association, never steer from it.
            # This also covers a nearly square head-only proposal, not just a
            # wide shoulder crop. It is a conservative visibility gate.
            bottom_fragment=(y+h>=height-8 and h<height*.4)
            body['control_safe']=i not in unsafe and not bottom_fragment
            body['appearance_quality']&=i not in unsafe
        # Overlap vetoes steering and long-term appearance updates, not every
        # possible association. A uniquely matched low-score observation can
        # keep the ID alive through weak nested proposals. Row/column conflicts
        # still block ambiguous assignments in _associate.
        blocked=set()
        high=[i for i,b in enumerate(bodies) if b['score']>=self.high_threshold]
        low=[i for i,b in enumerate(bodies) if b['score']<self.high_threshold]
        if self.reid is not None:
            # At most two crops per frame. Stable geometric continuations sample
            # at 4 Hz; first sightings, appearance conflicts and lost recovery
            # need current embeddings. No stale crop is relabelled as current.
            budget=2
            for i in high:
                near=[t for t in self.tracks.values() if t.visible and
                      captured-t.last_seen<=self.max_gap and iou(t.motion.box,bodies[i]['box'])>.55]
                stable=(len(near)==1 and len(high)==1 and near[0].embedding is not None
                        and captured-near[0].last_embedding<self.reid_interval
                        and self._distance(near[0],features[i])<.45)
                if not stable and budget:
                    bodies[i]['embedding']=self.reid.encode(frame,bodies[i]['box']);budget-=1
        used=set();matched=set();visible=[]
        active=[t for t in self.tracks.values() if captured-t.last_seen<=self.max_gap and t.confirmed]
        tentative=[t for t in self.tracks.values() if not t.confirmed]
        lost=[t for t in self.tracks.values() if captured-t.last_seen>self.max_gap and t.confirmed]
        for pool,indices,stage in ((active,high,'high'),(active,low,'low'),
                                   (lost,high,'recover'),(tentative,high,'high')):
            pool=[t for t in pool if id(t) not in matched]
            indices=[i for i in indices if i not in used and i not in blocked]
            for track,index in self._associate(pool,indices,bodies,features,captured,stage,blocked):
                self._observe(track,bodies[index],features[index],captured,stage!='low')
                used.add(index);matched.add(id(track));visible.append(track)
        for track in self.tracks.values():
            if id(track) not in matched:
                track.visible=False;track.control_ready=False;track.high_streak=0
                track.state='LOST' if track.confirmed else 'TENTATIVE'
        # Do not create replacement identities beside an unresolved existing
        # trajectory. Wait for evidence instead of increasing the ID each frame.
        for i in high:
            if i in used or i in blocked or i in unsafe or features[i] is None:continue
            _,_,bw,bh=bodies[i]['box']
            # A head sliver or very wide partial proposal is not enough to
            # establish a new body identity. Existing tracks can still retain
            # such observations through the guarded matching stages.
            if bh<90 or bw<30 or bh/bw<.6:continue
            bx,by,_,_=bodies[i]['box']
            # Narrow slivers at a side border cannot establish an identity.
            # They often are curtain/chair false positives in the field clip.
            # A previously established person can still continue to that edge.
            if (bx<6 or bx+bw>width-6) and bw/bh<.6:continue
            if self.reid is not None and bodies[i].get('embedding') is None:continue
            if any(self._near(t.motion.box,bodies[i]['box'],.6) or
                   (t.confirmed and self._near(t.box,bodies[i]['box'],.6))
                   for t in self.tracks.values()):continue
            if len(self.tracks)>=self.max_tracks:continue
            key=self.next_candidate;self.next_candidate-=1
            body=bodies[i];feature=features[i]
            track=Track(key,body['box'],feature.copy(),body['score'],captured,
                        motion=BoxMotion(body['box']),anchor=feature.copy(),
                        gallery=deque([feature.copy()],maxlen=4),visible=True,
                        recent_feature=feature.copy(),recent_time=captured,
                        high_feature=feature.copy(),high_time=captured)
            if body.get('embedding') is not None:
                track.embedding=body['embedding'].copy();track.embedding_anchor=track.embedding.copy()
                track.embedding_gallery=deque([track.embedding.copy()],maxlen=4)
                track.last_embedding=captured
            self.tracks[key]=track;visible.append(track)
            self.counters['candidates_created']+=1
        if blocked or unsafe:self.counters['ambiguous_frames']+=1
        self.tracks={t.person_id:t for t in self.tracks.values()}
        return [dict(person_id=t.person_id,box=t.box,score=t.score,similarity=t.similarity,
                     confirmed=t.control_ready,established=t.confirmed,observed=True,
                     state=t.state,label='MOTION / GUARDED APPEARANCE') for t in visible]

    def metrics(self):
        result=dict(self.counters,retained_tracks=len(self.tracks),
                    confirmed_tracks=sum(t.confirmed for t in self.tracks.values()))
        if self.reid is not None:result['reid']=self.reid.metrics()
        return result


class BodySelection:
    """Selected ID stays selected on loss; no automatic switch to another ID."""
    def __init__(self):
        self.target_id=0;self.locked=False;self.box=None;self.mode='STANDBY'

    def select(self,pid):
        self.target_id=int(pid);self.locked=False;self.box=None;self.mode='VERIFY' if pid else 'STANDBY'

    def update(self,tracks,captured,width,height):
        candidates=[t for t in tracks if t['person_id']==self.target_id and t['confirmed']
                    and t.get('observed',True)]
        if self.target_id<=0 or len(candidates)!=1:
            self.locked=False;self.box=None;self.mode='LOST' if self.target_id else 'STANDBY'
            return Snapshot(self.target_id,captured=captured,image_width=width,image_height=height,source=0)
        track=candidates[0];self.locked=True;self.box=track['box'];self.mode='BODY'
        observation=Observation(self.target_id,*track['box'],track['score'],track['similarity'])
        # Existing firmware accepts source=2 for body observations. No face timestamp fabricated.
        return Snapshot(self.target_id,True,captured,observation,width,height,source=2,face_confirmed=0.)
