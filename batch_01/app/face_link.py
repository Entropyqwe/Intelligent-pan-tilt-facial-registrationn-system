"""Conservative face/body binding. A body slot is never a face identity."""
import math


def continuous(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if min(aw, ah, bw, bh) <= 0:
        return False
    return (.5 <= aw/bw <= 2 and .5 <= ah/bh <= 2 and
            math.hypot(ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2) < .4*max(ah, bh))


def associate(faces, tracks):
    """Require a unique face inside one body's upper 55%; reject overlaps."""
    proposals = {}
    counts = {}
    for face in faces:
        fx, fy, fw, fh = face['box']
        cx, cy = fx+fw/2, fy+fh/2
        candidates = []
        for tid, track in tracks.items():
            x, y, w, h = track['box']
            ix = max(0, min(fx+fw, x+w)-max(fx, x))
            iy = max(0, min(fy+fh, y+h*.55)-max(fy, y))
            if (fw > 0 and fh > 0 and x <= cx <= x+w and y <= cy <= y+h*.55
                    and ix*iy/(fw*fh) >= .8):
                candidates.append(tid)
        for tid in candidates:counts[tid]=counts.get(tid,0)+1
        if len(candidates) == 1:
            proposals.setdefault(candidates[0], []).append(face)
    return {tid: values[0] for tid, values in proposals.items()
            if len(values) == 1 and counts.get(tid)==1}


class FaceBindings:
    def __init__(self, gallery, max_age=.9, identity_ttl=1.5):
        self.gallery = gallery
        self.max_age, self.identity_ttl = max_age, identity_ttl
        self.tracks = {}
        self.counter = 0
        self.last_result = None

    def observe(self, tracks, captured):
        visible = {}
        for track in tracks:
            tid = track['person_id']
            if tid <= 0 or not track.get('confirmed') or not track.get('observed', True):
                continue
            old = self.tracks.get(tid)
            if old is None or captured-old['seen'] > .45 or not continuous(old['box'], track['box']):
                self.counter += 1
                old = dict(token=self.counter, pid=0, streak=0, started=captured,
                           verified=False, checked=-float('inf'), rel=None)
            old.update(box=tuple(track['box']), seen=captured)
            visible[tid] = old
        # Absence revokes the binding immediately, even if slot 01 reappears.
        self.tracks = visible

    def tokens(self):
        return {tid: dict(token=row['token'], box=row['box']) for tid, row in self.tracks.items()}

    def accept(self, result, now):
        stamp = result['captured']
        if not 0 <= now-stamp <= self.max_age or (self.last_result is not None and stamp <= self.last_result):
            return False
        self.last_result = stamp
        assigned = associate(result['faces'], result['tracks'])
        for tid, source in result['tracks'].items():
            row = self.tracks.get(tid)
            if row is None or row['token'] != source['token']:
                continue
            face = assigned.get(tid)
            pid = face['person_id'] if face else 0
            if not pid or pid != row['pid'] or stamp-row['checked'] > 1.2:
                row.update(pid=pid, streak=0, started=stamp, verified=False)
            row['checked'] = stamp
            row['rel'] = None
            if face:
                x, y, w, h = source['box']
                fx, fy, fw, fh = face['box']
                row['rel'] = ((fx-x)/w, (fy-y)/h, fw/w, fh/h)
            if pid:
                row['streak'] += 1
                row['verified'] = row['streak'] >= 3 and stamp-row['started'] >= .5
        return True

    def labels(self, now):
        labels = {}
        for tid, row in self.tracks.items():
            if not 0 <= now-row['checked'] <= self.identity_ttl:
                continue
            verified = row['verified'] and row['pid'] in self.gallery.people
            label = self.gallery.name(row['pid']) if verified else ('确认中' if row['pid'] else '未登记')
            box = None
            if row['rel'] is not None:
                x, y, w, h = row['box']
                rx, ry, rw, rh = row['rel']
                box = (x+rx*w, y+ry*h, rw*w, rh*h)
            labels[tid] = dict(name=label, verified=verified, identity_id=row['pid'] if verified else 0,
                               box=box, age=now-row['checked'])
        ids=[row['identity_id'] for row in labels.values() if row['verified']]
        for row in labels.values():
            if row['identity_id'] and ids.count(row['identity_id'])>1:
                row.update(name='身份歧义',verified=False,identity_id=0)
        return labels
