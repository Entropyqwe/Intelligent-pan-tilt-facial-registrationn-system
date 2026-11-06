"""Conservative face-to-body association; continuity is not body biometrics."""
from core import Observation, Snapshot


def intersection(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0., min(ax+aw, bx+bw)-max(ax, bx)) * max(
        0., min(ay+ah, by+bh)-max(ay, by))


def iou(a, b):
    shared = intersection(a, b)
    return shared / max(a[2]*a[3]+b[2]*b[3]-shared, 1e-9)


def face_inside(face, body):
    x, y, w, h = face
    bx, by, bw, bh = body
    return (intersection(face, body)/max(w*h, 1e-9) >= .8
            and bx <= x+w/2 <= bx+bw
            and by <= y+h/2 <= by+.65*bh)


class BodyTargetLock:
    """Bind after face confirmation, continue only an unambiguous body chain.

    On any missed detection, overlap, excessive movement, identity conflict or
    timestamp gap, discard the chain. Only the selected face can restore it.
    """
    def __init__(self, confirm_frames=3, max_gap=.5, min_iou=.25):
        self.confirm_frames, self.max_gap, self.min_iou = confirm_frames, max_gap, min_iou
        self.target_id = 0
        self.select(0)

    def select(self, person_id):
        self.target_id = int(person_id)
        self._lose()
        self.face_confirmed = 0.

    def _lose(self):
        self.box = None
        self.last_seen = 0.
        self.streak = 0
        self.locked = False
        self.mode = "LOST" if self.target_id else "UNLOCKED"

    def _plausible(self, box):
        if self.box is None:
            return False
        x, y, w, h = self.box
        bx, by, bw, bh = box
        ratio = bw*bh/max(w*h, 1e-9)
        movement = (((bx+bw/2-x-w/2)/max(w, 1))**2
                    + ((by+bh/2-y-h/2)/max(h, 1))**2)**.5
        return .5 <= ratio <= 2 and movement <= .6 and iou(self.box, box) >= self.min_iou

    @staticmethod
    def _isolated(index, bodies):
        box = bodies[index]["box"]
        for j, other in enumerate(bodies):
            if j == index:
                continue
            shared = intersection(box, other["box"])
            smaller = min(box[2]*box[3], other["box"][2]*other["box"][3])
            if iou(box, other["box"]) > .15 or shared/max(smaller, 1e-9) > .25:
                return False
        return True

    def update(self, faces, bodies, captured, width, height, face_checked=True):
        def empty():
            return Snapshot(self.target_id, captured=captured, image_width=width,
                            image_height=height, source=0, face_confirmed=self.face_confirmed)

        if self.target_id <= 0:
            return empty()
        if self.last_seen and not 0 < captured-self.last_seen <= self.max_gap:
            self._lose()
        matches = [f for f in faces if f["person_id"] == self.target_id]
        if len(matches) > 1:
            self._lose()
            return empty()

        index, verified = None, False
        if len(matches) == 1:
            attached = [i for i, b in enumerate(bodies)
                        if face_inside(matches[0]["box"], b["box"])]
            if len(attached) == 1:
                index, verified = attached[0], True
        if index is None and self.locked:
            possible = [i for i, b in enumerate(bodies) if self._plausible(b["box"])]
            if len(possible) == 1:
                index = possible[0]
        if index is None or not self._isolated(index, bodies):
            self._lose()
            return empty()

        body = bodies[index]
        contained = [f for f in faces if face_inside(f["box"], body["box"])]
        if len(contained) > 1 or any(f["person_id"] not in (0, self.target_id) for f in contained):
            self._lose()
            return empty()
        if face_checked and any(f["person_id"] == 0 for f in contained):
            self._lose()
            return empty()
        # A recognised target elsewhere must never allow continuation on this body.
        if matches and not verified:
            self._lose()
            return empty()
        if self.box is not None and not self._plausible(body["box"]):
            self._lose()
            return empty()
        if not self.locked and not verified:
            self._lose()
            return empty()

        self.box, self.last_seen = body["box"], captured
        if verified:
            self.face_confirmed = captured
            self.streak += 1
            self.locked = self.locked or self.streak >= self.confirm_frames
        if not self.locked:
            self.mode = "VERIFY"
            return empty()
        self.mode = "FACE" if verified else "BODY"
        obs = Observation(self.target_id, *body["box"], body["score"],
                          matches[0]["similarity"] if verified else -1.)
        return Snapshot(self.target_id, True, captured, obs, width, height,
                        source=1 if verified else 2, face_confirmed=self.face_confirmed)
