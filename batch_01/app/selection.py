"""Select a registered identity; the original tracker still confirms the lock."""
from body_tracker import face_inside, BodyTargetLock


def automatic_candidate(faces, bodies, body_mode=True):
    """Never choose between multiple visible registered identities."""
    known = [face for face in faces if face['person_id'] > 0]
    if len(known) != 1:
        return 0
    face = known[0]
    if not body_mode:
        return face['person_id']
    attached = [i for i, body in enumerate(bodies) if face_inside(face['box'], body['box'])]
    if len(attached) != 1 or not BodyTargetLock._isolated(attached[0], bodies):
        return 0
    contained = [f for f in faces if face_inside(f['box'], bodies[attached[0]]['box'])]
    return face['person_id'] if len(contained) == 1 else 0
