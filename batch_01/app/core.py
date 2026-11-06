"""Identity matching, target lock and serial protocol; no hardware dependencies."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import time
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class Gallery:
    """Persistent numbered identities. Registration is explicit, never automatic."""
    def __init__(self, path):
        self.path = Path(path)
        self.people = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema") != 1 or data.get("model") != "SFace-2021dec":
                raise ValueError("Unsupported gallery format or feature model")
            self.people = {int(k): [unit(x) for x in values]
                           for k, values in data["people"].items()}

    def add(self, samples, person_id=None):
        pid = person_id if person_id is not None else max(self.people, default=0) + 1
        if not samples or pid <= 0:
            raise ValueError("At least one sample and positive ID are required")
        self.people[pid] = (self.people.get(pid, []) + [unit(x) for x in samples])[-30:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": 1, "model": "SFace-2021dec", "people": {
            str(k): [v.tolist() for v in values] for k, values in self.people.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)
        return pid

    def match(self, embedding, threshold=0.45, margin=0.05):
        if not self.people:
            return 0, -1.0
        query = unit(embedding)
        scores = sorted(((max(float(query @ s) for s in samples), pid)
                         for pid, samples in self.people.items()), reverse=True)
        score, pid = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if score < threshold or score - second < margin:
            return 0, score
        return pid, score


def unique_identities(faces):
    """If two visible faces claim the same ID, mark both ambiguous."""
    counts = {}
    for f in faces:
        if f["person_id"]:
            counts[f["person_id"]] = counts.get(f["person_id"], 0) + 1
    for f in faces:
        if counts.get(f["person_id"], 0) > 1:
            f["person_id"] = 0
    return faces


@dataclass
class Observation:
    person_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    similarity: float


@dataclass
class Snapshot:
    target_id: int = 0
    valid: bool = False
    captured: float = 0.0
    observation: Observation = None
    image_width: int = 640
    image_height: int = 480
    source: int = 1  # F2: 0 invalid, 1 face verified, 2 body continuity
    face_confirmed: float = 0.0


class TargetLock:
    """Never switches to another ID without an explicit user command."""
    def __init__(self, confirm_frames=3, max_gap=0.5):
        self.target_id = 0
        self.confirm_frames = confirm_frames
        self.max_gap = max_gap
        self.streak = 0
        self.last_seen = 0.0

    def select(self, person_id):
        self.target_id = int(person_id)
        self.streak = 0
        self.last_seen = 0.0

    def update(self, observations, captured, width, height):
        candidates = [o for o in observations if o.person_id == self.target_id]
        if self.target_id <= 0 or len(candidates) != 1:
            self.streak = 0
            return Snapshot(self.target_id, captured=captured,
                            image_width=width, image_height=height)
        if self.last_seen and captured - self.last_seen > self.max_gap:
            self.streak = 0
        self.last_seen = captured
        self.streak += 1
        return Snapshot(self.target_id, self.streak >= self.confirm_frames,
                        captured, candidates[0], width, height)


def control_fields(snapshot, now=None, max_age=0.4, deadband=120):
    now = time.monotonic() if now is None else now
    age_ms = max(0, round((now - snapshot.captured) * 1000)) if snapshot.captured else 999999
    fresh = snapshot.captured > 0 and 0 <= now - snapshot.captured <= max_age
    valid = bool(snapshot.valid and snapshot.observation is not None and fresh)
    r = dict(target_id=snapshot.target_id, valid=int(valid),
             cx=0, cy=0, ex=0, ey=0, bw=0, bh=0, area=0,
             det=0, sim=0, direction=0, age_ms=age_ms, source=0,
             face_age_ms=(max(0, round((now-snapshot.face_confirmed)*1000))
                          if snapshot.face_confirmed else 999999))
    if valid:
        o = snapshot.observation
        cx = np.clip((o.x + o.w / 2) / snapshot.image_width, 0, 1)
        cy = np.clip((o.y + o.h / 2) / snapshot.image_height, 0, 1)
        r.update(cx=round(cx * 1000), cy=round(cy * 1000), source=snapshot.source,
                 ex=round((2 * cx - 1) * 1000), ey=round((2 * cy - 1) * 1000),
                 bw=round(o.w / snapshot.image_width * 1000),
                 bh=round(o.h / snapshot.image_height * 1000),
                 area=round(o.w * o.h / snapshot.image_width / snapshot.image_height * 1000),
                 det=round(o.score * 1000), sim=round(o.similarity * 1000))
        r["direction"] = -1 if r["ex"] < -deadband else (1 if r["ex"] > deadband else 0)
    return r


WIRE_FIELDS = ("seq", "valid", "target_id", "cx", "cy", "ex", "ey", "bw", "bh",
               "area", "det", "sim", "direction", "age_ms")
WIRE_FIELDS_V2 = WIRE_FIELDS + ("source", "face_age_ms")


def encode_packet(seq, fields, version="F1"):
    if version not in ("F1", "F2"):
        raise ValueError("Unsupported packet version")
    values = dict(fields, seq=seq)
    keys = WIRE_FIELDS_V2 if version == "F2" else WIRE_FIELDS
    body = version + "," + ",".join(str(values[k]) for k in keys)
    checksum = 0
    for b in body.encode("ascii"):
        checksum ^= b
    return f"@{body}*{checksum:02X}\n".encode("ascii")


def decode_packet(packet):
    text = packet.decode("ascii") if isinstance(packet, bytes) else packet
    body, check = text.strip().removeprefix("@").rsplit("*", 1)
    checksum = 0
    for b in body.encode("ascii"):
        checksum ^= b
    if not text.startswith("@") or checksum != int(check, 16):
        raise ValueError("Packet checksum or start marker invalid")
    parts = body.split(",")
    if parts[0] not in ("F1", "F2"):
        raise ValueError("Packet version or field count invalid")
    keys = WIRE_FIELDS_V2 if parts[0] == "F2" else WIRE_FIELDS
    if len(parts) != len(keys) + 1:
        raise ValueError("Packet version or field count invalid")
    values = dict(zip(keys, map(int, parts[1:])))
    if parts[0] == "F2":
        if values["source"] not in (0, 1, 2) or values["face_age_ms"] < 0:
            raise ValueError("Packet observation source or face age invalid")
        if bool(values["valid"]) != bool(values["source"]):
            raise ValueError("Packet validity and observation source disagree")
    if len(check) != 2 or values["seq"] <= 0 or values["target_id"] < 0 or values["age_ms"] < 0:
        raise ValueError("Packet counter, identity or age invalid")
    if values["valid"] not in (0, 1) or values["direction"] not in (-1, 0, 1):
        raise ValueError("Packet flag invalid")
    if any(not 0 <= values[k] <= 1000 for k in ("cx", "cy", "bw", "bh", "area", "det")):
        raise ValueError("Packet geometry or detection score out of range")
    if any(not -1000 <= values[k] <= 1000 for k in ("ex", "ey", "sim")):
        raise ValueError("Packet error or similarity out of range")
    if values["valid"] and values["target_id"] == 0:
        raise ValueError("Valid packet requires a selected identity")
    if not values["valid"] and any(values[k] for k in ("cx", "cy", "ex", "ey", "bw", "bh", "area", "det", "sim", "direction")):
        raise ValueError("Invalid observation must have zeroed position fields")
    return values
