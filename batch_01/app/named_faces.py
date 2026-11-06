"""Explicit named SFace enrollment. No automatic enrollment or image storage."""
import json
import os
from pathlib import Path
import re
import numpy as np


def embedding(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 128 or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError('Pinned SFace feature must contain 128 finite, nonzero values')
    return vector / norm


def checked_name(name):
    name = str(name).strip()
    if not 1 <= len(name) <= 24 or re.search(r'[\x00-\x1f\x7f/\\]', name):
        raise ValueError('姓名须为1～24个可见字符，不能含路径分隔符或控制字符')
    return name


class NamedGallery:
    def __init__(self, path):
        self.path = Path(path)
        self.people = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('schema') != 1 or data.get('model') != 'SFace-2021dec-named':
                raise ValueError('Unsupported named gallery format')
            for key, row in data['people'].items():
                pid = int(key)
                samples = [embedding(v) for v in row['samples']]
                if pid <= 0 or not 1 <= len(samples) <= 30:
                    raise ValueError('Invalid enrollment ID/sample count')
                self.people[pid] = dict(name=checked_name(row['name']), samples=samples)

    def name(self, pid):
        return self.people.get(pid, {}).get('name', '未登记')

    def match(self, feature, threshold=.45, margin=.05):
        query = embedding(feature)
        scores = sorted(((max(float(query @ v) for v in row['samples']), pid)
                         for pid, row in self.people.items()), reverse=True)
        if not scores:
            return 0, -1.
        score, pid = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.
        return (pid if score >= threshold and score-second >= margin else 0), score

    def add(self, name, samples, person_id=None):
        name = checked_name(name)
        values = [embedding(v) for v in samples]
        if not values:
            raise ValueError('At least one sample is required')
        # Repeating a name explicitly appends samples to the existing identity.
        matches = [pid for pid, row in self.people.items() if row['name'] == name]
        pid = person_id if person_id is not None else (matches[0] if matches else max(self.people, default=0)+1)
        if pid <= 0:
            raise ValueError('ID must be positive')
        if pid in self.people and self.people[pid]['name'] != name:
            raise ValueError('Existing ID has another name; delete it explicitly before reuse')
        self.people[pid] = dict(name=name, samples=(self.people.get(pid, {}).get('samples', [])+values)[-30:])
        self.save()
        return pid

    def delete(self, pid):
        del self.people[pid]
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(schema=1, model='SFace-2021dec-named', people={str(pid): dict(
            name=row['name'], samples=[v.tolist() for v in row['samples']]) for pid, row in self.people.items()})
        tmp = self.path.with_name(self.path.name+'.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(self.path)
