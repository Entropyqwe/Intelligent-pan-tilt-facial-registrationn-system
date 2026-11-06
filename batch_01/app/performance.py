"""Count completed frames using elapsed wall time; no synthetic display frames."""
from collections import deque


class PerformanceMeter:
    def __init__(self, started, window=2.):
        self.window = window
        self.samples = deque([(started, 0, False, False, 0., 0., 0., 0., False)])

    def add(self, done, camera_seq, face_checked, body_checked,
            face_ms, body_ms, pipeline_ms, age_ms, valid):
        self.samples.append((done, camera_seq, face_checked, body_checked,
                             face_ms, body_ms, pipeline_ms, age_ms, valid))
        # Keep one boundary sample, followed by completions counted in the window.
        while len(self.samples) > 2 and self.samples[1][0] <= done-self.window:
            self.samples.popleft()

    def read(self):
        samples = list(self.samples)
        frames = samples[1:]
        elapsed = samples[-1][0]-samples[0][0]
        if not frames or elapsed <= 0:
            return dict(fps=0., camera_fps=0., face_fps=0., body_fps=0.,
                        face_ms=0., body_ms=0., pipeline_ms=0., age_ms=0., valid_percent=0.)
        n = len(frames)
        return dict(fps=n/elapsed,
                    camera_fps=(samples[-1][1]-samples[0][1])/elapsed,
                    face_fps=sum(s[2] for s in frames)/elapsed,
                    body_fps=sum(s[3] for s in frames)/elapsed,
                    face_ms=sum(s[4] for s in frames)/n,
                    body_ms=sum(s[5] for s in frames)/n,
                    pipeline_ms=sum(s[6] for s in frames)/n,
                    age_ms=frames[-1][7], valid_percent=100*sum(s[8] for s in frames)/n)


def describe(stats):
    return (f"[PERF] FPS={stats['fps']:.1f} camera={stats['camera_fps']:.1f} "
            f"face={stats['face_ms']:.1f}ms body={stats['body_ms']:.1f}ms "
            f"age={stats['age_ms']:.0f}ms valid={stats['valid_percent']:.0f}%")
