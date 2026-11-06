"""Compare CPU thread counts on the deployment host, with no servo output."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int)
    source.add_argument('--image', type=Path)
    ap.add_argument('--candidates', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--face-every', type=int, default=3)
    ap.add_argument('--face-size', type=int, choices=(160,224,320), default=320)
    args = ap.parse_args()
    if args.frames < 1 or args.face_every < 1 or any(not 1 <= n <= 64 for n in args.candidates):
        ap.error('frames and face-every must be positive; candidates must be in 1..64')
    root = Path(__file__).resolve().parents[1]
    out = root/'logs'/('thread_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    source_args = ['--image', str(args.image.resolve())] if args.image else [
        '--camera', str(0 if args.camera is None else args.camera)]
    reports = []
    for threads in dict.fromkeys(args.candidates):
        print(f'Testing {threads} threads; keep the same person and scene visible...', flush=True)
        report_path = out/f'threads_{threads}.json'
        command = [sys.executable, str(root/'benchmark_vision.py'), *source_args,
                   '--frames', str(args.frames), '--face-every', str(args.face_every),
                   '--face-size', str(args.face_size), '--threads', str(threads),
                   '--output', str(report_path)]
        with (out/f'threads_{threads}.txt').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Benchmark failed; inspect {out / ("threads_"+str(threads)+".txt")}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        reports.append(report)
        print(f"  {report['loop_fps']:.2f} FPS; face mean {report['face']['mean_ms']:.2f} ms; "
              f"body mean {report['person']['mean_ms']:.2f} ms", flush=True)
    print('\nthreads  compute_FPS  loop_FPS  face_ms  body_ms  over_400ms')
    for r in reports:
        print(f"{r['threads']:7} {r['compute_fps']:12.2f} {r['loop_fps']:9.2f} "
              f"{r['face']['mean_ms']:8.2f} {r['person']['mean_ms']:8.2f} "
              f"{100*r['fraction_over_400ms']:10.1f}%")
    best = max(reports, key=lambda r: r['loop_fps'])
    print(f"\nFastest in this comparison: --threads {best['threads']} ({best['loop_fps']:.2f} FPS)")
    print('Repeat the comparison if the scene, temperatures or background load changed.')
    print('This measures CPU inference; the full application also draws and tracks.')
    print('Reports:', out)


if __name__ == '__main__':
    main()
