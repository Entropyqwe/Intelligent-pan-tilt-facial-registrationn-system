"""Evaluate real processed frames; camera/UI rates alone cannot pass tracking."""
import argparse
import json
from pathlib import Path


def evaluate(metrics):
    samples=metrics.get('quality_samples',[])
    color=bool(samples) and sum(s['color_fraction'] for s in samples)/len(samples)>.01
    visible=bool(samples) and sum(12<s['mean_luma']<245 and s['dark_fraction']<.8 and s['clipped_fraction']<.8 for s in samples)>=len(samples)*.8
    checks={
        'capture_at_least_640x480':metrics.get('capture_width',0)>=640 and metrics.get('capture_height',0)>=480,
        'color_3_channels':metrics.get('capture_channels')==3 and color,
        'not_mostly_black_or_white':visible,
        'measurement_at_least_60_seconds':metrics.get('elapsed_seconds',0)>=60,
        'real_processed_fps_at_least_20':metrics.get('steady_wall_fps',0)>=20,
        'npu_person_detection_at_least_20fps':metrics.get('body_detection_fps',0)>=20 and 'NPU' in metrics.get('backend',''),
        'tracking_valid_at_least_90_percent':metrics.get('valid_percent',0)>=90,
        'continuous_valid_at_least_10_seconds':metrics.get('longest_valid_seconds',0)>=10,
        'serial_opened_without_output_error':metrics.get('serial_enabled',False) and metrics.get('serial_error') is None,
        'original_steering_enabled':bool(metrics.get('steering_enabled')),
    }
    return dict(passed=all(checks.values()),checks=checks,measured=metrics,
                limits=['Color/brightness checks are heuristics, not visual quality certification.',
                        'Serial-open status does not prove MCU acceptance or physical servo movement.',
                        'Percentiles use up to the last 18000 measured frames; averages cover the measured run.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    report=evaluate(json.loads(args.metrics.read_text(encoding='utf-8')))
    rendered=json.dumps(report,ensure_ascii=False,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered,encoding='utf-8')
    print(rendered)
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
