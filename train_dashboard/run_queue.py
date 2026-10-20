#!/usr/bin/env python3
"""
Train the three examples one after another (one at a time, so each gets the whole accelerator).

    python3 train_dashboard/run_queue.py --max-hours 3

Order: superresolution, inpainting, deblurring. Pressing "Stopp" in the dashboard ends the current
example and the queue moves on; "Überspringen" on a waiting example (or the file runs/<name>/STOP)
skips it; "Alles beenden" (runs/STOP_ALL) stops the current one and skips the rest.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
ORDER = ['superresolution', 'inpainting', 'deblurring']
WORKERS = {'superresolution': 2, 'inpainting': 2, 'deblurring': 6}  # processes that generate training batches

ap = argparse.ArgumentParser()
ap.add_argument('--max-hours', type=float, default=3.0, help='time limit per example')
args = ap.parse_args()

RUNS.mkdir(exist_ok=True)
for f in [RUNS / 'STOP_ALL'] + [RUNS / n / 'STOP' for n in ORDER]:
    f.unlink(missing_ok=True)

for n in ORDER:
    if (RUNS / 'STOP_ALL').exists() or (RUNS / n / 'STOP').exists():
        (RUNS / n).mkdir(parents=True, exist_ok=True)
        (RUNS / n / 'status.json').write_text(json.dumps({'name': n, 'state': 'skipped', 'updated': time.time()}))
        print(f'{n}: skipped', flush=True)
        continue
    print(f'{n}: starting (max {args.max_hours} h)', flush=True)
    rc = subprocess.run([sys.executable, '-I', str(HERE / 'train.py'), n, '--max-hours', str(args.max_hours),
                         '--workers', str(WORKERS[n]), '--fresh']).returncode
    print(f'{n}: exited with code {rc}', flush=True)
print('queue finished', flush=True)
