#!/usr/bin/env python3
"""
Train one of the *_dlss_distill notebooks from scratch and publish live progress for the dashboard.

    python3 train_dashboard/train.py superresolution --max-hours 3

The model, dataset and degradation code is executed straight from the notebook cells, so the
notebook and this script can not drift apart. Progress is written to runs/<name>/ :
status.json (numbers, loss/psnr history), img/*.jpg (input | output | original for all Set14 images)
and checkpoints. Create the file runs/<name>/STOP (the dashboard button does this) to stop gracefully.
"""
import argparse
import collections
import copy
import io
import json
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELL = {'options': 2, 'imports': 4, 'dataset': 8, 'base': 10, 'build': 12}

ns = {}  # namespace the notebook cells are executed in (module level, so forked workers see it)


def run_cell(nb, index):
    lines = []
    for line in ''.join(nb['cells'][index]['source']).splitlines():
        # drop notebook magics (%cd, !git clone, %matplotlib) but keep indentation valid
        lines.append(line[:len(line) - len(line.lstrip())] + 'pass' if line.strip().startswith(('%', '!')) else line)
    src = '\n'.join(lines).replace('run_on_google_colab = True', 'run_on_google_colab = False')
    exec(compile(src, f'cell{index}', 'exec'), ns)


def worker_init():
    import torch
    torch.set_num_threads(1)
    random.seed(os.getpid() * 7919 + int(time.time() * 1000) % 100000)
    np.random.seed(os.getpid() % 2 ** 31)


def make_batch():
    x, y = ns['get_batch'](ns['batch_size'], ns['height_lr'], ns['width_lr'], ns['channels'])
    return x.numpy(), y.numpy()


def atomic_write(path, text):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', choices=['superresolution', 'inpainting', 'deblurring'])
    ap.add_argument('--max-hours', type=float, default=3.0)
    ap.add_argument('--eval-every', type=float, default=120.0, help='seconds between evaluations on Set14')
    ap.add_argument('--workers', type=int, default=2, help='processes that generate training batches')
    ap.add_argument('--fresh', action='store_true', help='ignore an existing checkpoint')
    args = ap.parse_args()

    name = args.name
    run_dir = ROOT / 'runs' / name
    (run_dir / 'img').mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / 'STOP'
    stop_all = ROOT / 'runs' / 'STOP_ALL'
    max_seconds = args.max_hours * 3600.0

    status = {'name': name, 'state': 'starting', 'max_seconds': max_seconds, 'step': 0, 'elapsed': 0.0, 'lr': 0.0,
              'loss': None, 'steps_per_sec': 0.0, 'history': [], 'psnr_history': [], 'images': {}, 'eval_version': 0,
              'eval_time': None, 'message': '', 'params': None, 'updated': time.time()}

    def publish(**kw):
        status.update(kw)
        status['updated'] = time.time()
        atomic_write(run_dir / 'status.json', json.dumps(status))

    publish()
    try:
        os.chdir(ROOT)  # the notebooks read ./Set14
        nb = json.load(open(ROOT / f'{name}_dlss_distill.ipynb'))
        for key in ('options', 'imports', 'dataset', 'base'):
            run_cell(nb, CELL[key])
        import torch
        import torch.nn.functional as F
        from PIL import Image

        train_dev = ns['device']
        ns['device'] = torch.device('cpu')  # batches are generated on the cpu, the model is moved below

        # fork the batch workers before the accelerator is touched
        pool = mp.get_context('fork').Pool(args.workers, initializer=worker_init)
        torch.set_num_threads(2)

        run_cell(nb, CELL['build'])
        generator = ns['generator'].to(train_dev)
        psnr = lambda a, b: float(10 * torch.log10(1. / F.mse_loss(a, b).clamp_min(1e-10)))
        images = ns['images']
        names = [Path(f).stem for f in sorted(__import__('glob').glob('./Set14/*.png'))]
        scale = ns.get('scale', 1)
        base_lr = ns['gen_lr']
        batch_size = ns['batch_size']
        publish(params=sum(p.numel() for p in generator.parameters()),
                config={'batch_size': batch_size, 'height_lr': ns['height_lr'], 'gen_lr': base_lr, 'device': str(train_dev)})

        # ---- fixed evaluation pairs: every Set14 image, degraded deterministically
        def make_pair(img, seed):
            random.seed(seed)
            hr = torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())
            if name == 'superresolution':
                hr = hr[:, :hr.shape[1] // scale * scale, :hr.shape[2] // scale * scale]
                inp = F.avg_pool2d(hr[None], scale)[0]
                base = F.interpolate(inp[None], scale_factor=scale, mode='bicubic', align_corners=False).clamp(0, 1)[0]
            elif name == 'inpainting':
                inp = ns['cut_holes'](hr.clone(), hr.shape[1], hr.shape[2])
                base = inp
            else:
                blurred = ns['motion_blur'](img * 255., img.shape[0] / 100) / 255.
                inp = torch.from_numpy(np.transpose(blurred, (2, 0, 1)).copy()).float()
                base = inp
            return inp[None], hr[None], base[None]

        eval_set = [make_pair(img, 1000 + i) for i, img in enumerate(images)]
        for (inp, hr, base), n in zip(eval_set, names):
            status['images'][n] = {'psnr_base': psnr(base, hr), 'psnr_model': None}

        def to_u8(t):
            return (t[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

        def evaluate(elapsed, step):
            generator.eval()
            model_psnr, base_psnr = [], []
            with torch.no_grad():
                for (inp, hr, base), n in zip(eval_set, names):
                    out = generator(inp.to(train_dev)).cpu().clamp(0, 1)
                    p_model, p_base = psnr(out, hr), psnr(base, hr)
                    model_psnr.append(p_model)
                    base_psnr.append(p_base)
                    status['images'][n] = {'psnr_model': p_model, 'psnr_base': p_base}
                    shown_in = F.interpolate(inp, size=hr.shape[-2:], mode='nearest') if name == 'superresolution' else inp
                    triple = np.concatenate([to_u8(shown_in), to_u8(out), to_u8(hr)], axis=1)
                    im = Image.fromarray(triple)
                    if im.width > 900:
                        im = im.resize((900, max(1, round(im.height * 900 / im.width))), Image.LANCZOS)
                    im.save(run_dir / 'img' / f'{n}.jpg', quality=82)
            generator.train()
            status['psnr_history'].append([elapsed, step, float(np.mean(model_psnr)), float(np.mean(base_psnr))])
            status['eval_version'] += 1
            status['eval_time'] = time.time()
            publish()

        # ---- optimizer, optional resume
        optimizer = torch.optim.Adam(generator.parameters(), lr=base_lr)
        step, elapsed_before, ema = 0, 0.0, None
        ckpt_path = run_dir / 'ckpt.pt'
        if ckpt_path.exists() and not args.fresh:
            ck = torch.load(ckpt_path, map_location='cpu')
            generator.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optim'])
            step, elapsed_before, ema = ck['step'], ck['elapsed'], ck['ema']
            status['history'], status['psnr_history'] = ck['history'], ck['psnr_history']
            status['message'] = f'resumed from step {step}'

        stop_requested = {'sigterm': False}
        signal.signal(signal.SIGTERM, lambda *a: stop_requested.update(sigterm=True))

        generator.train()
        t_start = time.time()
        last_eval = last_ckpt = last_publish = last_hist = t_start
        if not status['psnr_history']:
            evaluate(elapsed_before, step)
        pending = collections.deque(pool.apply_async(make_batch) for _ in range(max(4, args.workers * 3)))
        rate_t, rate_step = t_start, step
        publish(state='running')

        reason = None
        while True:
            elapsed = elapsed_before + time.time() - t_start
            if stop_file.exists() or stop_all.exists() or stop_requested['sigterm']:
                reason = 'stopped'
                break
            if elapsed >= max_seconds:
                reason = 'finished'
                break
            lr = base_lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, elapsed / max_seconds))))
            for g in optimizer.param_groups:
                g['lr'] = lr

            xb, yb = pending.popleft().get()
            pending.append(pool.apply_async(make_batch))
            X, Y = torch.from_numpy(xb).to(train_dev), torch.from_numpy(yb).to(train_dev)
            loss = F.l1_loss(generator(X), Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            lv = float(loss.item())
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv

            now = time.time()
            if now - last_hist >= 10:
                status['history'].append([elapsed, step, ema])
                last_hist = now
            if now - last_publish >= 3:
                publish(step=step, elapsed=elapsed, lr=lr, loss=ema, steps_per_sec=(step - rate_step) / max(1e-6, now - rate_t))
                rate_t, rate_step, last_publish = now, step, now
            if now - last_eval >= args.eval_every:
                evaluate(elapsed, step)
                last_eval = time.time()
            if now - last_ckpt >= 300:
                torch.save({'model': generator.state_dict(), 'optim': optimizer.state_dict(), 'step': step, 'elapsed': elapsed,
                            'ema': ema, 'history': status['history'], 'psnr_history': status['psnr_history']}, ckpt_path)
                last_ckpt = time.time()

        elapsed = elapsed_before + time.time() - t_start
        pool.terminate()
        evaluate(elapsed, step)
        torch.save(generator.state_dict(), run_dir / f'{name}_dlss_distill_weights.pt')
        deploy = copy.deepcopy(generator).cpu()
        deploy.switch_to_deploy()
        torch.save(deploy.state_dict(), run_dir / f'{name}_dlss_distill_deploy.pt')
        torch.save({'model': generator.state_dict(), 'optim': optimizer.state_dict(), 'step': step, 'elapsed': elapsed,
                    'ema': ema, 'history': status['history'], 'psnr_history': status['psnr_history']}, ckpt_path)
        publish(state=reason, step=step, elapsed=elapsed, loss=ema,
                message='stopped by you' if reason == 'stopped' else 'time limit reached')
    except Exception:
        publish(state='crashed', message=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
