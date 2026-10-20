#!/usr/bin/env python3
"""
Tiny dashboard server for the training runs: http://localhost:9000

    python3 train_dashboard/server.py [--port 9000]

GET  /                      dashboard page
GET  /api/status            status of all runs (reads runs/<name>/status.json)
GET  /runs/<name>/img/<f>   result images written by train.py
POST /api/stop/<name>       ask a run to stop (creates runs/<name>/STOP); name 'all' stops the running one and skips the rest
"""
import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
NAMES = ['superresolution', 'inpainting', 'deblurring']


def read_status():
    out = {}
    for n in NAMES:
        p = RUNS / n / 'status.json'
        try:
            out[n] = json.loads(p.read_text())
            # a run that stopped writing without a final state has died
            if out[n].get('state') in ('starting', 'running') and time.time() - out[n].get('updated', 0) > 90:
                out[n]['state'] = 'stalled'
        except (OSError, ValueError):
            out[n] = {'name': n, 'state': 'not started'}
    return {'now': time.time(), 'runs': out}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json', cache=False):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if not cache:
            self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self._send(200, (HERE / 'index.html').read_text(), 'text/html; charset=utf-8')
        if path == '/api/status':
            return self._send(200, json.dumps(read_status()))
        parts = path.strip('/').split('/')
        if len(parts) == 4 and parts[0] == 'runs' and parts[1] in NAMES and parts[2] == 'img' and parts[3].endswith('.jpg'):
            f = RUNS / parts[1] / 'img' / Path(parts[3]).name
            if f.is_file():
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or 'image/jpeg', cache=True)
        self._send(404, '{"error": "not found"}')

    def do_POST(self):
        parts = self.path.strip('/').split('/')
        if len(parts) == 3 and parts[:2] == ['api', 'stop'] and (parts[2] in NAMES or parts[2] == 'all'):
            if parts[2] == 'all':
                RUNS.mkdir(exist_ok=True)
                (RUNS / 'STOP_ALL').write_text('stop')
            else:  # also works for a run that has not started yet: the queue then skips it
                (RUNS / parts[2]).mkdir(parents=True, exist_ok=True)
                (RUNS / parts[2] / 'STOP').write_text('stop')
            return self._send(200, '{"ok": true}')
        self._send(404, '{"error": "not found"}')

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9000)
    args = ap.parse_args()
    # localhost only: the page can stop training runs
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'dashboard on http://localhost:{args.port}', flush=True)
    srv.serve_forever()
