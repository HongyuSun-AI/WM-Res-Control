import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent
def read_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def stage(command,log,deadline,timeout):
    with log.open('w') as stream:
        child=subprocess.Popen([sys.executable]+[str(v) for v in command],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            until=min(deadline,time.monotonic()+timeout)
            while child.poll() is None:
                if time.monotonic()>=until:raise TimeoutError('Stage/time budget expired')
                time.sleep(.5)
            return child.returncode
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise

def lane_summary(directory):
    rows=read_lines(directory/'lane_metrics.jsonl')
    covered=[r for r in rows if r['covered']]
    longest=run=0
    for r in rows:
        run=run+1 if r['covered'] and r['outside_depth_m']>=.15 else 0
        longest=max(longest,run)
    refs=read_lines(next(directory.glob('reference_*.jsonl')))
    xy=np.array([[r['state']['x'],r['state']['y']] for r in refs])
    traveled=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()) if len(xy)>1 else 0.
    return dict(frames=len(rows),covered=len(covered),coverage=len(covered)/len(rows) if rows else 0,
                max_outside_m=max((r['outside_depth_m'] for r in covered),default=None),
                outside_frames=sum(r['outside'] for r in covered),
                outside_015m_longest_frames=longest,travel_m=traveled)
