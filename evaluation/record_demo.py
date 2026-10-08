import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'experiments/low_friction'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenario', choices=['autonomous', 'speed20'], default='autonomous', help='autonomous: collision; speed20: 20km/h')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--steps', type=int, default=1200)
    a = p.parse_args()
    if not 41 <= a.steps <= 1200: p.error('--steps must be 41..1200')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    results = []
    scenario = a.scenario
    folder = out / scenario; folder.mkdir()
    for mode in ['tcp', 'direct', 'sac']:
        dest = folder / mode
        cmd = [sys.executable, str(CORE/'record_tcp_reference.py'), '--output', str(dest),
               '--weather', 'MidRainyNoon', '--port', str(a.port), '--steps', str(a.steps),
               '--tire-friction', '.5', '--stop-at-route-end', '--lane-metrics', '--video']
        cmd += ['--route-id', '24785', '--target-speed-kmh', '20'] if scenario == 'speed20' else ['--route-id', '25951', '--uncapped-tcp', '--extend-route-m', '250']
        if mode == 'direct': cmd += ['--shadow-policy', 'demo/models/direct.pt', '--execute-residual']
        if mode == 'sac': cmd += ['--sac-checkpoint', 'demo/models/sac.pt', '--sac-evaluate']
        print('START', scenario, mode, flush=True)
        with (folder/(mode+'.log')).open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        session = json.loads((dest/'session.json').read_text())
        if session['cleanup_errors'] or not session['control_readback_complete']:
            raise RuntimeError('Incomplete recording: '+str(dest))
        if result.returncode and not session['collision_frames']:
            raise RuntimeError('Recording failed: '+str(dest))
        results.append(dict(scenario=scenario, mode=mode, session=session))
        (out/'sessions.json').write_text(json.dumps(results, indent=2))
        print('DONE', scenario, mode, session['reason'], session['steps'], flush=True)
    subprocess.run([sys.executable, str(CORE/'render_three_way_demo.py'), '--output', str(folder/'render'),
                    '--tcp', str(folder/'tcp'), '--direct', str(folder/'direct'), '--sac', str(folder/'sac')], cwd=ROOT, check=True)
    print('OUTPUT', out, flush=True)

if __name__ == '__main__': main()
