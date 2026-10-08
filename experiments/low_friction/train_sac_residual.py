import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from sac_core import SAC, Replay

SCENARIOS = [dict(name='speed20',route='24785',flags=['--target-speed-kmh','20']),
             dict(name='tcp_speed',route='25951',flags=['--uncapped-tcp','--extend-route-m','250'])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--episodes',type=int,default=2)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--weather',choices=('ClearNoon','MidRainyNoon','HardRainNoon'),default=None,help='Weather for fresh training')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--learning-starts',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--update-every',type=int,default=4)
    p.add_argument('--capacity',type=int,default=100000)
    p.add_argument('--normalization-checkpoint',type=Path,default=Path('demo/models/world.pt'))
    p.add_argument('--collision-penalty',type=float,default=300.)
    p.add_argument('--world-model',type=Path,help='Enable world-model replay')
    p.add_argument('--model-ratio',type=float,default=.25,help='Synthetic batch fraction')
    args = p.parse_args()
    if not np.isfinite(args.collision_penalty) or args.collision_penalty<=0:p.error('Positive collision penalty required')
    if args.output.exists():p.error('Choose a new output directory')
    if args.episodes<1 or not 41<=args.steps<=1200:p.error('Invalid episode or step count')
    if min(args.batch_size,args.learning_starts,args.update_every,args.capacity)<1 or args.batch_size>args.capacity:
        p.error('Invalid replay/update configuration')
    if args.evaluate and not args.resume:p.error('Evaluation requires --resume')
    if not 0<args.model_ratio<=.5:p.error('Model ratio: (0,0.5]')
    if args.world_model and (not args.world_model.is_file() or args.evaluate):p.error('Training world model required')
    root = Path(__file__).resolve().parents[2]
    os.chdir(root); torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True)
    if args.resume:
        sac = SAC.load(args.resume)
        if getattr(sac,"inference_only",False):p.error('Demo export is inference-only')
        if args.weather is not None and args.weather != sac.config.get('weather','ClearNoon'):p.error('Resume inherits weather')
    else:
        manifest = args.normalization_checkpoint.resolve()
        stats=torch.load(manifest,map_location='cpu',weights_only=True)['normalization']
        config = dict(normalization=stats,collision_penalty=args.collision_penalty,weather=args.weather or 'ClearNoon',
                      normalization_source=str(manifest),normalization_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      learning_rate=3e-4,gamma=.99,tau=.005,alpha=.02,
                      batch_size=args.batch_size,learning_starts=args.learning_starts,update_every=args.update_every,
                      capacity=args.capacity,seed=args.seed,episodes_completed=0,
                      observation_dim=287,action_limits=[.05,.02],fixed_delta_seconds=.05,
                      tire_friction=.5,initialization='independent zero-mean actor, log_std=-2',
                      reward_version='sac_online_v1',training_scope='two named demonstration routes')
        sac = SAC(config)
    if args.world_model:
        world_path=args.world_model.resolve();world_hash=hashlib.sha256(world_path.read_bytes()).hexdigest()
        if sac.config.get('world_model_sha256',world_hash)!=world_hash:p.error('World model mismatch')
        sac.config.update(world_model=str(world_path),world_model_sha256=world_hash,model_ratio=args.model_ratio,
                          model_capacity=4096,model_horizon=1,model_contract='Held TCP; fixed reference')
        if sac.model_replay is None:sac.model_replay=Replay(sac.config['model_capacity'])
    initial_steps, initial_updates = sac.steps,sac.updates
    start_index = sac.config.get('episodes_completed',0)
    checkpoint = args.output/'initial.pt'; sac.save(checkpoint)
    config = dict(sac.config,cli=vars(args),scenarios=SCENARIOS)
    (args.output/'config.json').write_text(json.dumps(config,indent=2,default=str))
    source = args.output/'source'; source.mkdir()
    for name in ['sac_core.py','sac_online.py','sac_world.py','train_sac_residual.py','record_tcp_reference.py','tcp_reference_agent.py']:
        (source/name).write_bytes((root/'experiments/low_friction'/name).read_bytes())
    frozen = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [root/'demo/models/tcp.ckpt.gz',root/'demo/models/direct.pt',
               root/'demo/models/world.pt',root/'team_code/tcp_b2d_agent.py']}
    (args.output/'preserved_sha256.json').write_text(json.dumps(frozen,indent=2))
    del sac
    reports = []; started = time.monotonic(); status = 'running'
    try:
        for i in range(args.episodes):
            scenario = SCENARIOS[(start_index+i)%len(SCENARIOS)]
            episode = args.output/('episode_{:03d}_{}'.format(i,scenario['name']))
            log = args.output/('episode_{:03d}.log'.format(i))
            cmd = [sys.executable,'-u','experiments/low_friction/record_tcp_reference.py',
                   '--output',str(episode),'--sac-checkpoint',str(checkpoint),'--port',str(args.port),
                   '--steps',str(args.steps),'--route-id',scenario['route'],'--tire-friction','.5',
                   '--stop-at-route-end','--weather',config.get('weather','ClearNoon')]+scenario['flags']
            if args.evaluate:cmd.append('--sac-evaluate')
            print('START {} log={}'.format(scenario['name'],log.resolve()),flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
                try: code = proc.wait()
                except KeyboardInterrupt:
                    if proc.poll() is None:proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=60)
                    if (episode/'sac_last.pt').exists():checkpoint = episode/'sac_last.pt'
                    raise
            if (episode/'sac_last.pt').exists():checkpoint = episode/'sac_last.pt'
            if code!=0:raise RuntimeError('Episode failed; inspect '+str(log))
            report = json.loads((episode/'sac_summary.json').read_text())
            report.update(scenario=scenario['name'],directory=str(episode.resolve()))
            reports.append(report)
            print('DONE '+json.dumps(report),flush=True)
            if report['reason'] not in ('step_limit','collision','route_complete'):
                raise RuntimeError('Episode interrupted: '+report['reason'])
            session = json.loads((episode/'session.json').read_text())
            if session['cleanup_errors'] or not session['control_readback_complete']:
                raise RuntimeError('Cleanup/readback failed')
        status = 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception:
        status = 'failed'
        raise
    finally:
        same = all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in frozen.items())
        state = SAC.load(checkpoint)
        summary = dict(status=status,episodes=reports,last_checkpoint=str(checkpoint.resolve()),
                       steps_before=initial_steps,steps_after=state.steps,updates_before=initial_updates,
                       updates_after=state.updates,preserved_models_unchanged=same,
                       wall_seconds=time.monotonic()-started,evaluation=args.evaluate)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('OUTPUT '+str(args.output.resolve()),flush=True)
        if not same:raise RuntimeError('Protected baseline changed')
    if status=='completed':print('SAC training complete',flush=True)


if __name__=='__main__':main()
