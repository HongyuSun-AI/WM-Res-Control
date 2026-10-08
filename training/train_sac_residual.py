import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/low_friction'))
from sac_core import SAC, Replay, DIM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/prepared/sac'))
    p.add_argument('--output', type=Path, default=Path('outputs/training/sac'))
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--model-ratio', type=float, default=.25)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output directory')
    if min(a.updates, a.batch_size) < 1 or not 0 < a.lr < 1 or not 0 <= a.model_ratio < 1:
        p.error('Invalid training parameters')
    if round(a.batch_size*a.model_ratio) >= a.batch_size:
        p.error('Real samples required')
    path = a.data/'replay.pt' if a.data.is_dir() else a.data
    if not path.is_file(): p.error('Missing dataset: '+str(path))
    torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['format'] != 'sac_replay_dataset_v1': raise ValueError('Unsupported replay dataset')
    config = dict(data['config'], capacity=data['replay']['capacity'],
                  learning_rate=a.lr, batch_size=a.batch_size, gamma=.99, tau=.005,
                  model_ratio=a.model_ratio, training_mode='offline_fixed_replay',
                  world_assisted_policy=a.model_ratio > 0,
                  normalization_source='downloaded replay dataset',
                  source_world_sha256=data['world_sha256'])
    config.setdefault('alpha', .1)
    sac = SAC(config)
    sac.replay.restore(data['replay'])
    if a.model_ratio > 0:
        sac.model_replay = Replay(data['model_replay']['capacity'])
        sac.model_replay.restore(data['model_replay'])
    a.output.mkdir(parents=True)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(config=config, seed=a.seed, data_sha256=source_hash,
                    real_transitions=sac.replay.size,
                    synthetic_transitions=sac.model_replay.size if sac.model_replay else 0,
                    new_carla_steps=0, new_model_steps=0)
    (a.output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    started = time.monotonic()
    with (a.output/'progress.jsonl').open('w') as log:
        for step in range(1, a.updates+1):
            metrics = sac.update()
            if step == 1 or step % 100 == 0 or step == a.updates:
                metrics['seconds'] = time.monotonic()-started
                log.write(json.dumps(metrics)+'\n'); log.flush()
                print('UPDATE', step, 'q_loss={:.5f} actor_loss={:.5f}'.format(metrics['q_loss'], metrics['actor_loss']), flush=True)
    torch.save(dict(format='sac_inference_v1', actor=sac.actor.state_dict(),
                    config=config, steps=0, updates=sac.updates, model_steps=0), a.output/'actor.pt')
    training = dict(format='sac_offline_training_v1', config=config,
                    updates=sac.updates, log_alpha=sac.log_alpha.detach(),
                    data_sha256=source_hash, torch_rng=torch.get_rng_state())
    for key in ('actor', 'q1', 'q2', 'target1', 'target2', 'actor_opt', 'q_opt', 'alpha_opt'):
        training[key] = getattr(sac, key).state_dict()
    torch.save(training, a.output/'training.pt')
    (a.output/'summary.json').write_text(json.dumps(dict(metadata, updates=sac.updates,
        seconds=time.monotonic()-started, actor='actor.pt', metrics=metrics), indent=2)+'\n')
    print('PASS: offline SAC training')
    print('OUTPUT', a.output.resolve())


if __name__ == '__main__': main()
