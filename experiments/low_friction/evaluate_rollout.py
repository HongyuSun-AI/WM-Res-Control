import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from build_samples import construct
from inspect_recording import require, validate
from train_world_model import predict, write_json
from world_model import DynamicsTransformer


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def window_indices(data, horizon):
    starts = []
    for start in range(len(data['action'])-horizon+1):
        end = start+horizon
        if (np.all(data['episode_id'][start:end] == data['episode_id'][start])
                and np.all(np.diff(data['step'][start:end]) == 1)
                and np.all(data['frames'][start:end, 1] == data['frames'][start:end, 0]+1)
                and np.array_equal(data['frames'][start:end-1, 1], data['frames'][start+1:end, 0])
                and not np.any(data['terminated'][start:end-1] | data['truncated'][start:end-1])):
            starts.append(start)
    require(bool(starts), 'Continuous windows unavailable')
    return np.asarray(starts)[:, None]+np.arange(horizon)[None, :]


def load_validation(directory, manifest):
    with np.load(directory/'validation.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    offset = 0
    for source in manifest['sources']:
        if source['split'] != 'validation':
            continue
        path = Path(source['path'])
        if not path.is_absolute(): path = directory/path
        require(hashlib.sha256((path/'transitions.jsonl').read_bytes()).hexdigest() == source['sha256'],
                'Validation source changed')
        metadata, rows = validate(path)
        require(metadata['episode_id'] == source['episode_id'], 'Episode id changed')
        expected = construct(rows, manifest['history_length'])
        n = len(expected['action'])
        require(n == source['samples'], 'Source sample count mismatch')
        for key, value in expected.items():
            require(np.array_equal(data[key][offset:offset+n], value), 'Source alignment failed: '+key)
        require(np.all(data['episode_id'][offset:offset+n] == source['episode_id']), 'Episode mismatch')
        offset += n
    require(offset == len(data['action']) and offset > 0, 'Unexpected validation sample count')
    stats = manifest['normalization']
    for name in ('history', 'action', 'target'):
        mean = stats['state']['mean']+stats['action']['mean'] if name == 'history' else stats[name]['mean']
        scale = stats['state']['scale']+stats['action']['scale'] if name == 'history' else stats[name]['scale']
        expected = ((data[name]-np.asarray(mean))/np.asarray(scale)).astype(np.float32)
        require(np.isfinite(expected).all() and np.allclose(data[name+'_normalized'], expected, atol=1e-6),
                'Normalization mismatch: '+name)
    print('PASS: validation alignment and normalization', flush=True)
    return data


def rollout(model, history, actions, stats, device, batch_size=256):
    hm = np.asarray(stats['state']['mean']+stats['action']['mean'])
    hs = np.asarray(stats['state']['scale']+stats['action']['scale'])
    am, asc = np.asarray(stats['action']['mean']), np.asarray(stats['action']['scale'])
    tm, ts = np.asarray(stats['target']['mean']), np.asarray(stats['target']['scale'])
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            h = history[start:start+batch_size].astype(np.float64).copy()
            future = actions[start:start+batch_size]
            predictions = []
            for k in range(future.shape[1]):
                a = future[:, k]
                ht = torch.as_tensor(((h-hm)/hs).astype(np.float32), device=device)
                at = torch.as_tensor(((a-am)/asc).astype(np.float32), device=device)
                p = model(ht, at).cpu().numpy()*ts+tm
                require(np.isfinite(p).all(), 'Nonfinite rollout prediction')
                predictions.append(p)
                h = np.concatenate((h[:, 1:], np.concatenate((p[:, :5], a), axis=1)[:, None]), axis=1)
            result.append(np.stack(predictions, axis=1))
    return np.concatenate(result)


def integrate(values):
    pose = np.zeros((len(values), 3), dtype=np.float64)
    poses = []
    for k in range(values.shape[1]):
        dx, dy, dyaw = values[:, k, 5:].T
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        pose[:, 0] += c*dx-s*dy
        pose[:, 1] += s*dx+c*dy
        pose[:, 2] = wrap(pose[:, 2]+dyaw)
        poses.append(pose.copy())
    return np.stack(poses, axis=1)


def scalar_error(diff):
    if not len(diff):
        return dict(count=0, mae=None, rmse=None)
    return dict(count=len(diff), mae=float(np.mean(np.abs(diff))), rmse=float(np.sqrt(np.mean(diff**2))))


def metrics(prediction, truth, poses, truth_poses):
    result = []
    for k in range(truth.shape[1]):
        diff = prediction[:, k]-truth[:, k]
        row = {name: scalar_error(diff[:, i]) for i, name in enumerate(('vx','vy','r','ax','ay'))}
        distance = np.linalg.norm(poses[:, :k+1, :2]-truth_poses[:, :k+1, :2], axis=-1)
        row['position'] = scalar_error(distance[:, -1])
        row['ade_m'] = float(distance.mean())
        row['yaw_deg'] = scalar_error(np.rad2deg(wrap(poses[:, k, 2]-truth_poses[:, k, 2])))
        beta_true = np.arctan2(truth[:, k, 1], truth[:, k, 0])
        beta_pred = np.arctan2(prediction[:, k, 1], prediction[:, k, 0])
        valid = truth[:, k, 0] >= .5
        slip = (truth[:, k, 0] >= 2) & (np.abs(beta_true) > np.deg2rad(5))
        beta_error = np.rad2deg(wrap(beta_pred-beta_true))
        row['beta_deg'] = scalar_error(beta_error[valid])
        row['beta_gt5deg_vx_ge2'] = scalar_error(beta_error[slip])
        row['vx_beta_gt5deg_vx_ge2'] = scalar_error(diff[slip, 0])
        row['predicted_vx_below_threshold_on_valid_truth'] = int(np.sum(valid & (prediction[:, k, 0] < .5)))
        result.append(row)
    return result


def plot_report(output, report, truth_pose, model_pose, baseline_pose):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time = np.arange(1, report['horizon']+1)*report['dt']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key, unit in zip(axes.flat, ('vx','vy','r','position','yaw_deg','beta_deg'),
                             ('m/s','m/s','rad/s','m','deg','deg')):
        for field, label in (('model','Transformer'), ('persistence_baseline','Persistence')):
            ax.plot(time, [row[key]['rmse'] for row in report[field]], marker='.', label=label)
        ax.set(xlabel='Horizon (s)', ylabel='RMSE ('+unit+')', title=key)
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output/'error_by_horizon.png', dpi=130)
    plt.close(fig)
    worst = int(np.argmax(np.linalg.norm(model_pose[:, -1, :2]-truth_pose[:, -1, :2], axis=1)))
    fig, ax = plt.subplots(figsize=(6, 6))
    for pose, label in ((truth_pose,'Observed'), (model_pose,'Transformer'), (baseline_pose,'Persistence')):
        path = np.vstack((np.zeros((1, 3)), pose[worst]))
        ax.plot(path[:, 0], path[:, 1], '.-', label=label)
    ax.set(xlabel='Start-frame forward (m)', ylabel='Start-frame right (m)',
           title='Worst endpoint: window {}'.format(worst))
    ax.axis('equal'); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/'worst_trajectory.png', dpi=130); plt.close(fig)
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists() or args.horizon < 1:
        parser.error('Invalid output or horizon')
    torch.set_num_threads(2)
    manifest_bytes = (args.data/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    require('manifest_sha256' in checkpoint, 'Full training checkpoint required')
    require(checkpoint['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest(), 'Checkpoint manifest mismatch')
    require(checkpoint['normalization'] == manifest['normalization'], 'Checkpoint normalization mismatch')
    require(checkpoint['model_config']['history_length'] == manifest['history_length'], 'History length mismatch')
    data = load_validation(args.data, manifest)
    indices = window_indices(data, args.horizon)
    starts = indices[:, 0]
    dt = manifest['environment']['fixed_delta_seconds']
    model = DynamicsTransformer(**checkpoint['model_config']).to(args.device)
    model.load_state_dict(checkpoint['model_state'])
    prediction = rollout(model, data['history'][starts], data['action'][indices], checkpoint['normalization'], args.device)
    normalized = predict(model, data, args.device)[starts]
    stats = checkpoint['normalization']['target']
    first_step = normalized*np.asarray(stats['scale'])+np.asarray(stats['mean'])
    require(np.allclose(prediction[:, 0], first_step, atol=1e-5, rtol=1e-5), 'First-step regression check failed')
    truth = data['target'][indices].astype(np.float64)
    initial = data['history'][starts, -1, :5].astype(np.float64)
    baseline_step = np.column_stack((initial, initial[:, 0]*dt, initial[:, 1]*dt, initial[:, 2]*dt))
    baseline = np.repeat(baseline_step[:, None], args.horizon, axis=1)
    truth_pose, model_pose, base_pose = [integrate(x) for x in (truth, prediction, baseline)]
    report = dict(epoch=checkpoint['epoch'], horizon=args.horizon, dt=dt, windows=len(starts),
                  episodes=len(np.unique(data['episode_id'][starts])), device=args.device,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  manifest_sha256=checkpoint['manifest_sha256'],
                  model=metrics(prediction, truth, model_pose, truth_pose),
                  persistence_baseline=metrics(baseline, truth, base_pose, truth_pose),
                  note='Autoregressive validation with recorded actions')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'predictions.npz', prediction=prediction, target=truth, baseline=baseline,
                        model_pose=model_pose, target_pose=truth_pose, baseline_pose=base_pose,
                        frames=data['frames'][indices], episode_id=data['episode_id'][starts],
                        start_step=data['step'][starts], actions=data['action'][indices])
    report['worst_trajectory_window'] = plot_report(args.output, report, truth_pose, model_pose, base_pose)
    write_json(args.output/'metrics.json', report)
    print('PASS: rollout windows={} horizon={} time={:.2f}s'.format(len(starts), args.horizon, dt*args.horizon))
    for k in sorted(set((0, min(4, args.horizon-1), args.horizon-1))):
        print('HORIZON step={} time={:.2f}s'.format(k+1, (k+1)*dt))
        for key in ('vx','vy','r','position','yaw_deg','beta_deg'):
            print('  {} RMSE model={} baseline={}'.format(key, report['model'][k][key]['rmse'], report['persistence_baseline'][k][key]['rmse']))
    print('EVALUATION {}'.format(args.output.resolve()))


if __name__ == '__main__':
    main()
