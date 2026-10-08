import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from world_model import DynamicsTransformer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensors(data):
    return [torch.from_numpy(data[k]) for k in ('history_normalized', 'action_normalized', 'target_normalized')]


def predict(model, data, device):
    model.eval()
    h, a, _ = tensors(data)
    with torch.no_grad():
        return torch.cat([model(h[i:i+256].to(device), a[i:i+256].to(device)).cpu()
                          for i in range(0, len(h), 256)]).numpy()


def errors(prediction, target, names):
    diff = prediction.astype(np.float64)-target
    diff[:, 7] = np.arctan2(np.sin(diff[:, 7]), np.cos(diff[:, 7]))
    return {name: dict(mae=float(np.mean(np.abs(diff[:, i]))),
                       rmse=float(np.sqrt(np.mean(diff[:, i]**2)))) for i, name in enumerate(names)}


def evaluate(checkpoint_path, data_dir, output, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    manifest = json.loads((data_dir/'manifest.json').read_text())
    if 'manifest_sha256' not in checkpoint:raise ValueError('Use a full training checkpoint')
    if checkpoint['manifest_sha256'] != hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Checkpoint/sample manifest mismatch')
    with np.load(data_dir/'validation.npz', allow_pickle=False) as src:
        data = {k: src[k] for k in src.files}
    model = DynamicsTransformer(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    normalized = predict(model, data, device)
    stats = checkpoint['normalization']['target']
    prediction = normalized*np.asarray(stats['scale'])+np.asarray(stats['mean'])
    current = data['history'][:, -1, :5]
    dt = manifest['environment']['fixed_delta_seconds']
    baseline = np.column_stack((current, current[:, 0]*dt, current[:, 1]*dt, current[:, 2]*dt))
    truth = data['target']
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    report = dict(epoch=checkpoint['epoch'], samples=len(truth), units=['m/s','m/s','rad/s','m/s2','m/s2','m','m','rad'],
                  normalized_mse=float(np.mean((normalized-data['target_normalized'])**2)),
                  model=errors(prediction, truth, manifest['target_fields']),
                  persistence_baseline=errors(baseline, truth, manifest['target_fields']),
                  note='Teacher-forced one-step evaluation')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'metrics.json', report)
    np.savez_compressed(output/'predictions.npz', prediction=prediction, target=truth,
                        baseline=baseline, frames=data['frames'], episode_id=data['episode_id'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for episode in np.unique(data['episode_id']):
        mask = data['episode_id']==episode
        time = (data['frames'][mask, 1]-data['frames'][mask, 1][0])*dt
        fig, axes = plt.subplots(4, 2, figsize=(13, 11))
        for i, ax in enumerate(axes.flat):
            ax.plot(time, truth[mask, i], label='Observed', linewidth=1)
            ax.plot(time, prediction[mask, i], label='Transformer', linewidth=1)
            ax.plot(time, baseline[mask, i], label='Persistence', alpha=.6, linewidth=.8)
            ax.set(title=manifest['target_fields'][i], xlabel='Time (s)', ylabel=report['units'][i])
            ax.grid(alpha=.2)
        axes.flat[0].legend()
        fig.tight_layout()
        tag=hashlib.sha256(str(episode).encode()).hexdigest()[:8]
        fig.savefig(output/('validation_'+tag+'.png'), dpi=130)
        plt.close(fig)
    for name in manifest['target_fields']:
        print('{} RMSE: model={:.6f} baseline={:.6f}'.format(name, report['model'][name]['rmse'], report['persistence_baseline'][name]['rmse']))
    print('EVALUATION {}'.format(output.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/prepared/world'))
    parser.add_argument('--output', type=Path, default=Path('outputs/training/world'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--checkpoint', type=Path, help='Evaluate checkpoint')
    args = parser.parse_args()
    if not (args.data/'manifest.json').is_file():
        parser.error('Missing dataset: '+str(args.data/'manifest.json'))
    if args.output.exists():
        parser.error('Output already exists')
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive training parameters required')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device=='cuda': torch.cuda.manual_seed_all(args.seed)
    manifest=json.loads((args.data/'manifest.json').read_text())
    if not manifest['validation_available']:
        parser.error('Validation episode required')
    if args.checkpoint:
        evaluate(args.checkpoint, args.data, args.output, device)
        return
    data={}
    for split in ('train','validation'):
        with np.load(args.data/(split+'.npz'), allow_pickle=False) as src:
            data[split]={k:src[k] for k in src.files}
    model=DynamicsTransformer(history_length=manifest['history_length']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(*tensors(data['train'])),batch_size=args.batch_size,shuffle=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'run_config.json',dict(device=device,epochs=args.epochs,batch_size=args.batch_size,
               lr=args.lr,seed=args.seed,model=model.config,torch_version=str(torch.__version__),data=str(args.data.resolve())))
    print('TRAIN device={} parameters={}'.format(device,sum(p.numel() for p in model.parameters())),flush=True)
    best=float('inf')
    with (args.output/'loss.csv').open('w',buffering=1) as log:
        log.write('epoch,train_normalized_mse,validation_normalized_mse\n')
        for epoch in range(1,args.epochs+1):
            model.train(); total=0
            for h,a,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=(model(h.to(device),a.to(device))-y.to(device)).square().mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step(); total+=loss.item()*len(h)
            train_loss=total/len(data['train']['action'])
            val_loss=float(np.mean((predict(model,data['validation'],device)-data['validation']['target_normalized'])**2))
            if not np.isfinite(val_loss): raise RuntimeError('Nonfinite validation loss')
            log.write('{},{},{}\n'.format(epoch,train_loss,val_loss))
            ckpt=dict(model_config=model.config,model_state=model.state_dict(),epoch=epoch,
                      normalization=manifest['normalization'],validation_loss=val_loss,
                      manifest_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest())
            if val_loss < best:
                best=val_loss; torch.save(ckpt,args.output/'best.pt')
            if epoch==args.epochs: torch.save(ckpt,args.output/'last.pt')
            if epoch==1 or epoch%10==0 or epoch==args.epochs:
                print('epoch={} train_mse={:.6f} val_mse={:.6f}'.format(epoch,train_loss,val_loss),flush=True)
    evaluate(args.output/'best.pt',args.data,args.output/'evaluation',device)


if __name__=='__main__':
    main()
