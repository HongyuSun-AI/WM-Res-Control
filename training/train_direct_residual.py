import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/low_friction'))
from world_model import DynamicsTransformer
from residual_policy import ResidualPolicy
from residual_rollout import policy_rollout
from lane_objective import lane_combined_objective


def objective(model,policy,batch,stats,lane_weight=20.):
    inputs=tuple(torch.cat([c['inputs'][i] for c in batch],dim=0) for i in range(5))
    kw={k:[v for c in batch for v in c['kwargs'][k]] for k in batch[0]['kwargs']}
    r=policy_rollout(model,policy,*inputs,stats,lane_geometries=[b['lane_sides'] for b in batch])
    return lane_combined_objective(r['prediction'],lane_left_segments=[b['lane_sides'][0] for b in batch],
        lane_right_segments=[b['lane_sides'][1] for b in batch],footprint_corners=[b['corners'] for b in batch],
        lane_frame=[b['frame'] for b in batch],lane_weight=lane_weight,low_speed_mode='regularized',**kw)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/prepared/direct'));p.add_argument('--output',type=Path,default=Path('outputs/training/direct'))
    p.add_argument('--world',type=Path,default=Path('demo/models/world.pt'));p.add_argument('--init-policy',type=Path)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--lane-weight',type=float,default=20.);p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.output.exists() or min(a.epochs,a.batch_size)<1 or not 0<a.lr<1 or not 0<=a.lane_weight<1e6:p.error('Invalid output or hyperparameters')
    if not (a.data/'windows.pt').is_file():p.error('Missing dataset: '+str(a.data/'windows.pt'))
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(a.seed)
    d=torch.load(a.data/'windows.pt',map_location='cpu',weights_only=True);ck=torch.load(a.world,map_location='cpu',weights_only=True)
    if d['format']!='direct_windows_v1' or d['normalization']!=ck['normalization']:raise ValueError('Dataset/world mismatch')
    train,val=d['splits']['train'],d['splits']['validation']
    if not train or not val:raise ValueError('Empty split')
    for k in ('episode_id','route_id'):
        if {b[k] for b in train}&{b[k] for b in val}:raise ValueError('Split leakage: '+k)
    model=DynamicsTransformer(**ck['model_config']).double().eval().requires_grad_(False);model.load_state_dict(ck['model_state'])
    policy=ResidualPolicy(lane_input=True).double().eval()
    with torch.no_grad():policy.limits.copy_(torch.tensor([.05,.02],dtype=torch.float64))
    if a.init_policy:
        old=torch.load(a.init_policy,map_location='cpu',weights_only=True)
        if old['normalization']!=ck['normalization']:raise ValueError('Initial policy normalization mismatch')
        policy.load_state_dict(old['policy_state'])
    opt=torch.optim.Adam(policy.parameters(),lr=a.lr);a.output.mkdir(parents=True)
    sources={'world_sha256':hashlib.sha256(a.world.read_bytes()).hexdigest(),
             'data_sha256':hashlib.sha256((a.data/'windows.pt').read_bytes()).hexdigest()}
    history=[];best=float('inf')
    def evaluate():
        values=[];invalid=Counter()
        with torch.no_grad():
            for c in val:
                try:
                    loss=objective(model,policy,[c],ck['normalization'],a.lane_weight)
                    if not torch.isfinite(loss['total']):raise ValueError('Nonfinite validation loss')
                    values.append(float(loss['total']))
                except (ValueError,RuntimeError) as e:invalid[str(e)]+=1
        return dict(valid=len(values),planned=len(val),invalid=dict(invalid),total=sum(values)/len(values) if values else None)
    for epoch in range(a.epochs+1):
        accepted=0;skipped=Counter()
        if epoch:
            order=torch.randperm(len(train)).tolist()
            for start in range(0,len(order),a.batch_size):
                batch=[train[i] for i in order[start:start+a.batch_size]];opt.zero_grad()
                try:
                    loss=objective(model,policy,batch,ck['normalization'],a.lane_weight)
                    if not torch.isfinite(loss['total']):raise ValueError('Nonfinite training loss')
                    loss['total'].backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                except (ValueError,RuntimeError) as e:skipped[str(e)]+=len(batch);opt.zero_grad();continue
                opt.step();accepted+=len(batch)
            if not accepted:raise RuntimeError('No valid training updates: '+str(dict(skipped)))
        metrics=evaluate();history.append(dict(epoch=epoch,accepted_train=accepted,skipped=dict(skipped),validation=metrics))
        saved=dict(format='residual_policy_lane_v1',policy_state=policy.state_dict(),normalization=ck['normalization'],epoch=epoch,
                   fixed_map='Town04',fixed_lane_id=None,lane_reference_mode='planned_route',action_limits=policy.limits.tolist(),
                   sources=sources,config=vars(a),validation_total=metrics['total'])
        saved['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        torch.save(saved,a.output/'last.pt')
        if metrics['valid']==metrics['planned'] and metrics['total']<best:best=metrics['total'];torch.save(saved,a.output/'best.pt')
        (a.output/'progress.json').write_text(json.dumps(history,indent=2));print('EPOCH',epoch,json.dumps(metrics),flush=True)
    print('PASS: direct training; world frozen')


if __name__=='__main__':main()
