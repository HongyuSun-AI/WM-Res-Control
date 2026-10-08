import torch
from residual_policy import compose_action


def to_vehicle(points,pose,vectors=False):
    extra=points.ndim-2
    c=pose[:,2].cos().reshape(-1,*([1]*extra))
    s=pose[:,2].sin().reshape(-1,*([1]*extra))
    p=points if vectors else points-pose[:,:2].reshape(-1,*([1]*extra),2)
    return torch.stack((c*p[...,0]+s*p[...,1],-s*p[...,0]+c*p[...,1]),dim=-1)


def policy_rollout(model,policy,history,base_action,waypoint,boundary,mask,stats,lane_geometries=None):
    'history[B,10,7];action[B,2];waypoint[B,2];boundary[B,21,2,4];mask[B,21,2]'
    batch=len(history)
    def norm(name):
        mean=history.new_tensor(stats[name]['mean']);scale=history.new_tensor(stats[name]['scale'])
        return mean,scale
    sm,ss=norm('state');am,asc=norm('action');tm,ts=norm('target')
    hm=torch.cat((sm,am));hs=torch.cat((ss,asc))
    h=history;pose=history.new_zeros((batch,3))
    saved={k:[] for k in ('features','history','poses_before','poses_after','prediction','residual','actions','effective_residual')}
    for _ in range(10):
        points=to_vehicle(boundary[...,:2],pose)
        tangent=to_vehicle(boundary[...,2:],pose,vectors=True)
        geometry=torch.cat((points/30.,tangent),dim=-1)
        geometry=torch.where(mask[...,None],geometry,torch.zeros_like(geometry))
        features=torch.cat((((h-hm)/hs).reshape(batch,70),(base_action-am)/asc,
                            to_vehicle(waypoint,pose)/30.,geometry.reshape(batch,168),mask.reshape(batch,42).to(h.dtype)),dim=-1)
        if getattr(policy,'lane_input',False):
            from lane_features import lane_features
            if lane_geometries is None:raise ValueError('Lane geometry required')
            features=torch.cat((features,lane_features(pose,lane_geometries)),dim=-1)
        residual=policy(features)
        controls=compose_action(base_action,residual,limits=policy.limits);action=controls['action']
        p=model((h-hm)/hs,(action-am)/asc)*ts+tm
        saved['poses_before'].append(pose);saved['features'].append(features);saved['history'].append(h)
        c,s=pose[:,2].cos(),pose[:,2].sin()
        pose=torch.stack((pose[:,0]+c*p[:,5]-s*p[:,6],pose[:,1]+s*p[:,5]+c*p[:,6],pose[:,2]+p[:,7]),dim=-1)
        saved['poses_after'].append(pose);saved['prediction'].append(p);saved['residual'].append(residual)
        saved['actions'].append(action);saved['effective_residual'].append(controls['effective_residual'])
        h=torch.cat((h[:,1:],torch.cat((p[:,:5],action),dim=-1)[:,None]),dim=1)
    return {k:torch.stack(v,dim=1) for k,v in saved.items()}
