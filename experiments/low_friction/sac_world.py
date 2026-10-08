import copy
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from world_model import DynamicsTransformer
from residual_rollout import to_vehicle
from lane_features import lane_features
from sac_core import LIMITS


def next_features(inputs, sides, prediction, action, stats):
    t=lambda v:torch.as_tensor(v,dtype=torch.float64)
    p=t(prediction);pose=p[5:8][None]
    history=t(inputs['history']);history=torch.cat((history[1:],torch.cat((p[:5],t(action)))[None]),0)
    hm=t(stats['state']['mean']+stats['action']['mean']);hs=t(stats['state']['scale']+stats['action']['scale'])
    am=t(stats['action']['mean']);asc=t(stats['action']['scale'])
    boundary=t(inputs['boundary_geometry'])[None];mask=torch.as_tensor(inputs['boundary_mask'])
    points=to_vehicle(boundary[...,:2],pose)[0]/30.
    tangent=to_vehicle(boundary[...,2:],pose,vectors=True)[0]
    geometry=torch.where(mask[...,None],torch.cat((points,tangent),-1),torch.zeros_like(boundary[0]))
    waypoint=to_vehicle(t(inputs['waypoint_actor'])[None],pose)[0]/30.
    lane=lane_features(pose,[sides])[0]
    features=torch.cat((((history-hm)/hs).flatten(),(t(inputs['base_action'])-am)/asc,
                        waypoint,geometry.flatten(),mask.flatten().double(),lane))
    if features.shape!=(287,) or not torch.isfinite(features).all():raise ValueError('Nonfinite predicted observation')
    return features.numpy().astype(np.float32)


def predicted_state(state,prediction):
    p=np.asarray(prediction);c,s=math.cos(state['yaw']),math.sin(state['yaw'])
    new=dict(state)
    new.update(x=state['x']+c*p[5]-s*p[6],y=state['y']+s*p[5]+c*p[6],
               yaw=math.atan2(math.sin(state['yaw']+p[7]),math.cos(state['yaw']+p[7])),
               frame=state['frame']+1,time=state['time']+.05)
    new.update(zip(('vx','vy','r','ax','ay'),map(float,p[:5])))
    new.update(beta=math.atan2(new['vy'],new['vx']),beta_valid=new['vx']>=.5)
    c,s=math.cos(new['yaw']),math.sin(new['yaw'])
    new.update(world_vx=c*new['vx']-s*new['vy'],world_vy=s*new['vx']+c*new['vy'])
    return new


class WorldBranches:
    def __init__(self,config):
        path=Path(config['world_model']);self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256!=config['world_model_sha256']:raise ValueError('World checkpoint changed')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        self.stats=ck['normalization']
        if self.stats!=config['normalization']:raise ValueError('World/SAC normalization mismatch')
        with torch.random.fork_rng(devices=[]):
            self.model=DynamicsTransformer(**ck['model_config']).double().eval().requires_grad_(False)
        self.model.load_state_dict(ck['model_state'])
        self.initial={k:v.clone() for k,v in self.model.state_dict().items()}

    def unchanged(self):
        return not self.model.training and all(not p.requires_grad and p.grad is None for p in self.model.parameters()) and all(torch.equal(v,self.initial[k]) for k,v in self.model.state_dict().items())

    def sample(self,sac,context,obs,state,last_action,lane_monitor,progress_plan,progress_high,reward_fn):
        inputs,sides=context
        action=sac.act(obs)
        actual=np.clip(inputs['base_action']+action*LIMITS,-1,1)
        t=lambda v:torch.as_tensor(v,dtype=torch.float64)
        hm=t(self.stats['state']['mean']+self.stats['action']['mean']);hs=t(self.stats['state']['scale']+self.stats['action']['scale'])
        am=t(self.stats['action']['mean']);asc=t(self.stats['action']['scale'])
        with torch.inference_mode():
            p=(self.model(((t(inputs['history'])-hm)/hs)[None],((t(actual)-am)/asc)[None])[0]
               *t(self.stats['target']['scale'])+t(self.stats['target']['mean'])).numpy()
        if not np.isfinite(p).all():raise ValueError('Nonfinite world prediction')
        if not (.5<=p[0]<=12 and abs(p[1])<=6 and abs(p[2])<=2 and
                np.linalg.norm(p[5:7])<=1 and abs(p[7])<=.15):
            raise ValueError('prediction_domain')
        new=predicted_state(state,p)
        lane=copy.deepcopy(lane_monitor).observe(new)
        if not lane.get('covered'):raise ValueError('predicted_lane_coverage')
        plan=copy.deepcopy(progress_plan)
        remaining=plan.observe(new)['remaining_route_m']
        if remaining<10:raise ValueError('near_goal')
        progress=max(0.,float(plan.arc[-1]-remaining-progress_high))
        with torch.inference_mode():nxt=next_features(inputs,sides,p,actual,self.stats)
        terms=reward_fn(state,new,progress,lane,action,last_action,inputs['waypoint_actor'],collision=False,goal=False)
        transition=dict(obs=obs,action=action,reward=sum(terms.values()),next_obs=nxt,
                        terminated=False,truncated=True,gate=True,next_gate=True)
        diagnostic=dict(seed_frame=state['frame'],source='world_model',horizon=1,held_TCP_base=inputs['base_action'].tolist(),
                        action=action.tolist(),physical_action=actual.tolist(),prediction=p.tolist(),
                        reward_terms=terms,lane=lane,terminated=False,truncated=True,
                        truncation_reason='model_horizon_bootstrap',collision_label='unknown_not_predicted')
        return transition,diagnostic
