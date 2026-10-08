import copy
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIM = 287
LIMITS = np.array([.05, .02], dtype=np.float32)


def mlp(n, out):
    return nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(DIM, 4)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad(): self.net[-1].bias[2:].fill_(-2.)

    def sample(self, obs, deterministic=False):
        mean, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(-5., 1.)
        normal = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else normal.rsample()
        action = z.tanh()
        log_prob = (normal.log_prob(z) - 2*(math.log(2)-z-F.softplus(-2*z))).sum(-1, keepdim=True)
        return action, log_prob


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = self.size = 0
        self.arrays = {k: np.empty((capacity, n), np.float32) for k,n in
                       [('obs',DIM),('action',2),('reward',1),('next_obs',DIM),
                        ('terminated',1),('truncated',1),('gate',1),('next_gate',1)]}

    def add(self, **transition):
        if set(transition) != set(self.arrays): raise ValueError('Replay fields mismatch')
        for key, value in transition.items():
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            if value.shape != self.arrays[key].shape[1:] or not np.isfinite(value).all():
                raise ValueError('Invalid replay field '+key)
            self.arrays[key][self.pos] = value
        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def batch(self, n):
        indices = np.random.randint(self.size, size=n)
        return {k:torch.from_numpy(v[indices]) for k,v in self.arrays.items()}

    def state(self):
        return dict(pos=self.pos,size=self.size,capacity=self.capacity,
                    arrays={k:torch.from_numpy(v[:self.size].copy()) for k,v in self.arrays.items()})

    def restore(self, state):
        if state['capacity'] != self.capacity: raise ValueError('Replay capacity mismatch')
        self.pos, self.size = state['pos'], state['size']
        for k,v in state['arrays'].items(): self.arrays[k][:self.size] = v.numpy()


class SAC:
    def __init__(self, config):
        self.config = config
        self.actor = Actor()
        self.q1, self.q2 = mlp(DIM+2, 1), mlp(DIM+2, 1)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target1.requires_grad_(False); self.target2.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config['alpha'])))
        lr = config['learning_rate']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.replay = Replay(config['capacity'])
        self.model_replay = Replay(config.get('model_capacity',4096)) if config.get('world_model') else None
        self.model_steps = 0
        self.steps = self.updates = 0

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a,_ = self.actor.sample(torch.as_tensor(obs,dtype=torch.float32)[None], deterministic)
        return a[0].numpy()

    def update(self):
        size = self.config['batch_size']
        model_count = round(size*self.config.get('model_ratio',0.)) if self.model_replay is not None else 0
        if self.model_replay is None or self.model_replay.size<model_count:model_count=0
        if not 0<=model_count<size:raise ValueError('Real samples required')
        b = self.replay.batch(size-model_count)
        if model_count:
            predicted = self.model_replay.batch(model_count)
            b = {k:torch.cat((v,predicted[k]),dim=0) for k,v in b.items()}
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a,lp = self.actor.sample(b['next_obs'])
            a = a*b['next_gate']; lp = lp*b['next_gate']
            x = torch.cat([b['next_obs'],a],-1)
            target = b['reward'] + self.config['gamma']*(1-b['terminated'])*(torch.minimum(self.target1(x),self.target2(x))-alpha*lp)
        x = torch.cat([b['obs'],b['action']],-1)
        q_loss = F.mse_loss(self.q1(x),target)+F.mse_loss(self.q2(x),target)
        if not torch.isfinite(q_loss): raise RuntimeError('Nonfinite critic loss')
        self.q_opt.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),10.)
        self.q_opt.step()
        gate = b['gate'][:,0].bool()
        actor_loss = torch.tensor(0.); alpha_loss = torch.tensor(0.)
        if gate.any():
            self.q1.requires_grad_(False); self.q2.requires_grad_(False)
            a,lp = self.actor.sample(b['obs'][gate])
            x = torch.cat([b['obs'][gate],a],-1)
            actor_loss = (alpha*lp-torch.minimum(self.q1(x),self.q2(x))).mean()
            if not torch.isfinite(actor_loss): raise RuntimeError('Nonfinite actor loss')
            self.actor_opt.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),10.)
            self.actor_opt.step()
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
            alpha_loss = -(self.log_alpha*(lp.detach()-2.)).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            for net,target_net in [(self.q1,self.target1),(self.q2,self.target2)]:
                for p,t in zip(net.parameters(),target_net.parameters()): t.lerp_(p,self.config['tau'])
        self.updates += 1
        return dict(q_loss=float(q_loss),actor_loss=float(actor_loss),alpha=float(self.log_alpha.exp()),updates=self.updates,
                    real_samples=size-model_count,model_samples=model_count)

    def save(self, path):
        state = dict(format='sac_world_residual_v1' if self.model_replay is not None else 'sac_residual_v1',config=self.config,steps=self.steps,updates=self.updates,
                     replay=self.replay.state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                     log_alpha=self.log_alpha.detach())
        if self.model_replay is not None:
            state.update(model_replay=self.model_replay.state(),model_steps=self.model_steps)
        for k in ('actor','q1','q2','target1','target2','actor_opt','q_opt','alpha_opt'):
            state[k] = getattr(self,k).state_dict()
        path = Path(path); tmp = path.with_suffix('.tmp')
        torch.save(state,tmp); os.replace(tmp,path)

    @classmethod
    def load(cls,path):
        state = torch.load(path,map_location='cpu',weights_only=False)
        if state['format']=='sac_inference_v1':return InferenceSAC.load(path)
        if state['format'] not in ('sac_residual_v1','sac_world_residual_v1'): raise ValueError('Invalid SAC checkpoint')
        obj = cls(state['config'])
        for k in ('actor','q1','q2','target1','target2','actor_opt','q_opt','alpha_opt'):
            getattr(obj,k).load_state_dict(state[k])
        with torch.no_grad(): obj.log_alpha.copy_(state['log_alpha'])
        obj.steps, obj.updates = state['steps'],state['updates']
        obj.replay.restore(state['replay'])
        if obj.model_replay is not None:
            obj.model_replay.restore(state['model_replay']);obj.model_steps=state['model_steps']
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        return obj


class InferenceSAC:
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['format']!='sac_inference_v1':raise ValueError('Use demo/models/sac.pt (inference export)')
        obj=cls();obj.config=state['config'];obj.actor=Actor().eval().requires_grad_(False)
        obj.actor.load_state_dict(state['actor'])
        obj.inference_only=True;obj.model_replay=None;obj.steps=state['steps'];obj.updates=state['updates'];obj.model_steps=state['model_steps']
        return obj

    def act(self,obs,deterministic=True):
        if not deterministic:raise ValueError('Deterministic inference required')
        with torch.inference_mode():
            a,_=self.actor.sample(torch.as_tensor(obs,dtype=torch.float32)[None],True)
        return a[0].numpy()
