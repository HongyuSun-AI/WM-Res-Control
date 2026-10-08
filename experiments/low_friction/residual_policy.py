import torch
from torch import nn


class ResidualPolicy(nn.Module):
    '[B, 284] -> [B, 2]'
    def __init__(self,lane_input=False):
        super().__init__()
        self.lane_input=lane_input
        self.feature_count=287 if lane_input else 284
        self.history_encoder=nn.Sequential(nn.Linear(70,64),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(64+214+(3 if lane_input else 0),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.output=nn.Linear(32,2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('limits',torch.tensor([.02,.02]))

    def forward(self,features):
        mask=features[:,242:284]
        enabled=(mask==1).all(dim=-1,keepdim=True)
        history=self.history_encoder(features[:,:70])
        hidden=self.fusion(torch.cat((history,features[:,70:]),dim=-1))
        residual=torch.tanh(self.output(hidden))*self.limits
        return torch.where(enabled,residual,torch.zeros_like(residual))


def compose_action(base_action,residual,limits=(.02,.02)):
    action=(base_action+residual).clamp(-1,1)
    return dict(action=action,steer=action[:,0],throttle=action[:,1].clamp(min=0),
                brake=(-action[:,1]).clamp(min=0),effective_residual=action-base_action)
