import math
import torch
from reference_alignment import align_first_waypoint


def reference_objective(prediction, waypoint, actor_yaw, tcp_yaw, *,
                        initial_vx, prediction_frame, reference_frame,
                        position_scale_m=1., beta_scale_deg=5.,
                        position_weight=1., beta_weight=1., low_speed_mode='strict'):
    'prediction [B, 10, 8]'
    def fixed(x):
        return torch.as_tensor(x,dtype=prediction.dtype,device=prediction.device).detach()
    initial_vx=fixed(initial_vx)
    aligned=align_first_waypoint(prediction[...,5:],fixed(waypoint),fixed(actor_yaw),fixed(tcp_yaw),
                                prediction_frame=prediction_frame,reference_frame=reference_frame)
    eligible=initial_vx>=.5
    mask=eligible[:,None].expand(-1,10)
    if low_speed_mode not in ('strict','regularized'):raise ValueError('Unknown low-speed mode')
    if low_speed_mode=='strict' and torch.any(mask & (prediction[...,0]<.5)):
        raise ValueError('Sideslip requires vx>=0.5m/s')
    vx=torch.where(mask,prediction[...,0],torch.ones_like(prediction[...,0]))
    vy=torch.where(mask,prediction[...,1],torch.zeros_like(prediction[...,1]))
    guard=((.5-vx).clamp(min=0)/.5).square() if low_speed_mode=='regularized' else vx*0
    speed_guard_per_window=guard.mean(dim=-1)
    speed_guard=speed_guard_per_window.sum()/eligible.sum().clamp(min=1)
    if low_speed_mode=='regularized':vx=vx.clamp(min=.5)
    beta=torch.atan2(vy,vx)
    position_per_window=(aligned['residual']/position_scale_m).square().sum(dim=-1)
    beta_per_window=(beta/math.radians(beta_scale_deg)).square().mean(dim=-1)
    position_loss=position_per_window.mean()
    beta_loss=beta_per_window.sum()/eligible.sum().clamp(min=1)
    total=position_weight*position_loss+beta_weight*beta_loss+speed_guard
    return dict(total=total,position=position_loss,beta=beta_loss,
                position_per_window=position_per_window,beta_per_window=beta_per_window,
                beta_rad=beta,beta_valid=mask,eligible_windows=eligible.sum(),
                speed_guard=speed_guard,speed_guard_per_window=speed_guard_per_window,
                predicted_low_speed_steps=(mask & (prediction[...,0]<.5)).sum(),
                residual_m=aligned['residual'],endpoint=aligned['endpoint'])
