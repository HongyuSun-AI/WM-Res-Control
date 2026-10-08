import math
from pathlib import Path
import numpy as np
import torch
from residual_inputs import build_inputs
from residual_policy import ResidualPolicy,compose_action


class ResidualController:
    def __init__(self,checkpoint,device='cpu',enabled=True):
        saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        if saved.get('format') not in ('residual_policy_v2','residual_policy_lane_v1'):raise ValueError('Unsupported residual checkpoint')
        self.lane_input=saved['format']=='residual_policy_lane_v1'
        self.fixed_lane_id=saved.get('fixed_lane_id');self.fixed_map=saved.get('fixed_map')
        self.planned_lane=saved.get('lane_reference_mode')=='planned_route'
        self.device=torch.device(device)
        self.policy=ResidualPolicy(lane_input=self.lane_input).double().to(self.device).eval().requires_grad_(False)
        self.policy.load_state_dict(saved['policy_state'])
        if any(not torch.isfinite(v).all() for v in self.policy.state_dict().values()):raise ValueError('Nonfinite policy checkpoint')
        configured=saved.get('action_limits',[.02,.02])
        if configured not in ([.02,.02],[.03,.02],[.04,.02],[.05,.02]):raise ValueError('Unsupported checkpoint action limits')
        torch.testing.assert_close(self.policy.limits,self.policy.limits.new_tensor(configured),rtol=0,atol=1e-9)
        self.normalization=saved['normalization'];self.sources=saved['sources'];self.epoch=saved['epoch']
        self.enabled=bool(enabled)

    def set_enabled(self,enabled):
        self.enabled=bool(enabled)

    def control(self,base_control,*,frame=None,past_rows=None,boundary=None,readbacks=None,lane_geometry=None):
        base=dict(base_control)
        for k,low,high in [('steer',-1,1),('throttle',0,1),('brake',0,1)]:
            if k not in base or not math.isfinite(base[k]) or not low<=base[k]<=high:
                raise ValueError('Invalid TCP base control: '+k)
        def fallback(reason):
            return dict(control=base.copy(),applied=False,reason=reason,residual=[0.,0.],effective_residual=[0.,0.])
        if not self.enabled:return fallback('disabled')
        if base.get('reverse',False) or base.get('hand_brake',False) or base.get('manual_gear_shift',False):
            return fallback('unsupported_control_mode')
        if min(base['throttle'],base['brake'])>0:return fallback('overlapping_base_pedals')
        try:
            if frame is None or past_rows is None or len(past_rows)!=11:raise ValueError('missing_history_or_frame')
            if readbacks is None:raise ValueError('missing_verified_readbacks')
            row=past_rows[-1]
            if row['state']['frame']!=frame:raise ValueError('stale_current_frame')
            if row['state']['vx']<.5:raise ValueError('initial_low_speed')
            if any(abs(row['control_final'][k]-base[k])>1e-7 for k in ('steer','throttle','brake')):
                raise ValueError('current_TCP_control_mismatch')
            inputs=build_inputs(past_rows,boundary,self.normalization,readbacks=readbacks)
            if not inputs['boundary_mask'].all():raise ValueError('incomplete_boundary')
            features=torch.as_tensor(inputs['features'],dtype=torch.float64,device=self.device)[None]
            if self.lane_input:
                from lane_features import lane_features
                if lane_geometry is None:raise ValueError('missing_fixed_lane')
                if lane_geometry['frame']!=frame or (not self.planned_lane and lane_geometry['lane_id']!=self.fixed_lane_id) or row['map'].rsplit('/',1)[-1]!=self.fixed_map:raise ValueError('Fixed lane context mismatch')
                extra=lane_features(features.new_zeros((1,3)),[lane_geometry['sides']])
                features=torch.cat((features,extra),dim=-1)
            action=torch.as_tensor([[base['steer'],base['throttle']-base['brake']]],dtype=torch.float64,device=self.device)
            with torch.inference_mode():
                residual=self.policy(features);composed=compose_action(action,residual,limits=self.policy.limits)
            result=base.copy()
            for k in ('steer','throttle','brake'):result[k]=float(composed[k][0])
            return dict(control=result,applied=True,reason='applied',residual=residual[0].tolist(),
                        effective_residual=composed['effective_residual'][0].tolist())
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            return fallback(str(error))
