import math
import torch
from reference_geometry import integrate_local_deltas, actor_poses_to_gnss


def align_first_waypoint(deltas, waypoint, actor_yaw, tcp_yaw, *,
                         prediction_frame, reference_frame, dt=.05,
                         waypoint_time=.5, offset=(-1.4,0.)):
    'deltas[B,10,3];waypoint[B,2];world_headings[B]'
    if not math.isfinite(dt) or not math.isfinite(waypoint_time) or not math.isclose(dt,.05,abs_tol=1e-9,rel_tol=0) or not math.isclose(waypoint_time,.5,abs_tol=1e-9,rel_tol=0):
        raise ValueError('Required horizon: 10×0.05s')
    def value(x,shape,name):
        t=torch.as_tensor(x,device=deltas.device,dtype=deltas.dtype)
        return t
    batch=len(deltas)
    waypoint=value(waypoint,(batch,2),'waypoint')
    actor_yaw=value(actor_yaw,(batch,),'actor_yaw')
    tcp_yaw=value(tcp_yaw,(batch,),'tcp_yaw')
    frames=[]
    for name,x in (('prediction_frame',prediction_frame),('reference_frame',reference_frame)):
        t=torch.as_tensor(x,device=deltas.device)
        frames.append(t)
    if not torch.equal(*frames): raise ValueError('TCP frame mismatch')
    actor_poses=integrate_local_deltas(deltas)
    gnss=actor_poses_to_gnss(actor_poses,offset)
    angle=actor_yaw-tcp_yaw
    c,s=angle.cos()[:,None],angle.sin()[:,None]
    trajectory=torch.stack((c*gnss[...,0]-s*gnss[...,1],s*gnss[...,0]+c*gnss[...,1]),dim=-1)
    residual=trajectory[:,-1]-waypoint
    return dict(gnss_trajectory_tcp_axes=trajectory, endpoint=trajectory[:,-1],
                reference=waypoint,residual=residual,distance_m=torch.linalg.vector_norm(residual,dim=-1))
