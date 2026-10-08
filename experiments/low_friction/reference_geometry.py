import torch




def integrate_local_deltas(deltas):
    '[..., T, 3]'
    yaw = deltas[..., 2].cumsum(dim=-1)
    previous_yaw = torch.cat((torch.zeros_like(yaw[..., :1]), yaw[..., :-1]), dim=-1)
    c, s = previous_yaw.cos(), previous_yaw.sin()
    x = (c*deltas[..., 0]-s*deltas[..., 1]).cumsum(dim=-1)
    y = (s*deltas[..., 0]+c*deltas[..., 1]).cumsum(dim=-1)
    return torch.stack((x,y,yaw), dim=-1)


def actor_poses_to_gnss(poses, offset=(-1.4, 0.0)):
    '[..., T, 2]'
    b = torch.as_tensor(offset, dtype=poses.dtype, device=poses.device)
    c, s = poses[..., 2].cos(), poses[..., 2].sin()
    gx = poses[..., 0]+c*b[0]-s*b[1]-b[0]
    gy = poses[..., 1]+s*b[0]+c*b[1]-b[1]
    return torch.stack((gx,gy), dim=-1)
