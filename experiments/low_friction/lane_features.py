import math
import torch


def lane_features(poses,geometries):
    if poses.ndim!=2 or poses.shape[1]!=3 or len(poses)!=len(geometries):raise ValueError('Lane pose batch mismatch')
    outputs=[]
    for pose,(left,right) in zip(poses,geometries):
        left=left.detach().to(pose);right=right.detach().to(pose)
        if left.shape!=right.shape or left.ndim!=3 or left.shape[1:]!=(2,2) or len(left)==0:raise ValueError('Lane sides mismatch')
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('Invalid lane geometry')
        center=(left+right)/2;delta=center[:,1]-center[:,0];length2=delta.square().sum(-1)
        if (length2<=1e-8).any():raise ValueError('Degenerate lane geometry')
        u=((pose[:2]-center[:,0])*delta).sum(-1)/length2
        point=center[:,0]+u.clamp(0,1)[:,None]*delta
        j=(point-pose[:2]).square().sum(-1).argmin()
        if not 0<=u[j]<=1:raise ValueError('Lane feature outside finite coverage')
        tangent=delta[j]/length2[j].sqrt();offset=pose[:2]-point[j]
        lateral=tangent[0]*offset[1]-tangent[1]*offset[0]
        widths=(right[j]-left[j]).norm(dim=-1);width=widths[0]*(1-u[j])+widths[1]*u[j]
        if width<=0:raise ValueError('Invalid lane width')
        angle=pose[2]-torch.atan2(tangent[1],tangent[0]);angle=torch.atan2(angle.sin(),angle.cos())
        outputs.append(torch.stack((lateral/(width/2),angle/math.pi,width/3.5)))
    return torch.stack(outputs)
