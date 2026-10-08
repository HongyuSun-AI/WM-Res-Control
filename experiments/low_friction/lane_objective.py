import torch
from boundary_distance import signed_boundary_distance
from reference_geometry import integrate_local_deltas
from combined_objective import combined_objective


def lane_objective(poses,left_segments,right_segments,corners,*,prediction_frame,lane_frame,
                   margin_m=.1,distance_scale_m=1.):
    'poses[T,3];corners[4,2]'
    if prediction_frame!=lane_frame:raise ValueError('Lane/pose frame mismatch')
    corners=corners.detach()
    c,s=poses[:,2].cos()[:,None],poses[:,2].sin()[:,None]
    x=poses[:,0,None]+c*corners[None,:,0]-s*corners[None,:,1]
    y=poses[:,1,None]+s*corners[None,:,0]+c*corners[None,:,1]
    points=torch.stack((x,y),dim=-1)
    left=signed_boundary_distance(points,left_segments,'left')
    right=signed_boundary_distance(points,right_segments,'right')
    if not (left['valid'].all() and right['valid'].all()):raise ValueError('Incomplete lane footprint coverage')
    distances=torch.stack((left['signed_distance'],right['signed_distance']),dim=-1)
    min_clearance=distances.amin(dim=(1,2))
    penalty=(torch.relu(margin_m-min_clearance)/distance_scale_m).square()
    return dict(lane=penalty.mean(),lane_per_step=penalty,min_clearance_m=min_clearance,
                footprint_points=points,outside_depth_m=torch.relu(-min_clearance))


def lane_combined_objective(prediction,*,lane_left_segments,lane_right_segments,
                            footprint_corners,lane_frame,lane_weight=1.,lane_margin_m=.1,**kwargs):
    result=combined_objective(prediction,**kwargs)
    n=len(prediction)
    poses=integrate_local_deltas(prediction[...,5:])
    terms=[lane_objective(poses[i],lane_left_segments[i],lane_right_segments[i],footprint_corners[i],
                          prediction_frame=kwargs['prediction_frame'][i],lane_frame=lane_frame[i],margin_m=lane_margin_m)
           for i in range(n)]
    per_window=torch.stack([t['lane'] for t in terms]);lane=per_window.mean()
    result.update(total=result['total']+lane_weight*lane,lane=lane,lane_per_window=per_window,
                  lane_min_clearance_m=torch.stack([t['min_clearance_m'] for t in terms]))
    return result
