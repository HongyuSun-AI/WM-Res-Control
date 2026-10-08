
import torch

from reference_geometry import integrate_local_deltas
from reference_objective import reference_objective
from road_objective import road_objective


def combined_objective(prediction, waypoint, actor_yaw, tcp_yaw, *, initial_vx,
                       prediction_frame, reference_frame, boundary_frame,
                       left_segments, right_segments, position_scale_m=1.,
                       beta_scale_deg=5., road_scale_m=1.,
                       position_weight=1., beta_weight=1., road_weight=1., low_speed_mode='strict'):
    'prediction[B,10,8];boundaries[S,2,2]'
    batch=len(prediction)
    reference=reference_objective(prediction,waypoint,actor_yaw,tcp_yaw,
        initial_vx=initial_vx,prediction_frame=prediction_frame,reference_frame=reference_frame,
        position_scale_m=position_scale_m,beta_scale_deg=beta_scale_deg,low_speed_mode=low_speed_mode)
    actor_positions=integrate_local_deltas(prediction[...,5:])[...,:2]
    roads=[road_objective(actor_positions[i],left_segments[i],right_segments[i],
                         prediction_frame=prediction_frame[i],boundary_frame=boundary_frame[i],
                         distance_scale_m=road_scale_m) for i in range(batch)]
    road_per_window=torch.stack([r['road'] for r in roads])
    road=road_per_window.mean()
    weighted=torch.stack((position_weight*reference['position'],beta_weight*reference['beta'],road_weight*road))
    return dict(total=weighted.sum()+reference['speed_guard'],position=reference['position'],beta=reference['beta'],road=road,
                speed_guard=reference['speed_guard'],predicted_low_speed_steps=reference['predicted_low_speed_steps'],
                weighted_terms=weighted,position_per_window=reference['position_per_window'],
                beta_per_window=reference['beta_per_window'],road_per_window=road_per_window,
                beta_valid=reference['beta_valid'],eligible_windows=reference['eligible_windows'],
                actor_positions=actor_positions,reference_endpoint=reference['endpoint'],
                road_per_step=torch.stack([r['per_step'] for r in roads]),
                signed_distance_m=torch.stack([r['signed_distance_m'] for r in roads]))
