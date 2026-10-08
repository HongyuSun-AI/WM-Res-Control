
import torch

from boundary_distance import signed_boundary_distance


def road_objective(points, left_segments, right_segments, *,
                   prediction_frame, boundary_frame, distance_scale_m=1.):
    'points[T,2];boundaries[S,2,2]'
    if prediction_frame != boundary_frame:
        raise ValueError('Boundary frame mismatch')
    left = signed_boundary_distance(points, left_segments, 'left')
    right = signed_boundary_distance(points, right_segments, 'right')
    if not (left['valid'].all() and right['valid'].all()):
        raise ValueError('Incomplete boundary coverage')
    distances = torch.stack((left['signed_distance'], right['signed_distance']), dim=-1)
    violations = torch.relu(-distances)
    per_step = (violations/distance_scale_m).square().sum(-1)
    return dict(road=per_step.mean(), per_step=per_step,
                signed_distance_m=distances, violation_m=violations)
