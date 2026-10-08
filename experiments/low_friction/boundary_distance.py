import torch


def signed_boundary_distance(points, segments, side):
    'points[...,2];segments[S,2,2]'
    if side not in ('left', 'right'):
        raise ValueError('Invalid boundary side')
    geometry = segments.detach()
    start = geometry[:,0]
    tangent = geometry[:,1]-start
    length2 = tangent.square().sum(-1)
    if (length2 <= 1e-12).any():
        raise ValueError('degenerate boundary segment')
    delta = points.unsqueeze(-2)-start
    projection = (delta*tangent).sum(-1)/length2
    closest = start+projection.clamp(0,1).unsqueeze(-1)*tangent
    squared = (points.unsqueeze(-2)-closest).square().sum(-1)
    selected = squared.argmin(-1)
    selected_projection = projection.gather(-1,selected.unsqueeze(-1)).squeeze(-1)
    nearest = closest.gather(-2,selected[...,None,None].expand(*selected.shape,1,2)).squeeze(-2)
    direction = tangent[selected]
    displacement = points-nearest
    cross = direction[...,0]*displacement[...,1]-direction[...,1]*displacement[...,0]
    orientation = 1 if side == 'left' else -1
    signed = orientation*cross/length2[selected].sqrt()
    interior = (selected_projection >= 0)&(selected_projection <= 1)
    joins = torch.linalg.vector_norm(geometry[:,1,None,:]-geometry[None,:,0,:],dim=-1) <= 1e-8
    joins.fill_diagonal_(False)
    connected_start = joins.any(dim=0)[selected]
    connected_end = joins.any(dim=1)[selected]
    valid = interior | ((selected_projection < 0)&connected_start) | ((selected_projection > 1)&connected_end)
    vertex_distance = orientation*cross.sign()*torch.linalg.vector_norm(displacement,dim=-1)
    signed = torch.where(interior,signed,vertex_distance)
    signed = torch.where(valid,signed,torch.full_like(signed,float('nan')))
    return dict(signed_distance=signed,valid=valid,segment_index=selected,
                closest_point=nearest)
