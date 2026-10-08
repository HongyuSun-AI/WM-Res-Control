import math
import torch
from boundary_distance import signed_boundary_distance


class LiveRoadStatus:
    def __init__(self):self.cached=None

    def observe(self,state,boundary):
        if boundary is not None and all(any(s['side']==side for s in boundary['segments']) for side in ('left','right')):
            self.cached=(dict(state),boundary)
        if self.cached is None:return dict(covered=False,reason='no_boundary')
        origin,b=self.cached
        dx=state['x']-origin['x'];dy=state['y']-origin['y'];c=math.cos(origin['yaw']);s=math.sin(origin['yaw'])
        point=torch.tensor([[c*dx+s*dy,-s*dx+c*dy]],dtype=torch.float64)
        distances={}
        for side in ('left','right'):
            seg=torch.tensor([x['points'] for x in b['segments'] if x['side']==side],dtype=torch.float64)
            value=signed_boundary_distance(point,seg,side)
            if not value['valid'].all():return dict(covered=False,reason='outside_finite_coverage')
            distances[side]=float(value['signed_distance'][0])
        return dict(covered=True,left_m=distances['left'],right_m=distances['right'],
                    outside=min(distances.values())<0,boundary_frame=origin['frame'])
