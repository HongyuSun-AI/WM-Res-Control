import math
import numpy as np


def footprint_clearance(points,centers,widths,valid):
    a=centers[:-1];delta=centers[1:]-a;length2=(delta*delta).sum(1)
    good=valid[:-1]&valid[1:]&(length2>1e-8)&(length2<16.)
    projection=((points[:,None,:]-a)*delta).sum(2)/np.maximum(length2,1e-8)
    near=a+projection.clip(0,1)[...,None]*delta
    distances=((points[:,None,:]-near)**2).sum(2)
    indices=distances.argmin(1);u=projection[np.arange(len(points)),indices]
    if not np.all(good[indices]&(u>=0)&(u<=1)):
        return dict(covered=False,reason='junction_topology_or_finite_coverage')
    tangent=delta[indices];offset=points-near[np.arange(len(points)),indices]
    lateral=(tangent[:,0]*offset[:,1]-tangent[:,1]*offset[:,0])/np.sqrt(length2[indices])
    half=(widths[indices]*(1-u)+widths[indices+1]*u)/2
    clearance=float(np.min(half-np.abs(lateral)))
    return dict(covered=True,min_clearance_m=clearance,outside=clearance<0,
                outside_depth_m=max(0.,-clearance))


class RouteLaneMonitor:
    def __init__(self,road_map,route,box):
        self.centers=np.array([[t.location.x,t.location.y] for t,_ in route])
        waypoints=[road_map.get_waypoint(t.location) for t,_ in route]
        self.widths=np.array([w.lane_width if w else 0. for w in waypoints])
        self.valid=np.array([w is not None and not w.is_junction and w.lane_width>0 for w in waypoints])
        keys=[(w.road_id,w.section_id,w.lane_id) if w else None for w in waypoints]
        for i in range(len(keys)-1):
            if keys[i]!=keys[i+1]:self.valid[i:i+2]=False
        corners=np.array([[x*box.extent.x,y*box.extent.y] for x,y in ((-1,-1),(-1,1),(1,-1),(1,1))])
        angle=math.radians(box.rotation.yaw);c,s=math.cos(angle),math.sin(angle)
        self.corners=corners@np.array([[c,s],[-s,c]])+[box.location.x,box.location.y]
        self.index=0

    def observe(self,state):
        pos=np.array([state['x'],state['y']]);lo=max(0,self.index-15);hi=min(len(self.centers),self.index+81)
        self.index=lo+int(((self.centers[lo:hi]-pos)**2).sum(1).argmin())
        lo=max(0,self.index-10);hi=min(len(self.centers),self.index+11)
        yaw=state['yaw'];c,s=math.cos(yaw),math.sin(yaw)
        points=self.corners@np.array([[c,s],[-s,c]])+pos
        if hi-lo<2:return dict(covered=False,reason='route_end')
        value=footprint_clearance(points,self.centers[lo:hi],self.widths[lo:hi],self.valid[lo:hi])
        value['reference_index']=self.index
        return value
