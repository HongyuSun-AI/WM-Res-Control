import math
import numpy as np
import torch

def fixed_lane(road,state,carla,target=(45,0,-4),allow_partial=False):
    w=road.get_waypoint(carla.Location(x=state['x'],y=state['y'],z=state['z']))
    todo=[w];seen=set();found=None
    while todo:
        q=todo.pop();key=(q.road_id,q.section_id,q.lane_id)
        if key in seen:continue
        seen.add(key)
        if key==tuple(target):found=q;break
        if len(seen)<16:todo.extend(n for n in (q.get_left_lane(),q.get_right_lane()) if n)
    if found is None:raise ValueError('Outside audited fixed lane topology')
    left=[];right=[];back=[];ahead=[]
    for method,points in (('previous',back),('next',ahead)):
        q=found
        for _ in range(15):
            options=getattr(q,method)(1.)
            if len(options)!=1:
                if allow_partial:break
                raise ValueError('Lane branch')
            q=options[0]
            if q.is_junction or (q.road_id,q.section_id,q.lane_id)!=tuple(target):
                if allow_partial:break
                raise ValueError('Lane transition')
            points.append(q)
    if allow_partial and (len(back)<3 or len(ahead)<3):raise ValueError('Insufficient finite lane extent')
    c,s=math.cos(state['yaw']),math.sin(state['yaw']);rotation=np.array([[c,-s],[s,c]])
    for q in list(reversed(back))+[found]+ahead:
        yaw=math.radians(q.transform.rotation.yaw);loc=q.transform.location
        for sign,points in ((-1,left),(1,right)):
            xy=np.array([loc.x-sign*math.sin(yaw)*q.lane_width/2,loc.y+sign*math.cos(yaw)*q.lane_width/2])
            points.append((xy-[state['x'],state['y']])@rotation)
    return [torch.tensor(np.stack((p[:-1],p[1:]),axis=1),dtype=torch.float64) for p in (left,right)]
