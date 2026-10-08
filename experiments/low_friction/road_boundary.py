import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def lane_key(w):
    return (w.road_id,w.section_id,w.lane_id)


def edge(w,side):
    yaw=math.radians(w.transform.rotation.yaw)
    sign=-1 if side=='left' else 1
    p=w.transform.location
    return [p.x-sign*math.sin(yaw)*w.lane_width/2,
            p.y+sign*math.cos(yaw)*w.lane_width/2,p.z]


def outer_edge(start,side,driving):
    w=start;visited={lane_key(w)}
    for _ in range(16):
        n=w.get_left_lane() if side=='left' else w.get_right_lane()
        if n is None:
            return dict(point=edge(w,side),valid=True,reason='no_adjacent_lane',lanes=sorted(visited))
        if n.lane_type!=driving:
            return dict(point=edge(w,side),valid=True,reason='adjacent_'+str(n.lane_type),lanes=sorted(visited))
        reason=None
        if n.is_junction:reason='junction'
        elif lane_key(n) in visited:reason='lane_cycle'
        elif n.road_id!=w.road_id or n.section_id!=w.section_id:reason='different_road_or_section'
        elif n.lane_id*w.lane_id<0:
            return dict(point=edge(w,side),valid=True,reason='opposite_lane_envelope_end',lanes=sorted(visited))
        elif n.lane_id*w.lane_id==0 or math.cos(math.radians(n.transform.rotation.yaw-w.transform.rotation.yaw))<.9:
            reason='opposite_or_divergent_direction'
        elif not math.isfinite(n.lane_width) or n.lane_width<=0:reason='invalid_width'
        else:
            a=np.asarray(edge(w,side));b=np.asarray(edge(n,'right' if side=='left' else 'left'))
            if np.linalg.norm(a[:2]-b[:2])>.5 or abs(a[2]-b[2])>.5:reason='nonadjacent_geometry'
        if reason:
            return dict(point=None,valid=False,reason=reason,lanes=sorted(visited))
        visited.add(lane_key(n));w=n
    return dict(point=None,valid=False,reason='lane_limit',lanes=sorted(visited))


def station(w,driving):
    if w.is_junction or w.lane_type!=driving or not math.isfinite(w.lane_width) or w.lane_width<=0:
        raise ValueError('Invalid driving-lane station')
    p=w.transform.location
    return dict(center=[p.x,p.y,p.z],lane=list(lane_key(w)),s=w.s,
                left=outer_edge(w,'left',driving),right=outer_edge(w,'right',driving))


def trace(start,method,count,spacing,driving):
    out=[];w=start;reason='distance_limit'
    for _ in range(count):
        options=getattr(w,method)(spacing)
        if len(options)!=1:
            reason='branch' if options else 'lane_end';break
        n=options[0]
        if n.is_junction:reason='junction';break
        if n.lane_type!=driving:reason='non_driving';break
        if n.road_id!=w.road_id or n.section_id!=w.section_id or n.lane_id!=w.lane_id:
            reason='topology_transition';break
        out.append(station(n,driving));w=n
    return out,reason


def local(points,state):
    a=np.asarray(points);d=a[:,:2]-[state['x'],state['y']]
    c,s=math.cos(state['yaw']),math.sin(state['yaw'])
    return d@np.array([[c,-s],[s,c]])


def extract_boundary(road_map,row,index,recording,carla,map_hash=None):
    state=row['state']
    if road_map.name.rsplit('/',1)[-1]!=row['map'].rsplit('/',1)[-1]:raise ValueError('Server map differs from recording')
    w=road_map.get_waypoint(carla.Location(x=state['x'],y=state['y'],z=state['z']),
                           project_to_road=False,lane_type=carla.LaneType.Driving)
    if w is None:raise ValueError('Actor outside driving lane')
    if abs(w.transform.location.z-state['z'])>2:raise ValueError('Waypoint elevation mismatch')
    middle=station(w,carla.LaneType.Driving)
    back,back_stop=trace(w,'previous',5,2.,carla.LaneType.Driving)
    ahead,ahead_stop=trace(w,'next',15,2.,carla.LaneType.Driving)
    stations=list(reversed(back))+[middle]+ahead
    for st in stations:
        st['center_local']=local([st['center']],state)[0].tolist()
        for side in ('left','right'):
            e=st[side];e['local']=local([e['point']],state)[0].tolist() if e['valid'] else None
    segments=[]
    for side in ('left','right'):
        for a,b in zip(stations,stations[1:]):
            ea,eb=a[side],b[side]
            if ea['valid'] and eb['valid'] and ea['lanes']==eb['lanes'] and np.linalg.norm(np.array(ea['point'])-eb['point'])<=6:
                segments.append(dict(side=side,points=[ea['local'],eb['local']]))
    report=dict(frame=state['frame'],recording=str(Path(recording).resolve()),index=index,map=road_map.name,
                opendrive_sha256=map_hash or hashlib.sha256(road_map.to_opendrive().encode()).hexdigest(),
                coordinate_frame='initial actor x forward y right, meters; not GNSS origin',
                bounds_requested_m=dict(behind=10,ahead=30),spacing_m=2,
                stop_reasons=dict(behind=back_stop,ahead=ahead_stop),stations=stations,segments=segments,
                ready_for_loss=False,
                note='Same-direction contiguous Driving envelope')
    return report


