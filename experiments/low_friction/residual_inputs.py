import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verified_controls import verified_action

FIELDS=('vx','vy','r','ax','ay')
SENSORS=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','GPS','IMU','SPEED')
OFFSETS=np.arange(-10.,31.,2.)


def build_inputs(past_rows,boundary,normalization,readbacks=None):
    if len(past_rows)!=11:raise ValueError('Supply exactly 11 past/current rows')
    row=past_rows[-1];state=row['state'];frame=state['frame']
    for i,r in enumerate(past_rows):
        if r['state']['frame']!=frame-10+i or abs(r['state']['time']-state['time']-(i-10)*.05)>1e-4:
            raise ValueError('History frame/time mismatch')
        if not r['heading_valid'] or not r['frame_alignment_verified'] or any(r['sensor_frames'].get(k)!=r['state']['frame'] for k in SENSORS):
            raise ValueError('Unaligned input')
        a=np.asarray(r['action']);c=r['control_final']
        if a.shape!=(2,) or not np.isfinite(a).all() or (np.abs(a)>1).any():raise ValueError('Invalid action')
        if not (0<=c['throttle']<=1 and 0<=c['brake']<=1) or min(c['throttle'],c['brake'])>1e-4:
            raise ValueError('Invalid longitudinal controls')
        if not np.allclose(a,[c['steer'],c['throttle']-c['brake']],atol=1e-6,rtol=0):raise ValueError('Control mismatch')
    previous_actions=([r['action'] for r in past_rows[:-1]] if readbacks is None else
                      [verified_action(r,readbacks,available_frame=frame) for r in past_rows[:-1]])
    history=np.concatenate((np.asarray([[r['state'][k] for k in FIELDS] for r in past_rows[1:]]),
                            np.asarray(previous_actions)),axis=1)
    mean=np.asarray(normalization['state']['mean']+normalization['action']['mean'])
    scale=np.asarray(normalization['state']['scale']+normalization['action']['scale'])
    if mean.shape!=(7,) or scale.shape!=(7,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid normalization')
    if not np.allclose(row['waypoint_times_seconds'],[.5,1.,1.5,2.]):raise ValueError('TCP waypoint timing mismatch')
    wp=np.asarray(row['waypoints_forward_right'][0]);angle=row['tcp_yaw']-state['yaw']
    c,s=np.cos(angle),np.sin(angle)
    waypoint=np.array([[c,-s],[s,c]])@wp+[-1.4,0.]
    geometry=np.zeros((21,2,4));mask=np.zeros((21,2),dtype=bool)
    if boundary is not None:
        if boundary['frame']!=frame or boundary['map']!=row['map']:
            raise ValueError('Boundary frame/map mismatch')
        if boundary['coordinate_frame']!='initial actor x forward y right, meters; not GNSS origin':
            raise ValueError('Boundary coordinates unsupported')
        stations=boundary['stations']
        if stations:
            anchor=min(stations,key=lambda v:np.linalg.norm(v['center_local']))
            for j,offset in enumerate(OFFSETS):
                forward_s=-np.sign(anchor['lane'][2])
                matches=[st for st in stations if st['lane']==anchor['lane'] and abs((st['s']-anchor['s'])*forward_s-offset)<1e-4]
                if len(matches)>1:raise ValueError('Ambiguous boundary station')
                if not matches:continue
                st=matches[0]
                for side_id,side in enumerate(('left','right')):
                    edge=st[side]
                    if not edge['valid']:continue
                    point=np.asarray(edge['local'])
                    segments=[np.asarray(seg['points']) for seg in boundary['segments'] if seg['side']==side]
                    connected=[seg for seg in segments if np.allclose(seg[0],point,atol=1e-8,rtol=0)]
                    if not connected:connected=[seg for seg in segments if np.allclose(seg[1],point,atol=1e-8,rtol=0)]
                    if len(connected)!=1:continue
                    tangent=connected[0][1]-connected[0][0];length=np.linalg.norm(tangent)
                    if not np.isfinite(length) or length<=1e-8:continue
                    geometry[j,side_id]=np.r_[point,tangent/length];mask[j,side_id]=True
    if not np.isfinite(history).all() or not np.isfinite(waypoint).all() or not np.isfinite(geometry).all():
        raise ValueError('Nonfinite inputs')
    action=np.asarray(row['action'])
    normalized_geometry=geometry.copy();normalized_geometry[:,:,:2]/=30.
    features=np.concatenate(((history-mean).reshape(-1)/np.tile(scale,10),
                             (action-mean[5:])/scale[5:],waypoint/30.,
                             normalized_geometry.reshape(-1),mask.astype(float).reshape(-1)))
    return dict(history=history,history_normalized=(history-mean)/scale,
                base_action=action,base_action_normalized=(action-mean[5:])/scale[5:],
                waypoint_actor=waypoint,waypoint_normalized=waypoint/30.,
                boundary_geometry=geometry,boundary_normalized=normalized_geometry,boundary_mask=mask,
                features=features,history_frames=np.array([r['state']['frame'] for r in past_rows[1:]]),
                history_action_frames=np.array([r['state']['frame'] for r in past_rows[:-1]]))

