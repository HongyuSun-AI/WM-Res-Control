import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/low_friction'))
from residual_inputs import build_inputs
from verified_controls import index_readbacks
from road_boundary import extract_boundary
from lane_geometry import fixed_lane
from planned_lane import PlannedLane


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,nargs='+',default=[Path('data/collection/tcp/train')])
    p.add_argument('--validation',type=Path,nargs='+',default=[Path('data/collection/tcp/validation')])
    p.add_argument('--world',type=Path,default=Path('demo/models/world.pt'))
    p.add_argument('--map-file',type=Path,default=Path('data/maps/Town04.xodr'))
    p.add_argument('--output',type=Path,default=Path('data/prepared/direct_custom'))
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--max-per-episode',type=int,default=128)
    a=p.parse_args()
    if a.output.exists() or a.stride<1 or a.max_per_episode<1:p.error('Invalid output or sampling limits')
    import carla
    first_recording=next(a.train[0].glob('reference_*.jsonl'))
    with first_recording.open() as stream:map_name=json.loads(next(stream))['map']
    if map_name.rsplit('/',1)[-1]!='Town04':raise ValueError('Town04 required')
    map_text=a.map_file.read_text();road=carla.Map(map_name,map_text);map_hash=hashlib.sha256(map_text.encode()).hexdigest()
    ck=torch.load(a.world,map_location='cpu',weights_only=True);splits={};excluded=Counter();owners={};ids=set();sources=[]
    for split,folders in [('train',a.train),('validation',a.validation)]:
        cases=[]
        for folder in folders:
            session=read(folder/'session.json');refs=list(folder.glob('reference_*.jsonl'))
            if len(refs)!=1:raise ValueError('Expected one reference recording: '+str(folder))
            identity=hashlib.sha256(refs[0].read_bytes()).hexdigest();route=session['route_id']
            if identity in ids or (route in owners and owners[route]!=split):raise ValueError('Episode/route split leakage')
            ids.add(identity);owners[route]=split
            if session['cleanup_errors'] or not session['control_readback_complete'] or session['actual_wheel_friction']!=[.5]*4:raise ValueError('Incomplete or incompatible recording')
            rows=lines(refs[0]);rb=index_readbacks(lines(folder/'control_readback.jsonl'));tracker=PlannedLane(read(folder/'planned_route.json'))
            bbox=read(folder/'vehicle_box.json');ex,ey=bbox['extent']['x'],bbox['extent']['y'];yaw=math.radians(bbox['rotation']['yaw']);c,s=math.cos(yaw),math.sin(yaw)
            corners=torch.tensor(np.array([[-ex,-ey],[-ex,ey],[ex,-ey],[ex,ey]])@np.array([[c,s],[-s,c]])+[bbox['location']['x'],bbox['location']['y']],dtype=torch.float64)
            before=len(cases);impact=min(session['collision_frames'],default=float('inf'))
            for i,row in enumerate(rows):
                target=tracker.target(row['state']);frame=row['state']['frame']
                if i<10 or (i-10)%a.stride or frame>=impact:continue
                if len(cases)-before>=a.max_per_episode:break
                try:
                    if row['state']['vx']<.5:raise ValueError('initial_low_speed')
                    boundary=extract_boundary(road,row,i,refs[0],carla,map_hash)
                    d=build_inputs(rows[i-10:i+1],boundary,ck['normalization'],readbacks=rb)
                    if not d['boundary_mask'].all():raise ValueError('incomplete_boundary')
                    lane=fixed_lane(road,row['state'],carla,target,allow_partial=True)
                    sides=[torch.tensor([s['points'] for s in boundary['segments'] if s['side']==side],dtype=torch.float64) for side in ('left','right')]
                except (ValueError,RuntimeError,KeyError) as e:excluded[str(e)]+=1;continue
                tensor=lambda k:torch.as_tensor(d[k],dtype=torch.float64)[None]
                cases.append(dict(frame=frame,episode_id=identity,route_id=route,lane_sides=lane,corners=corners,
                    inputs=(tensor('history'),tensor('base_action'),tensor('waypoint_actor'),tensor('boundary_geometry'),torch.as_tensor(d['boundary_mask'])[None]),
                    kwargs=dict(waypoint=[row['waypoints_forward_right'][0]],actor_yaw=[row['state']['yaw']],tcp_yaw=[row['tcp_yaw']],initial_vx=[row['state']['vx']],
                                prediction_frame=[frame],reference_frame=[frame],boundary_frame=[frame],left_segments=[sides[0]],right_segments=[sides[1]])))
            sources.append(dict(split=split,episode_id=identity,route_id=route,windows=len(cases)-before))
        if not cases:raise ValueError('No usable '+split+' windows: '+str(dict(excluded)))
        splits[split]=cases
    a.output.mkdir(parents=True)
    payload=dict(format='direct_windows_v1',world_sha256=hashlib.sha256(a.world.read_bytes()).hexdigest(),normalization=ck['normalization'],splits=splits,sources=sources)
    torch.save(payload,a.output/'windows.pt')
    report=dict(counts={k:len(v) for k,v in splits.items()},excluded=dict(excluded),sources=sources,map_sha256=map_hash)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
