import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from demo_utils import ROOT,SCRIPTS
from demo_utils import stage
import time


def read_lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def gnss(state):
    return np.array([state['x']-1.4*math.cos(state['yaw']),state['y']-1.4*math.sin(state['yaw'])])


def summarize(directory):
    session=json.loads((directory/'session.json').read_text())
    files=list(directory.glob('reference_*.jsonl'))
    rows=read_lines(files[0]) if len(files)==1 else []
    rb=read_lines(directory/'control_readback.jsonl')
    shadow=read_lines(directory/'shadow.jsonl') if (directory/'shadow.jsonl').exists() else []
    beta=[math.degrees(r['state']['beta']) for r in rows if r['state']['beta_valid']]
    errors=[]
    for a,b in zip(rows,rows[10:]):
        if b['state']['frame']-a['state']['frame']!=10:continue
        yaw=a['tcp_yaw'];c,s=math.cos(yaw),math.sin(yaw)
        wp=np.asarray(a['waypoints_forward_right'][0]);target=gnss(a['state'])+np.array([[c,-s],[s,c]])@wp
        errors.append(float(np.linalg.norm(gnss(b['state'])-target)))
    road=[x['road'] for x in shadow if x.get('road',{}).get('covered')]
    sac=read_lines(directory/'sac_transitions.jsonl') if (directory/'sac_transitions.jsonl').exists() else None
    return dict(session_reason=session['reason'],steps=session['steps'],collision_events=len(session['collision_frames']),
                cleanup_errors=session['cleanup_errors'],readbacks=len(rb),readback_matched=sum(x['matched'] for x in rb),
                beta_valid_frames=len(beta),beta_rms_deg=float(np.sqrt(np.mean(np.square(beta)))) if beta else None,
                beta_abs_max_deg=max(map(abs,beta)) if beta else None,
                own_TCP_05s_position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                position_windows=len(errors),road_covered_frames=len(road),road_unknown_frames=len(shadow)-len(road),
                road_outside_frames=sum(x['outside'] for x in road),
                residual_submitted_frames=sum(x['gate'] for x in sac) if sac is not None else sum(x['residual_submitted'] for x in shadow),
                road_metrics_source='shadow actor-point envelope' if shadow else 'See lane_metrics.jsonl for vehicle footprint'),rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=Path('demo/models/direct.pt'))
    p.add_argument('--sac-policy',action='store_true',help='Compare SAC with TCP')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--weather',choices=('ClearNoon','MidRainyNoon','HardRainNoon'),default='ClearNoon')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--route-id',default='24785')
    p.add_argument('--start-fraction',type=float,default=0.)
    p.add_argument('--video',action='store_true')
    p.add_argument('--target-speed-kmh',type=float)
    p.add_argument('--stop-at-route-end',action='store_true')
    p.add_argument('--uncapped-tcp',action='store_true')
    p.add_argument('--extend-route-m',type=float,default=0.)
    a=p.parse_args()
    if a.output.exists() or not 41<=a.steps<=1200 or not 0<=a.start_fraction<=.8:p.error('Invalid output or route parameters')
    a.output=a.output.resolve();a.policy=a.policy.resolve()
    if not a.policy.is_file():p.error('Policy missing')
    a.output.mkdir(parents=True)
    reports={};recordings={}
    for name in ('tcp','residual'):
        print('START',name,'log=',a.output/(name+'.log'),flush=True)
        command=[SCRIPTS/'record_tcp_reference.py','--output',a.output/name,'--port',a.port,
                 '--steps',a.steps,'--route-id',a.route_id,'--start-fraction',a.start_fraction,
                 '--tire-friction',.5,'--lane-metrics','--weather',a.weather]
        if a.sac_policy:
            if name=='residual':command+=['--sac-checkpoint',a.policy,'--sac-evaluate']
        else:
            command+=['--shadow-policy',a.policy]
            if name=='residual':command+=['--execute-residual']
        if a.video:command+=['--video']
        if a.target_speed_kmh is not None:command+=['--target-speed-kmh',a.target_speed_kmh]
        if a.stop_at_route_end:command+=['--stop-at-route-end']
        if a.uncapped_tcp:command+=['--uncapped-tcp']
        if a.extend_route_m:command+=['--extend-route-m',a.extend_route_m]
        code=stage(command,a.output/(name+'.log'),time.monotonic()+1800,1800)
        if not (a.output/name/'session.json').exists():raise RuntimeError('No session; inspect '+str(a.output/(name+'.log')))
        reports[name],recordings[name]=summarize(a.output/name)
        if reports[name]['cleanup_errors']:raise RuntimeError('Cleanup failed')
        print('DONE',name,json.dumps(reports[name]),flush=True)
        (a.output/'comparison.json').write_text(json.dumps(dict(runs=reports,complete=False),indent=2))
        if code and not reports[name]['collision_events']:raise RuntimeError('Recorder failed')
    left,right=recordings['tcp'],recordings['residual'];n=min(len(left),len(right))
    deviation=[math.hypot(a['state']['x']-b['state']['x'],a['state']['y']-b['state']['y']) for a,b in zip(left,right)]
    result=dict(runs=reports,weather=a.weather,uncapped_tcp=a.uncapped_tcp,route_id=a.route_id,extend_route_m=a.extend_route_m,target_speed_kmh=a.target_speed_kmh,stop_at_route_end=a.stop_at_route_end,base_control_mode='TCP_without_extra_low_speed_throttle_cap' if a.uncapped_tcp else ('frozen_TCP_steering+fixed_speed_PID' if a.target_speed_kmh is not None else 'original_TCP'),complete=all(r['session_reason'] in ('step_limit','route_complete') for r in reports.values()),
                policy_type='SAC' if a.sac_policy else 'direct_gradient',policy=str(a.policy),policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
                common_frames=n,residual_vs_TCP_actor_trajectory_rmse_m=float(np.sqrt(np.mean(np.square(deviation)))) if deviation else None,
                note='Single paired run')
    if a.sac_policy:
        sac_report=json.loads((a.output/'residual/sac_summary.json').read_text())
        result['world_assisted_policy']=sac_report['world_assisted_policy']
        result['policy_evaluation_only']=sac_report['evaluation'] and sac_report['new_updates']==0 and not sac_report['actor_changed']
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    print('PAIR RECORDED: comparison saved',flush=True)
    print('OUTPUT',a.output)


if __name__=='__main__':main()
