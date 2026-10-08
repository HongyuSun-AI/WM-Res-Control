import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from demo_utils import lane_summary


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tcp', type=Path, required=True)
    parser.add_argument('--direct', type=Path, required=True)
    parser.add_argument('--sac', type=Path, required=True)
    args = parser.parse_args()
    sources = [
        ('TCP', args.tcp),
        ('TCP + direct-gradient residual', args.direct),
        ('TCP + world-model-trained SAC', args.sac),
    ]
    sessions = [load(folder/'session.json') for _, folder in sources]
    keys = ['route_id', 'actual_wheel_friction', 'uncapped_tcp', 'target_speed_kmh', 'route_extension_m', 'stop_at_route_end', 'start_fraction', 'weather']
    assert all(all(s.get(k) == sessions[0].get(k) for k in keys) for s in sessions)
    if sessions[0].get('weather') is not None:
        weather = [load(folder/'weather_readback.json') for _, folder in sources]
        assert all(w['matched'] and w['preset']==sessions[0]['weather'] and w['actual']==weather[0]['actual'] for w in weather)
    routes = [load(folder/'planned_route.json') for _, folder in sources]
    assert routes[0] == routes[1] == routes[2]
    args.output.mkdir(parents=True, exist_ok=False)
    panels = []
    evidence = []
    for (label, folder), session in zip(sources, sessions):
        frames = load(folder/'camera_frames.json')['frames']
        refs = {r['state']['frame']: r for r in rows(next(folder.glob('reference_*.jsonl')))}
        lanes = {r['frame']: r for r in rows(folder/'lane_metrics.jsonl')}
        assert len(frames) == session['steps'] and all(b == a+1 for a,b in zip(frames,frames[1:]))
        if session.get('sac_checkpoint') and session['reason']=='step_limit':
            assert set(refs) == set(frames[1:]) | {frames[-1]+1}
            refs.pop(frames[-1]+1)
        assert set(refs) == set(lanes) == set(frames[1:])
        cap = cv2.VideoCapture(str(folder/'camera.mp4'))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
        metrics = lane_summary(folder)
        panels.append(dict(label=label, cap=cap, frames=frames, refs=refs, lanes=lanes, metrics=metrics, session=session))
        evidence.append(dict(label=label, source=str(folder.resolve()), frames=len(frames), session_reason=session['reason'],
                             collision_events=len(session['collision_frames']), metrics=metrics,
                             camera_sha256=hashlib.sha256((folder/'camera.mp4').read_bytes()).hexdigest()))
    count = max(len(p['frames']) for p in panels)
    dest = args.output/'comparison_three_way.mp4'
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), 20., (2880, 740))
    assert writer.isOpened()
    def text(canvas, value, x, y, color=(235,235,235), scale=.75):
        cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    for i in range(count):
        canvas = np.full((740,2880,3),22,dtype=np.uint8)
        for j,p in enumerate(panels):
            x = j*960; held = i >= len(p['frames'])
            if not held:
                ok, image = p['cap'].read(); assert ok and image.shape[:2] == (540,960)
                p['last'] = image
            canvas[95:635,x:x+960] = p['last']
            text(canvas,p['label'],x+20,30,scale=.85)
            f = p['frames'][min(i,len(p['frames'])-1)]; r=p['refs'].get(f); lane=p['lanes'].get(f)
            collision = bool(p['session']['collision_frames'])
            if r:
                status = 'UNKNOWN' if not lane['covered'] else ('OUTSIDE %.3f m'%lane['outside_depth_m'] if lane['outside'] else 'INSIDE')
                color = (80,80,255) if lane['covered'] and lane['outside'] else (100,225,130) if lane['covered'] else (170,180,200)
                text(canvas,'t=%.2fs speed=%.1fkm/h %s'%((i+1)*.05,r['state']['vx']*3.6,status),x+20,61,color,scale=.70)
            else:text(canvas,'t=0.05s | measurement warmup',x+20,61,scale=.70)
            if held:text(canvas,'COLLISION - final frame held' if collision else 'FINISHED - final frame held',x+20,86,(80,160,255),scale=.62)
            m=p['metrics']
            text(canvas,'Departure: %.3fm; coverage: %.1f%%'%(m['max_outside_m'],100*m['coverage']),x+20,663,scale=.72)
            text(canvas,'Collisions: %d; duration: %.2fs'%(len(p['session']['collision_frames']),len(p['frames'])/20),x+20,691,scale=.68)
            if j:canvas[:,x-2:x+2]=80
        text(canvas,'Town04 route=%s %s friction=0.5 %s' % (sessions[0]['route_id'], sessions[0].get('weather','ClearNoon'), ('target 20 km/h' if sessions[0]['target_speed_kmh'] else 'TCP autonomous speed')),20,725,scale=.72)
        writer.write(canvas)
        if i in (300,460):cv2.imwrite(str(args.output/('preview_%03d.jpg'%i)),canvas)
    writer.release()
    for p in panels:
        assert not p['cap'].read()[0]
        p['cap'].release()
    cap=cv2.VideoCapture(str(dest));n=0
    while cap.read()[0]:n+=1
    cap.release();assert n==count
    report=dict(video=str(dest.resolve()),frames=n,fps=20,duration_seconds=n/20,resolution=[2880,740],
                settings_match=True,planned_routes_match=True,sources=evidence,
                note='Simulation-time-aligned recordings',
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: video frames=%d duration=%.2fs'%(n,n/20))
    print(dest.resolve())


if __name__ == '__main__':
    main()
