from collections import Counter,deque
import hashlib
import json
import time
import copy
import numpy as np
from residual_inference import ResidualController
from road_boundary import extract_boundary
from verified_controls import index_readbacks
from live_road_status import LiveRoadStatus


class ShadowResidual:
    def __init__(self,checkpoint,output,road_map,carla,execute=False):
        self.execute=execute
        self.submitted=0
        self.road_status=LiveRoadStatus()
        self.controller=ResidualController(checkpoint)
        self.output=output;self.road_map=road_map;self.carla=carla
        self.map_hash=hashlib.sha256(road_map.to_opendrive().encode()).hexdigest()
        self.rows=deque(maxlen=11);self.readbacks=deque(maxlen=12)
        self.reasons=Counter();self.times=[];self.policy_times=[];self.boundary_times=[]
        self.policy_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.stream=(output/'shadow.jsonl').open('x',buffering=1)

    def add_readback(self,row):
        self.readbacks.append(row)

    def observe(self,row,control):
        started=time.perf_counter()
        fields=('steer','throttle','brake','hand_brake','reverse','manual_gear_shift','gear')
        base={k:getattr(control,k) for k in fields}
        boundary=None;boundary_error=None;boundary_ms=0.;policy_ms=0.
        if row is None:
            result=dict(applied=False,reason='TCP_warmup',control=dict(base),residual=[0.,0.],effective_residual=[0.,0.])
        else:
            self.rows.append(copy.deepcopy(row))
            if len(self.rows)==11 and row['state']['vx']>=.5:
                t=time.perf_counter()
                try:
                    boundary=extract_boundary(self.road_map,row,0,self.output/'shadow.jsonl',self.carla,self.map_hash)
                except (ValueError,RuntimeError) as e:boundary_error=str(e)
                boundary_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            extra={}
            if getattr(self.controller,'lane_input',False):
                from lane_geometry import fixed_lane
                try:
                    if self.road_map.name.rsplit('/',1)[-1]!='Town04':raise ValueError('Unsupported lane checkpoint context')
                    target=self.planned_lane.target(row['state']) if self.controller.planned_lane else self.controller.fixed_lane_id
                    if target is None:raise ValueError('Missing planned lane')
                    sides=fixed_lane(self.road_map,row['state'],self.carla,target=target,allow_partial=self.controller.planned_lane)
                    extra['lane_geometry']=dict(frame=row['state']['frame'],lane_id=target,sides=sides)
                except (ValueError,RuntimeError):extra['lane_geometry']=None
            result=self.controller.control(base,frame=row['state']['frame'],past_rows=list(self.rows),
                boundary=boundary,readbacks=index_readbacks(self.readbacks),**extra)
            policy_ms=(time.perf_counter()-t)*1000
        if base!={k:getattr(control,k) for k in fields}:raise RuntimeError('Shadow mutated TCP control')
        reason=result['reason'] if not result['applied'] else 'suggestion_available'
        elapsed=(time.perf_counter()-started)*1000
        item=dict(frame=row['state']['frame'] if row else None,base_control=base,
                  suggested_control=result['control'],suggestion_available=result['applied'],
                  reason=reason,residual=result['residual'],effective_residual=result['effective_residual'],
                  boundary_error=boundary_error,shadow_ms=elapsed,boundary_ms=boundary_ms,
                  inference_with_inputs_ms=policy_ms,submitted_source='TCP+residual' if self.execute and result['applied'] else 'TCP',
                  residual_submitted=bool(self.execute and result['applied']))
        chosen=result['control'] if self.execute else base
        item['submitted_control']=dict(chosen)
        item['road']=self.road_status.observe(row['state'],boundary) if row else dict(covered=False,reason='warmup')
        elapsed=(time.perf_counter()-started)*1000
        item['shadow_ms']=elapsed
        self.last_boundary=boundary
        self.last_item=item
        if self.execute and row is not None:
            self.rows[-1]['control_final']={k:chosen[k] for k in ('steer','throttle','brake')}
            self.rows[-1]['action']=[chosen['steer'],chosen['throttle']-chosen['brake']]
        self.submitted+=int(item['residual_submitted'])
        self.stream.write(json.dumps(item,allow_nan=False)+'\n')
        self.reasons[reason]+=1;self.times.append(elapsed)
        self.policy_times.append(policy_ms);self.boundary_times.append(boundary_ms)
        if self.execute:return dict(chosen)

    def close(self,reason):
        self.stream.close()
        def stats(values):
            return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(max(values))) if values else {}
        summary=dict(mode='residual_closed_loop' if self.execute else 'shadow_only',session_reason=reason,frames=len(self.times),
                     reasons=dict(self.reasons),policy_sha256=self.policy_hash,
                     shadow_ms=stats(self.times),boundary_ms=stats(self.boundary_times),
                     inference_with_inputs_ms=stats(self.policy_times),
                     shadow_over_50ms=sum(t>50 for t in self.times),residual_commands_submitted=self.submitted,
                     note='Residual-adapter latency')
        (self.output/'shadow_summary.json').write_text(json.dumps(summary,indent=2))
        print('SHADOW',json.dumps(summary),flush=True)
