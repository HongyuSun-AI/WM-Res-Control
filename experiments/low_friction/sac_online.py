from collections import deque, Counter
import copy
import hashlib
import json
import math
import numpy as np
import torch
from sac_core import SAC, LIMITS
from residual_inputs import build_inputs
from road_boundary import extract_boundary
from verified_controls import index_readbacks
from lane_geometry import fixed_lane
from lane_features import lane_features
from state_monitor import read_state
from route_speed import RouteSpeedPlan


def reward_terms(previous, current, progress, lane, action, last_action, waypoint,
                 collision=False, goal=False, collision_penalty=30.):
    if not math.isfinite(collision_penalty) or collision_penalty<=0:raise ValueError('Invalid collision penalty')
    dt = .05
    d = np.array([current['x']-previous['x'],current['y']-previous['y']])
    c,s = math.cos(previous['yaw']),math.sin(previous['yaw'])
    local = np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1]])
    length = np.linalg.norm(waypoint)
    lateral = float((waypoint[0]*local[1]-waypoint[1]*local[0])/length) if length>1e-4 else 0.
    beta = math.atan2(current['vy'],current['vx']) if current['vx']>=.5 else 0.
    outside = lane.get('outside_depth_m',0.) if lane.get('covered') else 0.
    return dict(progress=float(progress),time=-.02*dt,
                tcp_lateral=-.5*lateral**2*dt,road=-4.*outside**2*dt,
                beta=-.05*min((beta/math.radians(5))**2,36.)*dt,
                magnitude=-.01*float(np.mean(action**2))*dt,
                smoothness=-.02*float(np.mean((action-last_action)**2)),
                collision=-float(collision_penalty) if collision else 0.,goal=10. if goal else 0.)


class SACOnline:
    def __init__(self, checkpoint, output, road_map, carla, evaluate=False):
        torch.set_num_threads(1)
        self.sac = SAC.load(checkpoint)
        if getattr(self.sac, "inference_only", False) and not evaluate:raise ValueError('Training checkpoint required')
        self.evaluate = evaluate
        self.world = None
        if self.sac.model_replay is not None and not evaluate:
            from sac_world import WorldBranches
            self.world = WorldBranches(self.sac.config)
        self.model_context = None
        self.model_rejections = Counter()
        self.initial_model_steps = self.sac.model_steps
        self.output,self.road_map,self.carla = output,road_map,carla
        self.map_hash = hashlib.sha256(road_map.to_opendrive().encode()).hexdigest()
        self.rows,self.readbacks = deque(maxlen=11),deque(maxlen=12)
        self.pending = None
        self.last_action = np.zeros(2,np.float32)
        self.progress_plan = None
        self.progress_high = None
        self.reasons = Counter()
        self.terms = Counter()
        self.count = self.gated = self.covered = 0
        self.initial_updates = self.sac.updates
        self.initial_actor = {k:v.detach().clone() for k,v in self.sac.actor.state_dict().items()}
        self.stream = (output/'sac_transitions.jsonl').open('x',buffering=1)
        self.metrics = (output/'sac_updates.jsonl').open('x',buffering=1)
        self.model_stream = (output/'model_transitions.jsonl').open('x',buffering=1) if self.world is not None else None
        self.closed = False

    def add_readback(self,row):
        if not row['matched']: raise RuntimeError('SAC requires matched executed controls')
        self.readbacks.append(row)

    def inputs(self,row,control):
        self.model_context = None
        self.rows.append(copy.deepcopy(row))
        if len(self.rows)<11: return None,False,None,'history_warmup'
        boundary = None
        try:
            boundary = extract_boundary(self.road_map,row,0,self.output/'sac_transitions.jsonl',self.carla,self.map_hash)
        except (ValueError,RuntimeError): pass
        inputs = build_inputs(list(self.rows),boundary,self.sac.config['normalization'],
                              readbacks=index_readbacks(self.readbacks))
        extra = np.zeros(3,np.float32)
        lane_ok = False
        sides = None
        try:
            target = self.planned_lane.target(row['state'])
            if target is not None:
                sides = fixed_lane(self.road_map,row['state'],self.carla,target=target,allow_partial=True)
                extra = lane_features(torch.zeros((1,3),dtype=torch.float64),[sides])[0].numpy()
                lane_ok = True
        except (ValueError,RuntimeError): pass
        mode_ok = not (control.reverse or control.hand_brake or control.manual_gear_shift)
        gate = bool(inputs['boundary_mask'].all() and lane_ok and row['state']['vx']>=.5 and mode_ok)
        reason = 'enabled' if gate else 'geometry_speed_or_control_fallback'
        features = np.r_[inputs['features'],extra].astype(np.float32)
        if features.shape!=(287,) or not np.isfinite(features).all(): raise RuntimeError('Invalid SAC observation')
        if gate:self.model_context=(inputs,sides)
        return features,gate,inputs['waypoint_actor'],reason

    def progress(self,state):
        if self.progress_plan is None:
            planned = json.loads((self.output/'planned_route.json').read_text())
            self.progress_plan = RouteSpeedPlan(planned,10000.)
        remaining = self.progress_plan.observe(state)['remaining_route_m']
        p = self.progress_plan.arc[-1]-remaining
        if self.progress_high is None: self.progress_high = p
        gain = max(0.,float(p-self.progress_high))
        self.progress_high = max(self.progress_high,p)
        return gain

    def complete_transition(self,state,obs,gate,terminated=False,truncated=False,reason='running'):
        if self.pending is None:
            self.progress(state)
            return
        p = self.pending
        if state['frame']!=p['state']['frame']+1: raise RuntimeError('SAC transition frame gap')
        if not self.readbacks or self.readbacks[-1]['observed_frame']!=state['frame']:
            raise RuntimeError('Missing current action readback')
        lane = self.lane_monitor.observe(state)
        terms = reward_terms(p['state'],state,self.progress(state),lane,p['action'],p['previous_action'],
                             p['waypoint'],collision=reason=='collision',goal=reason=='route_complete',
                             collision_penalty=self.sac.config.get('collision_penalty',30.))
        reward = sum(terms.values())
        if not self.evaluate:
            self.sac.replay.add(obs=p['obs'],action=p['action'],reward=reward,next_obs=obs,
                                terminated=terminated,truncated=truncated,gate=p['gate'],next_gate=gate)
            self.sac.steps += 1
            cfg = self.sac.config
            if self.sac.steps>=cfg['learning_starts'] and self.sac.replay.size>=cfg['batch_size'] and self.sac.steps%cfg['update_every']==0:
                self.metrics.write(json.dumps(self.sac.update(),allow_nan=False)+'\n')
        self.count += 1; self.gated += int(p['gate']); self.covered += int(lane.get('covered',False))
        self.terms.update(terms)
        self.stream.write(json.dumps(dict(frame=p['state']['frame'],next_frame=state['frame'],
            action=p['action'].tolist(),effective_residual=p['effective'].tolist(),gate=p['gate'],next_gate=gate,
            reward=reward,terms=terms,terminated=terminated,truncated=truncated,reason=reason,
            lane=lane,speed_kmh=state['vx']*3.6),allow_nan=False)+'\n')
        if self.count%100==0: print('SAC transitions={} updates={} reward={:.4f}'.format(self.count,self.sac.updates,reward),flush=True)
        self.pending = None

    def observe(self,row,control,final_reason=None):
        base = {k:getattr(control,k) for k in ('steer','throttle','brake','hand_brake','reverse','manual_gear_shift','gear')}
        if row is None: return base
        obs,gate,waypoint,reason = self.inputs(row,control)
        self.reasons[reason] += 1
        if obs is None: return base
        self.complete_transition(row['state'],obs,gate,truncated=final_reason=='step_limit',reason=final_reason or 'running')
        if final_reason is not None: return base
        action = self.sac.act(obs,self.evaluate) if gate else np.zeros(2,np.float32)
        if self.world is not None and gate:
            try:
                transition,diagnostic=self.world.sample(self.sac,self.model_context,obs,row['state'],
                    self.last_action,self.lane_monitor,self.progress_plan,self.progress_high,reward_terms)
                self.sac.model_replay.add(**transition);self.sac.model_steps+=1
                self.model_stream.write(json.dumps(diagnostic,allow_nan=False)+'\n')
            except ValueError as error:
                self.model_rejections[str(error)]+=1
        residual = action*LIMITS
        base_action = np.array([base['steer'],base['throttle']-base['brake']])
        composed = np.clip(base_action+residual,-1,1)
        chosen = dict(base,steer=float(composed[0]),throttle=float(max(0,composed[1])),brake=float(max(0,-composed[1])))
        self.pending = dict(obs=obs,gate=gate,action=action,state=row['state'],waypoint=waypoint,
                            effective=composed-base_action,previous_action=self.last_action.copy())
        self.last_action = action.copy()
        self.rows[-1]['control_final'] = {k:chosen[k] for k in ('steer','throttle','brake')}
        self.rows[-1]['action'] = composed.tolist()
        return chosen

    def finish(self,snapshot,vehicle,reason,agent):
        if reason in ('collision','route_complete'):
            if self.pending is not None:
                self.complete_transition(read_state(snapshot,vehicle.id),np.zeros(287,np.float32),False,
                                         terminated=True,reason=reason)
        elif reason=='step_limit':
            from srunner.scenariomanager.timer import GameTime
            GameTime.on_carla_tick(snapshot.timestamp)
            agent.sensor_interface.set_snapshot_speed(snapshot.find(vehicle.id),snapshot.frame)
            control = agent()
            self.observe(agent.latest_reference_row,control,final_reason=reason)
            if self.pending is not None: raise RuntimeError('Time-limit transition bootstrap failed')

    def close(self,reason):
        if self.closed: return
        self.closed = True
        discarded = self.pending is not None
        self.pending = None
        if self.world is not None and not self.world.unchanged():raise RuntimeError('World model changed')
        if not self.evaluate:
            if reason in ('collision','route_complete','step_limit'):
                self.sac.config['episodes_completed'] = self.sac.config.get('episodes_completed',0)+1
            self.sac.save(self.output/'sac_last.pt')
        changed = any(not torch.equal(v,self.initial_actor[k]) for k,v in self.sac.actor.state_dict().items())
        summary = dict(reason=reason,evaluation=self.evaluate,transitions=self.count,enabled=self.gated,
                       lane_covered=self.covered,reward_terms=dict(self.terms),total_reward=sum(self.terms.values()),
                       steps=self.sac.steps,updates=self.sac.updates,new_updates=self.sac.updates-self.initial_updates,
                       actor_changed=changed,pending_discarded=discarded,reasons=dict(self.reasons),
                       world_model_used=self.world is not None,world_assisted_policy=self.sac.model_replay is not None or self.sac.config.get("world_assisted_policy",False),
                       collision_penalty=self.sac.config.get('collision_penalty',30.),
                       model_steps=self.sac.model_steps,new_model_steps=self.sac.model_steps-self.initial_model_steps,
                       model_rejections=dict(self.model_rejections),
                       world_model_unchanged=self.world.unchanged() if self.world is not None else None,
                       action_limits=LIMITS.tolist(),
                       normalization_source=self.sac.config['normalization_source'])
        (self.output/'sac_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        self.stream.close(); self.metrics.close()
        if self.model_stream is not None:self.model_stream.close()
        print('SAC SUMMARY '+json.dumps(summary),flush=True)
