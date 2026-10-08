import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def wait_for_spawn(world, vehicle, max_ticks=3):
    for _ in range(max_ticks):
        world.tick()
        snapshot = world.get_snapshot()
        if snapshot.find(vehicle.id) is not None and vehicle.is_alive:
            print('SPAWN confirmed actor={} frame={} role={}'.format(
                vehicle.id, snapshot.frame, vehicle.attributes.get('role_name')), flush=True)
            return
    raise RuntimeError('Spawn timeout: ticks={} actor={},alive={},role={}'.format(
        max_ticks, vehicle.id, vehicle.is_alive, vehicle.attributes.get('role_name')))


def configure_friction(world, vehicle, requested):
    if requested not in (.5, 3.5):
        raise ValueError('Supported friction values: 0.5, 3.5')
    physics=vehicle.get_physics_control()
    wheels=physics.wheels
    original=[float(w.tire_friction) for w in wheels]
    if len(original)!=4 or any(not math.isfinite(v) or abs(v-3.5)>1e-5 for v in original):
        raise RuntimeError('Expected friction=3.5, got {}'.format(original))
    if requested != 3.5:
        for wheel in wheels: wheel.tire_friction=requested
        physics.wheels=wheels
        vehicle.apply_physics_control(physics)
        world.tick()
    actual=[float(w.tire_friction) for w in vehicle.get_physics_control().wheels]
    if len(actual)!=4 or any(not math.isfinite(v) or abs(v-requested)>1e-5 for v in actual):
        raise RuntimeError('Friction readback mismatch: requested={} actual={}'.format(requested,actual))
    print('PASS: friction requested={} original={} actual={}'.format(requested,original,actual),flush=True)
    return original,actual


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--weather', choices=('ClearNoon','MidRainyNoon','HardRainNoon'), default='ClearNoon')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--route-id',default='24785')
    parser.add_argument('--start-fraction',type=float,default=0.,help='Route start fraction: 0..0.8')
    parser.add_argument('--tire-friction',type=float,choices=(.5,3.5),default=3.5,
                        help='Wheel friction; default 3.5')
    parser.add_argument('--carla-root',type=Path,default=Path(os.environ.get('CARLA_ROOT', '../CARLA_0.9.15')))
    parser.add_argument('--checkpoint',type=Path,default=Path('demo/models/tcp.ckpt.gz'))
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--shadow-policy',type=Path,help='Shadow residual suggestions')
    parser.add_argument('--sac-checkpoint',type=Path,help='Online SAC episode')
    parser.add_argument('--sac-evaluate',action='store_true',help='Deterministic SAC inference')
    parser.add_argument('--execute-residual',action='store_true',help='Live residual control using shadow-policy')
    parser.add_argument('--lane-metrics',action='store_true',help='Measure planned-lane footprint')
    parser.add_argument('--video',action='store_true',help='Record chase camera')
    parser.add_argument('--target-speed-kmh',type=float,help='Fixed-speed PID; TCP steering')
    parser.add_argument('--stop-at-route-end',action='store_true',help='Slow down at route end')
    parser.add_argument('--uncapped-tcp',action='store_true',help='Disable extra throttle cap')
    parser.add_argument('--extend-route-m',type=float,default=0.,help='Extend final lane')
    parser.add_argument('--replay-controls',type=Path,help='Executed-control sequence')
    args=parser.parse_args()
    if args.sac_evaluate and not args.sac_checkpoint:parser.error('SAC evaluation requires checkpoint')
    if args.sac_checkpoint:
        if args.shadow_policy or args.execute_residual or args.replay_controls:parser.error('Conflicting controllers')
        if not args.sac_checkpoint.is_file() or args.tire_friction!=.5:parser.error('SAC requires checkpoint, friction=0.5')
        if not args.stop_at_route_end:parser.error('SAC requires shared route-end stopping')
        args.lane_metrics=True
    if args.stop_at_route_end and args.target_speed_kmh is None and not args.uncapped_tcp:parser.error('Route-end speed mode required')
    if args.uncapped_tcp and (args.target_speed_kmh is not None or args.replay_controls):parser.error('Conflicting speed modes')
    if not math.isfinite(args.extend_route_m) or not 0<=args.extend_route_m<=500:parser.error('Route extension: 0..500m')
    if args.target_speed_kmh is not None and (not math.isfinite(args.target_speed_kmh) or not 0<args.target_speed_kmh<=40 or args.replay_controls):
        parser.error('Invalid speed or replay combination')
    replay=None
    if args.replay_controls:
        if args.shadow_policy:parser.error('Conflicting replay and residual modes')
        replay=json.loads(args.replay_controls.read_text())
        if len(replay)!=args.steps:parser.error('Replay length must equal steps')
        for control in replay:
            if set(control)!=set(('steer','throttle','brake')) or any(not math.isfinite(control[k]) for k in control):parser.error('Invalid replay control')
            if not -1<=control['steer']<=1 or not 0<=control['throttle']<=1 or not 0<=control['brake']<=1 or min(control['throttle'],control['brake'])>0:parser.error('Replay control bounds')
    if args.execute_residual and not args.shadow_policy:parser.error('execute-residual requires shadow-policy')
    if not math.isfinite(args.start_fraction) or not 0<=args.start_fraction<=.8:parser.error('start-fraction must be 0..0.8')
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    if args.steps<41 or args.output.exists(): parser.error('Invalid steps or output')
    if not args.checkpoint.is_file(): parser.error('Missing checkpoint')
    if args.shadow_policy and (not args.shadow_policy.is_file() or args.tire_friction!=.5):
        parser.error('Shadow requires checkpoint, friction=0.5')
    routes=ET.parse(root/'leaderboard/data/town04.xml').getroot().findall('route')
    selected=[r for r in routes if r.get('id')==args.route_id]
    if len(selected)!=1:parser.error('Unknown/ambiguous Town04 route ID')
    xml=selected[0]
    for path in (root,root/'leaderboard',root/'scenario_runner',args.carla_root/'PythonAPI/carla'):
        sys.path.insert(0,str(path))
    os.environ['IS_BENCH2DRIVE']='True'
    os.environ['PLANNER_TYPE']='only_traj'
    os.environ['TCP_REFERENCE_DIR']=str(args.output.resolve())
    os.environ.pop('SAVE_PATH',None)
    os.environ['INFERENCE_LATENCY_ENABLE']='0'
    import carla
    import torch
    from experiments.low_friction.tcp_reference_agent import TCPReferenceAgent
    from experiments.low_friction.environment_runtime import apply_control_confirmed
    from experiments.low_friction.control_readback import control_values, check_readback
    from experiments.low_friction.reference_inputs import ExactFrameInputs, gps_to_xy
    from leaderboard.autoagents.agent_wrapper import AgentWrapper
    from leaderboard.utils.route_manipulation import interpolate_trajectory, _get_latlon_ref
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime
    if not torch.cuda.is_available(): raise RuntimeError('Existing TCP agent requires CUDA')
    if args.check_only:
        print('PASS: imports, checkpoint, CUDA')
        return
    client=carla.Client('127.0.0.1',args.port); client.set_timeout(60)
    world=client.get_world()
    if world.get_settings().synchronous_mode: raise RuntimeError('Server already synchronous')
    actors=world.get_actors()
    occupied = [a for a in actors if a.type_id.startswith(
        ('vehicle.', 'walker.pedestrian.', 'sensor.', 'static.trigger.friction'))]
    if occupied:
        for actor in occupied:
            parent = actor.parent
            print('EXISTING actor={} type={} role={} parent={}'.format(
                actor.id, actor.type_id, actor.attributes.get('role_name'),
                parent.id if parent is not None else None), flush=True)
        raise RuntimeError('Server contains actors')
    if world.get_map().name.rsplit('/',1)[-1]!='Town04': world=client.load_world('Town04')
    old_settings=world.get_settings(); old_weather=world.get_weather()
    vehicle=agent=wrapper=collision=None
    extension_actual=0.; events=[]; collision_details=[]; completed=0; reason='error'; original_friction=None; friction=None
    args.output.mkdir(parents=True,exist_ok=False)
    readbacks=(args.output/'control_readback.jsonl').open('x',buffering=1)
    pending=None;readback_count=0
    shadow=None;demo_camera=None
    lane_monitor=None;lane_stream=None
    def confirm_previous(snapshot):
        nonlocal pending,readback_count
        if pending is None:return
        result=check_readback(pending[0],snapshot.frame,pending[1],control_values(vehicle.get_control()))
        readbacks.write(json.dumps(result,allow_nan=False)+'\n');readback_count+=1;pending=None
        if not result['matched']:raise RuntimeError('Next-frame control readback mismatch: '+str(result))
        if shadow is not None:shadow.add_readback(result)
    try:
        if args.shadow_policy:
            from shadow_residual import ShadowResidual
            shadow=ShadowResidual(args.shadow_policy,args.output,world.get_map(),carla,execute=args.execute_residual)
        if args.sac_checkpoint:
            from sac_online import SACOnline
            shadow=SACOnline(args.sac_checkpoint,args.output,world.get_map(),carla,evaluate=args.sac_evaluate)
        settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=.05
        settings.substepping=True; settings.max_substep_delta_time=.01; settings.max_substeps=10
        world.apply_settings(settings); world.set_weather(getattr(carla.WeatherParameters,args.weather))
        weather_fields = ('cloudiness','precipitation','precipitation_deposits','wind_intensity','sun_azimuth_angle','sun_altitude_angle','fog_density','fog_distance','wetness','fog_falloff')
        weather_actual = world.get_weather()
        weather_expected = getattr(carla.WeatherParameters,args.weather)
        weather_values = {k: float(getattr(weather_actual,k)) for k in weather_fields}
        if any(abs(weather_values[k]-getattr(weather_expected,k))>1e-4 for k in weather_fields):
            raise RuntimeError('Weather readback mismatch')
        (args.output/'weather_readback.json').write_text(json.dumps(dict(preset=args.weather,actual=weather_values,matched=True),indent=2))
        CarlaDataProvider.set_client(client); CarlaDataProvider.set_world(world); GameTime.restart()
        points=[carla.Location(x=float(p.get('x')),y=float(p.get('y')),z=float(p.get('z'))) for p in xml.find('waypoints')]
        extension_actual=0.
        if args.extend_route_m:
            end=world.get_map().get_waypoint(points[-1])
            for _ in range(int(args.extend_route_m/2)):
                options=end.next(2.)
                if len(options)!=1 or options[0].is_junction:break
                nxt=options[0];extension_actual+=end.transform.location.distance(nxt.transform.location);end=nxt
            if extension_actual<20:raise RuntimeError('Insufficient unambiguous route extension')
            points.append(end.transform.location)
        gps,route=interpolate_trajectory(points)
        if len(route)<2: raise RuntimeError('Empty route')
        if args.start_fraction:
            distances=[0.]
            for a,b in zip(route,route[1:]):distances.append(distances[-1]+a[0].location.distance(b[0].location))
            target=distances[-1]*args.start_fraction
            offset=next(i for i,d in enumerate(distances) if d>=target)
            for i in range(offset,min(offset+51,len(route)-1)):
                waypoint=world.get_map().get_waypoint(route[i][0].location)
                if waypoint is not None and not waypoint.is_junction:
                    offset=i;break
            gps=gps[offset:];route=route[offset:]
        from planned_lane import serialize_route,PlannedLane
        planned=serialize_route(world.get_map(),route)
        (args.output/'planned_route.json').write_text(json.dumps(planned))
        if shadow is not None:shadow.planned_lane=PlannedLane(planned)
        spawn=carla.Transform(route[0][0].location+carla.Location(z=.5),route[0][0].rotation)
        bp=world.get_blueprint_library().find('vehicle.lincoln.mkz_2020'); bp.set_attribute('role_name','hero')
        vehicle=world.spawn_actor(bp,spawn)
        apply_control_confirmed(client,vehicle,carla.VehicleControl(brake=1))
        wait_for_spawn(world,vehicle)
        original_friction,friction=configure_friction(world,vehicle,args.tire_friction)
        collision=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=vehicle)
        def record_collision(event):
            events.append(event.frame)
            other=event.other_actor;impulse=event.normal_impulse;location=event.transform.location
            collision_details.append(dict(frame=event.frame,timestamp=event.timestamp,
                other_actor_id=other.id,other_actor_type=other.type_id,
                location=dict(x=location.x,y=location.y,z=location.z),
                normal_impulse=dict(x=impulse.x,y=impulse.y,z=impulse.z)))
        collision.listen(record_collision)
        agent=TCPReferenceAgent('127.0.0.1',args.port)
        agent.bind_reference_vehicle(vehicle)
        agent.sensor_interface=ExactFrameInputs()
        agent.reference_georef=_get_latlon_ref(world)
        for x,y in ((0.,0.),(100.,0.),(0.,100.)):
            geo=world.get_map().transform_to_geolocation(carla.Location(x=x,y=y,z=0.))
            projected=gps_to_xy([geo.latitude,geo.longitude],*agent.reference_georef)
            if max(abs(projected[0]-x),abs(projected[1]-y))>1e-3:
                raise RuntimeError('Map GNSS projection self-check failed')
        print('PASS: map GNSS projection; reference={}'.format(agent.reference_georef),flush=True)
        agent.setup(str(args.checkpoint.resolve())+'+reference_short')
        if args.target_speed_kmh is not None:agent.experiment_target_speed_kmh=args.target_speed_kmh
        if args.uncapped_tcp:agent.experiment_uncapped_tcp=True
        if args.stop_at_route_end:
            from route_speed import RouteSpeedPlan
            agent.experiment_route_speed=RouteSpeedPlan(planned,args.target_speed_kmh if args.target_speed_kmh is not None else 10000.)
        agent.set_global_plan(gps,route)
        wrapper=AgentWrapper(agent); wrapper.setup_sensors(vehicle)
        if args.lane_metrics:
            from route_lane_monitor import RouteLaneMonitor
            lane_monitor=RouteLaneMonitor(world.get_map(),route,vehicle.bounding_box)
            box=vehicle.bounding_box
            (args.output/'vehicle_box.json').write_text(json.dumps(dict(
                extent={k:getattr(box.extent,k) for k in ('x','y','z')},
                location={k:getattr(box.location,k) for k in ('x','y','z')},
                rotation={k:getattr(box.rotation,k) for k in ('pitch','yaw','roll')}),indent=2))
            lane_stream=(args.output/'lane_metrics.jsonl').open('x',buffering=1)
            if args.sac_checkpoint:
                shadow.lane_monitor=RouteLaneMonitor(world.get_map(),route,vehicle.bounding_box)
        if args.video:
            from demo_camera import DemoCamera
            demo_camera=DemoCamera(world,vehicle,carla,args.output)
        print('RECORD route={} steps={} friction={} output={}'.format(xml.get('id'),args.steps,friction,args.output.resolve()),flush=True)
        for step in range(args.steps):
            world.tick(); snapshot=world.get_snapshot(); GameTime.on_carla_tick(snapshot.timestamp)
            confirm_previous(snapshot)
            if events:
                reason='collision';break
            actor_snapshot=snapshot.find(vehicle.id)
            if actor_snapshot is None: raise RuntimeError('Hero absent from current snapshot')
            agent.sensor_interface.set_snapshot_speed(actor_snapshot,snapshot.frame)
            if demo_camera is not None:demo_camera.capture(snapshot.frame)
            control=agent()
            if replay is not None:control=carla.VehicleControl(**replay[step])
            if lane_monitor is not None:
                row=getattr(agent,'latest_reference_row',None)
                if row is not None:
                    lane_stream.write(json.dumps(dict(frame=snapshot.frame,**lane_monitor.observe(row['state'])),allow_nan=False)+'\n')
            if shadow is not None:
                selected=shadow.observe(getattr(agent,'latest_reference_row',None),control)
                if args.execute_residual or args.sac_checkpoint:control=carla.VehicleControl(**selected)
            apply_control_confirmed(client,vehicle,control)
            pending=(snapshot.frame,control_values(control))
            completed=step+1
            current=getattr(agent,'latest_reference_row',None)
            if args.stop_at_route_end and current is not None and current.get('route_speed',{}).get('stopped'):
                reason='route_complete';print('ARRIVAL stopped before route endpoint',flush=True);break
            if completed%100==0: print('REFERENCE steps={}/{}'.format(completed,args.steps),flush=True)
        if reason!='collision':
            world.tick();snapshot=world.get_snapshot();confirm_previous(snapshot)
        if events:reason='collision'
        if reason not in ('route_complete','collision'):reason='step_limit'
        if args.sac_checkpoint:shadow.finish(snapshot,vehicle,reason,agent)
        print('CONTROL READBACK: matched={}/{}; tolerance=1e-5'.format(readback_count,completed),flush=True)
        print('PASS: bounded recording finished',flush=True)
    except KeyboardInterrupt:
        reason='user_interrupt'
        print('Interrupted; preserving recorded prefix',flush=True)
    finally:
        original_error = sys.exc_info()[1]
        errors=[]
        def cleanup(label,fn):
            try: fn()
            except Exception as exc: errors.append(label+': '+str(exc))
        if vehicle is not None and vehicle.is_alive: cleanup('brake',lambda: apply_control_confirmed(client,vehicle,carla.VehicleControl(brake=1)))
        if wrapper is not None: cleanup('sensors',wrapper.cleanup)
        if demo_camera is not None:cleanup('demo_camera',demo_camera.close)
        if collision is not None and collision.is_alive:
            cleanup('collision.stop',collision.stop); cleanup('collision.destroy',collision.destroy)
        if agent is not None: cleanup('agent',agent.destroy)
        if vehicle is not None and vehicle.is_alive: cleanup('vehicle',vehicle.destroy)
        cleanup('weather',lambda: world.set_weather(old_weather))
        cleanup('settings',lambda: world.apply_settings(old_settings))
        readbacks.close()
        if lane_stream is not None:lane_stream.close()
        if shadow is not None:cleanup('shadow',lambda:shadow.close('error' if args.sac_checkpoint and original_error is not None else reason))
        (args.output/'session.json').write_text(json.dumps(dict(reason=reason,steps=completed,collision_frames=events,collision_details=collision_details,
            cleanup_errors=errors,original_error=repr(original_error) if original_error else None,
            checkpoint=str(args.checkpoint.resolve()),requested_tire_friction=args.tire_friction,
            original_wheel_friction=original_friction,actual_wheel_friction=friction,
            route_id=args.route_id,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            start_fraction=args.start_fraction,weather=args.weather,
            target_speed_kmh=args.target_speed_kmh,
            uncapped_tcp=args.uncapped_tcp,route_extension_m=extension_actual,
            stop_at_route_end=args.stop_at_route_end,
            base_control_mode='TCP_without_extra_low_speed_throttle_cap' if args.uncapped_tcp else ('frozen_TCP_steering+fixed_speed_PID' if args.target_speed_kmh is not None else 'original_TCP'),
            control_mode='TCP+SAC' if args.sac_checkpoint else ('diagnostic_replay' if replay is not None else ('TCP+residual' if args.execute_residual else 'TCP')),
            sac_checkpoint=str(args.sac_checkpoint.resolve()) if args.sac_checkpoint else None,
            sac_evaluate=args.sac_evaluate,
            replay_controls_sha256=hashlib.sha256(args.replay_controls.read_bytes()).hexdigest() if replay is not None else None,
            control_readback_count=readback_count,control_readback_complete=reason in ('step_limit','route_complete','collision') and readback_count==completed,
            note='TCP proposals; executed-control readbacks'),indent=2))
        print('Cleanup finished; errors={}'.format(errors),flush=True)
        if errors and original_error is None: raise RuntimeError('Cleanup incomplete: '+str(errors))


if __name__=='__main__': main()
