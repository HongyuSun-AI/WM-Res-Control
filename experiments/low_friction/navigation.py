from collections import deque
import math
import random
import sys


def load_behavior_agent(carla_root):
    agents_root = carla_root / "PythonAPI" / "carla"
    if not (agents_root / "agents" / "navigation" / "behavior_agent.py").is_file():
        raise RuntimeError('Missing CARLA navigation: {}'.format(agents_root))
    sys.path.insert(0, str(agents_root))
    from agents.navigation.behavior_agent import BehaviorAgent
    return BehaviorAgent


def arrival_speed_limit(remaining_m, cruise_kmh):
    return min(cruise_kmh, max(3.0, 3.6 * math.sqrt(2.0 * max(0.0, remaining_m - 8.0))))


class Navigation:
    def __init__(self, world, vehicle, agent_type, dt, seed, destinations):
        import carla
        self.carla = carla
        self.world = world
        self.vehicle = vehicle
        self.dt = dt
        self.rng = random.Random(seed)
        self.limit = destinations
        self.completed = 0
        self.finished = False
        self.stopping = False
        self.idle_seconds = 0.0
        self.stop_seconds = 0.0
        self.stopped_seconds = 0.0
        self.events = deque()
        self.sensor = None
        self.map = world.get_map()
        self.spawns = self.map.get_spawn_points()
        self.agent = agent_type(vehicle, behavior="cautious", opt_dict={
            "dt": dt,
            "lateral_control_dict": {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": dt},
            "longitudinal_control_dict": {"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": dt},
        })
        self.cruise_kmh = self.agent._behavior.max_speed
        self.arrival_announced = False
        self.remaining_m = 0.0
        self.speed_cap_kmh = self.cruise_kmh
        self.choose_destination()
        try:
            blueprint = world.get_blueprint_library().find("sensor.other.collision")
            self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
            self.sensor.listen(self.events.append)
        except Exception:
            self.close()
            raise
        print("Navigation enabled: BehaviorAgent=cautious, seed={}, destinations={}".format(
            seed, destinations if destinations else "unlimited"), flush=True)

    def choose_destination(self):
        start = self.map.get_waypoint(self.vehicle.get_location())
        candidates = [i for i, spawn in enumerate(self.spawns)
                      if spawn.location.distance(self.vehicle.get_location()) >= 50.0]
        self.rng.shuffle(candidates)
        for index in candidates:
            end = self.map.get_waypoint(self.spawns[index].location)
            try:
                route = self.agent.trace_route(start, end)
            except (IndexError, KeyError):
                continue
            if len(route) < 2:
                continue
            route_end = route[-1][0].transform.location
            endpoint_gap = route_end.distance(self.spawns[index].location)
            if endpoint_gap > 5.0:
                print('Skipped spawn={}: endpoint_gap={:.1f}m'.format(
                    index, endpoint_gap), flush=True)
                continue
            self.agent.set_global_plan(route)
            self.agent._behavior.max_speed = self.cruise_kmh
            self.arrival_announced = False
            self.destination = self.spawns[index].location
            length = sum(a[0].transform.location.distance(b[0].transform.location)
                         for a, b in zip(route, route[1:]))
            self.route_info = dict(destination_spawn=index, length_m=length,
                                   destination=dict(x=self.destination.x, y=self.destination.y, z=self.destination.z),
                                   route_endpoint=dict(x=route_end.x, y=route_end.y, z=route_end.z),
                                   endpoint_gap_m=endpoint_gap)
            print('Route={} spawn={},points={} length={:.1f}m endpoint_gap={:.1f}m'.format(
                self.completed + 1, index, len(route), length, endpoint_gap), flush=True)
            return
        raise RuntimeError('Reachable destination >=50m required')

    def speed(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    def update_arrival_speed(self):
        plan = list(self.agent.get_local_planner().get_plan())
        points = [self.vehicle.get_location()]
        points.extend(waypoint.transform.location for waypoint, _ in plan)
        points.append(self.destination)
        self.remaining_m = sum(a.distance(b) for a, b in zip(points, points[1:]))
        self.speed_cap_kmh = arrival_speed_limit(self.remaining_m, self.cruise_kmh)
        self.agent._behavior.max_speed = self.speed_cap_kmh
        if self.speed_cap_kmh < self.cruise_kmh and not self.arrival_announced:
            self.arrival_announced = True
            print("Arrival slowdown: remaining_route={:.1f}m, speed_cap={:.1f}km/h".format(
                self.remaining_m, self.speed_cap_kmh), flush=True)

    def before_tick(self):
        if self.events:
            event = self.events.popleft()
            raise RuntimeError('Collision: frame={} actor={}'.format(
                event.frame, event.other_actor.type_id))
        if not self.stopping and self.agent.done():
            distance = self.vehicle.get_location().distance(self.destination)
            if distance > 10.0:
                raise RuntimeError('Planner exhausted: distance={:.1f}m route={}'.format(
                    distance, self.route_info))
            self.stopping = True
            print('Arrival braking: distance={:.1f}m'.format(distance), flush=True)
        if self.stopping:
            control = self.carla.VehicleControl(brake=1.0)
        else:
            self.update_arrival_speed()
            control = self.agent.run_step()
        control.manual_gear_shift = False
        return control

    def after_tick(self, count):
        if self.events:
            event = self.events.popleft()
            raise RuntimeError('Collision: frame={} actor={}'.format(
                event.frame, event.other_actor.type_id))
        speed = self.speed()
        if self.stopping:
            self.stop_seconds += self.dt
            self.stopped_seconds = self.stopped_seconds + self.dt if speed < 0.1 else 0.0
            if self.stop_seconds > 30.0:
                raise RuntimeError('Stopping timeout: 30s')
            if self.stopped_seconds >= 1.0:
                distance = self.vehicle.get_location().distance(self.destination)
                if distance > 10.0:
                    raise RuntimeError('Stopped {:.1f}m from destination'.format(distance))
                self.completed += 1
                print('Route {} complete: distance={:.1f}m'.format(
                    self.completed, distance), flush=True)
                if self.limit and self.completed >= self.limit:
                    self.finished = True
                    print("PASS: requested navigation destinations completed.", flush=True)
                else:
                    self.stopping = False
                    self.stop_seconds = self.stopped_seconds = self.idle_seconds = 0.0
                    self.choose_destination()
        else:
            self.idle_seconds = self.idle_seconds + self.dt if speed < 0.1 else 0.0
            if self.idle_seconds >= 120.0:
                raise RuntimeError('Stationary for 120 simulation seconds')
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        self.world.get_spectator().set_transform(self.carla.Transform(
            self.carla.Location(x=transform.location.x - 8 * forward.x,
                                y=transform.location.y - 8 * forward.y,
                                z=transform.location.z + 5),
            self.carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)))
        if count % 100 == 0:
            control = self.vehicle.get_control()
            print('NAV;step={};speed={:.1f}km/h;steer={:.3f};throttle={:.3f};brake={:.3f} destination_distance={:.1f}m remaining_route={:.1f}m speed_cap={:.1f}km/h'.format(
                count, speed * 3.6, control.steer, control.throttle, control.brake,
                transform.location.distance(self.destination), self.remaining_m, self.speed_cap_kmh), flush=True)

    def close(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
            finally:
                self.sensor.destroy()
                self.sensor = None
