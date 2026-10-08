#!/usr/bin/env python


from __future__ import print_function

import math
import re
import threading
from numpy import random
from six import iteritems

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


def calculate_velocity(actor):
    velocity_squared = actor.get_velocity().x**2
    velocity_squared += actor.get_velocity().y**2
    return math.sqrt(velocity_squared)


class CarlaDataProvider(object):


    _actor_velocity_map = {}
    _actor_location_map = {}
    _actor_transform_map = {}
    _traffic_light_map = {}
    _carla_actor_pool = {}
    _global_osc_parameters = {}
    _client = None
    _world = None
    _map = None
    _sync_flag = False
    _spawn_points = None
    _spawn_index = 0
    _blueprint_library = None
    _all_actors = None
    _ego_vehicle_route = None
    _traffic_manager_port = 8000
    _random_seed = 2000
    _rng = random.RandomState(_random_seed)
    _grp = None
    _runtime_init_flag = False
    _lock = threading.Lock()

    @staticmethod
    def register_actor(actor, transform=None):
        with CarlaDataProvider._lock:
            if actor in CarlaDataProvider._actor_velocity_map:
                raise KeyError(
                    "Vehicle '{}' already registered".format(actor.id))
            else:
                CarlaDataProvider._actor_velocity_map[actor] = 0.0
            if actor in CarlaDataProvider._actor_location_map:
                raise KeyError(
                    "Vehicle '{}' already registered".format(actor.id))
            elif transform:
                CarlaDataProvider._actor_location_map[actor] = transform.location
            else:
                CarlaDataProvider._actor_location_map[actor] = None

            if actor in CarlaDataProvider._actor_transform_map:
                raise KeyError(
                    "Vehicle '{}' already registered".format(actor.id))
            else:
                CarlaDataProvider._actor_transform_map[actor] = transform

    @staticmethod
    def update_osc_global_params(parameters):
        CarlaDataProvider._global_osc_parameters.update(parameters)

    @staticmethod
    def get_osc_global_param_value(ref):
        return CarlaDataProvider._global_osc_parameters.get(ref.replace("$", ""))

    @staticmethod
    def register_actors(actors, transforms=None):
        if transforms is None:
            transforms = [None] * len(actors)

        for actor, transform in zip(actors, transforms):
            CarlaDataProvider.register_actor(actor, transform)

    @staticmethod
    def on_carla_tick():
        with CarlaDataProvider._lock:
            for actor in CarlaDataProvider._actor_velocity_map:
                if actor is not None and actor.is_alive:
                    CarlaDataProvider._actor_velocity_map[actor] = calculate_velocity(actor)

            for actor in CarlaDataProvider._actor_location_map:
                if actor is not None and actor.is_alive:
                    CarlaDataProvider._actor_location_map[actor] = actor.get_location()

            for actor in CarlaDataProvider._actor_transform_map:
                if actor is not None and actor.is_alive:
                    CarlaDataProvider._actor_transform_map[actor] = actor.get_transform()

            world = CarlaDataProvider._world
            if world is None:
                print('WARNING: World unavailable')

            CarlaDataProvider._all_actors = None

    @staticmethod
    def get_velocity(actor):
        for key in CarlaDataProvider._actor_velocity_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_velocity_map[key]

        print('{}.get_velocity: missing {}' .format(__name__, actor))
        return 0.0

    @staticmethod
    def get_location(actor):
        for key in CarlaDataProvider._actor_location_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_location_map[key]

        print('{}.get_location: missing {}' .format(__name__, actor))
        return None

    @staticmethod
    def get_transform(actor):
        for key in CarlaDataProvider._actor_transform_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_transform_map[key]

        print('{}.get_transform: missing {}' .format(__name__, actor))
        return None

    @staticmethod
    def set_client(client):
        CarlaDataProvider._client = client

    @staticmethod
    def get_client():
        return CarlaDataProvider._client

    @staticmethod
    def set_world(world):
        CarlaDataProvider._world = world
        CarlaDataProvider._sync_flag = world.get_settings().synchronous_mode
        CarlaDataProvider._map = world.get_map()
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()
        CarlaDataProvider._grp = GlobalRoutePlanner(CarlaDataProvider._map, 2.0)
        CarlaDataProvider.generate_spawn_points()
        CarlaDataProvider.prepare_map()

    @staticmethod
    def get_world():
        return CarlaDataProvider._world

    @staticmethod
    def get_map(world=None):
        if CarlaDataProvider._map is None:
            if world is None:
                if CarlaDataProvider._world is None:
                    raise ValueError("Uninitialized class member 'world'")
                else:
                    CarlaDataProvider._map = CarlaDataProvider._world.get_map()
            else:
                CarlaDataProvider._map = world.get_map()

        return CarlaDataProvider._map

    @staticmethod
    def get_random_seed():
        return CarlaDataProvider._rng

    @staticmethod
    def get_global_route_planner():
        return CarlaDataProvider._grp

    @staticmethod
    def get_all_actors():
        if CarlaDataProvider._all_actors:
            return CarlaDataProvider._all_actors

        CarlaDataProvider._all_actors = CarlaDataProvider._world.get_actors()
        return CarlaDataProvider._all_actors

    @staticmethod
    def is_sync_mode():
        return CarlaDataProvider._sync_flag

    @staticmethod
    def set_runtime_init_mode(flag):
        CarlaDataProvider._runtime_init_flag = flag

    @staticmethod
    def is_runtime_init_mode():
        return CarlaDataProvider._runtime_init_flag

    @staticmethod
    def find_weather_presets():
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    @staticmethod
    def prepare_map():
        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = CarlaDataProvider._world.get_map()

        CarlaDataProvider._traffic_light_map.clear()
        for traffic_light in CarlaDataProvider._world.get_actors().filter('*traffic_light*'):
            if traffic_light not in list(CarlaDataProvider._traffic_light_map):
                CarlaDataProvider._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError(
                    "Traffic-light '{}' already registered".format(traffic_light.id))

    @staticmethod
    def annotate_trafficlight_in_group(traffic_light):
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        ref_location = CarlaDataProvider.get_trafficlight_trigger_location(traffic_light)
        ref_waypoint = CarlaDataProvider.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()

        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                target_location = CarlaDataProvider.get_trafficlight_trigger_location(target_tl)
                target_waypoint = CarlaDataProvider.get_map().get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360

                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)

        return dict_annotations

    @staticmethod
    def get_trafficlight_trigger_location(traffic_light):
        def rotate_point(point, angle):
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    @staticmethod
    def update_light_states(ego_light, annotations, states, freeze=False, timeout=1000000000):
        reset_params = []

        for state in states:
            relevant_lights = []
            if state == 'ego':
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            for light in relevant_lights:
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append({'light': light,
                                     'state': prev_state,
                                     'green_time': prev_green_time,
                                     'red_time': prev_red_time,
                                     'yellow_time': prev_yellow_time})

                light.set_state(states[state])
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params

    @staticmethod
    def reset_lights(reset_params):
        for param in reset_params:
            param['light'].set_state(param['state'])
            param['light'].set_green_time(param['green_time'])
            param['light'].set_red_time(param['red_time'])
            param['light'].set_yellow_time(param['yellow_time'])

    @staticmethod
    def get_next_traffic_light(actor, use_cached_location=True):

        if not use_cached_location:
            location = actor.get_transform().location
        else:
            location = CarlaDataProvider.get_location(actor)

        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = CarlaDataProvider._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    @staticmethod
    def generate_spawn_points():
        spawn_points = list(CarlaDataProvider.get_map(CarlaDataProvider._world).get_spawn_points())
        CarlaDataProvider._rng.shuffle(spawn_points)
        CarlaDataProvider._spawn_points = spawn_points
        CarlaDataProvider._spawn_index = 0

    @staticmethod
    def create_blueprint(model, rolename='scenario', color=None, actor_category="car", attribute_filter=None):
        def check_attribute_value(blueprint, name, value):
            if not blueprint.has_attribute(name):
                return False

            attribute_type = blueprint.get_attribute(key).type
            if attribute_type == carla.ActorAttributeType.Bool:
                return blueprint.get_attribute(name).as_bool() == value
            elif attribute_type == carla.ActorAttributeType.Int:
                return blueprint.get_attribute(name).as_int() == value
            elif attribute_type == carla.ActorAttributeType.Float:
                return blueprint.get_attribute(name).as_float() == value
            elif attribute_type == carla.ActorAttributeType.String:
                return blueprint.get_attribute(name).as_str() == value

            return False

        _actor_blueprint_categories = {
            'car': 'vehicle.tesla.model3',
            'van': 'vehicle.volkswagen.t2',
            'truck': 'vehicle.carlamotors.carlacola',
            'trailer': '',
            'semitrailer': '',
            'bus': 'vehicle.volkswagen.t2',
            'motorbike': 'vehicle.kawasaki.ninja',
            'bicycle': 'vehicle.diamondback.century',
            'train': '',
            'tram': '',
            'pedestrian': 'walker.pedestrian.0001',
        }

        try:
            blueprints = CarlaDataProvider._blueprint_library.filter(model)
            if attribute_filter is not None:
                for key, value in attribute_filter.items():
                    blueprints = [x for x in blueprints if check_attribute_value(x, key, value)]

            blueprint = CarlaDataProvider._rng.choice(blueprints)
        except ValueError:
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories[actor_category]
            if new_model != '':
                bp_filter = new_model
            print('Model {} unavailable; fallback {}'.format(model, new_model))
            blueprint = CarlaDataProvider._rng.choice(CarlaDataProvider._blueprint_library.filter(bp_filter))

        if color:
            if not blueprint.has_attribute('color'):
                print(
                    'Color {} unsupported: actor={}'.format(
                        color, blueprint.id))
            else:
                default_color_rgba = blueprint.get_attribute('color').as_color()
                default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
                try:
                    blueprint.set_attribute('color', color)
                except ValueError:
                    print('Color {} unsupported: actor={}, fallback={}'.format(
                        color, blueprint.id, default_color))
                    blueprint.set_attribute('color', default_color)
        else:
            if blueprint.has_attribute('color') and rolename != 'hero':
                color = CarlaDataProvider._rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', rolename)

        return blueprint

    @staticmethod
    def handle_actor_batch(batch, tick=True):
        sync_mode = CarlaDataProvider.is_sync_mode()
        actors = []

        if CarlaDataProvider._client:
            responses = CarlaDataProvider._client.apply_batch_sync(batch, sync_mode and tick)
        else:
            raise ValueError("Uninitialized class member 'client'")

        if not tick:
            pass
        elif CarlaDataProvider.is_runtime_init_mode():
            CarlaDataProvider._world.wait_for_tick()
        elif sync_mode:
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        actor_ids = [r.actor_id for r in responses if not r.error]
        for r in responses:
            if r.error:
                print('WARNING: Incomplete actor spawning')
                break
        actors = list(CarlaDataProvider._world.get_actors(actor_ids))
        return actors

    @staticmethod
    def request_new_actor(model, spawn_point, rolename='scenario', autopilot=False,
                          random_location=False, color=None, actor_category="car",
                          attribute_filter=None, tick=True):
        blueprint = CarlaDataProvider.create_blueprint(model, rolename, color, actor_category, attribute_filter)

        if random_location:
            actor = None
            while not actor:
                spawn_point = CarlaDataProvider._rng.choice(CarlaDataProvider._spawn_points)
                actor = CarlaDataProvider._world.try_spawn_actor(blueprint, spawn_point)

        else:
            z_offset = 0.2 if 'prop' not in model else 0

            _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
            _spawn_point.location.x = spawn_point.location.x
            _spawn_point.location.y = spawn_point.location.y
            _spawn_point.location.z = spawn_point.location.z + z_offset
            actor = CarlaDataProvider._world.try_spawn_actor(blueprint, _spawn_point)

        if actor is None:
            print('Spawn failed: actor={}, position={}'.format(model, spawn_point.location))
            return None

        if autopilot:
            if isinstance(actor, carla.Vehicle):
                actor.set_autopilot(autopilot, CarlaDataProvider._traffic_manager_port)
            else:
                print('WARNING: Autopilot requires vehicle')

        if not tick:
            pass
        elif CarlaDataProvider.is_runtime_init_mode():
            CarlaDataProvider._world.wait_for_tick()
        elif CarlaDataProvider.is_sync_mode():
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor, spawn_point)
        return actor

    @staticmethod
    def request_new_actors(actor_list, attribute_filter=None, tick=True):

        SpawnActor = carla.command.SpawnActor
        PhysicsCommand = carla.command.SetSimulatePhysics
        FutureActor = carla.command.FutureActor
        ApplyTransform = carla.command.ApplyTransform
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState

        batch = []

        CarlaDataProvider.generate_spawn_points()

        for actor in actor_list:

            blueprint = CarlaDataProvider.create_blueprint(
                actor.model, actor.rolename, actor.color, actor.category, attribute_filter)

            transform = actor.transform
            if actor.random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print('Spawn points exhausted')
                    break
                else:
                    _spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]
                    CarlaDataProvider._spawn_index += 1

            else:
                _spawn_point = carla.Transform()
                _spawn_point.rotation = transform.rotation
                _spawn_point.location.x = transform.location.x
                _spawn_point.location.y = transform.location.y
                if blueprint.has_tag('walker'):
                    map_name = CarlaDataProvider._map.name.split("/")[-1]
                    if not map_name.startswith('OpenDrive'):
                        _spawn_point.location.z = transform.location.z + 0.2
                    else:
                        _spawn_point.location.z = transform.location.z + 0.8
                else:
                    _spawn_point.location.z = transform.location.z + 0.2

            command = SpawnActor(blueprint, _spawn_point)
            command.then(SetAutopilot(FutureActor, actor.autopilot, CarlaDataProvider._traffic_manager_port))

            if actor.args is not None and 'physics' in actor.args and actor.args['physics'] == "off":
                command.then(ApplyTransform(FutureActor, _spawn_point)).then(PhysicsCommand(FutureActor, False))
            elif actor.category == 'misc':
                command.then(PhysicsCommand(FutureActor, True))
            if actor.args is not None and 'lights' in actor.args and actor.args['lights'] == "on":
                command.then(SetVehicleLightState(FutureActor, carla.VehicleLightState.All))

            batch.append(command)

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor, _spawn_point)

        return actors

    @staticmethod
    def request_new_batch_actors(model, amount, spawn_points, autopilot=False,
                                 random_location=False, rolename='scenario',
                                 attribute_filter=None, tick=True):

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        CarlaDataProvider.generate_spawn_points()

        batch = []

        for i in range(amount):
            blueprint = CarlaDataProvider.create_blueprint(model, rolename, attribute_filter=attribute_filter)

            if random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print('Spawn exhausted: {}/{} actors'.format(i + 1, amount))
                    break
                else:
                    spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]
                    CarlaDataProvider._spawn_index += 1
            else:
                try:
                    spawn_point = spawn_points[i]
                except IndexError:
                    print('Insufficient spawn points')
                    break

            if spawn_point:
                batch.append(SpawnActor(blueprint, spawn_point).then(
                    SetAutopilot(FutureActor, autopilot, CarlaDataProvider._traffic_manager_port)))

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor, command in zip(actors, batch):
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor, command.transform)

        return actors

    @staticmethod
    def get_actors():
        return iteritems(CarlaDataProvider._carla_actor_pool)

    @staticmethod
    def actor_id_exists(actor_id):
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return True

        return False

    @staticmethod
    def get_hero_actor():
        for actor_id in CarlaDataProvider._carla_actor_pool:
            if CarlaDataProvider._carla_actor_pool[actor_id].attributes['role_name'] == 'hero':
                return CarlaDataProvider._carla_actor_pool[actor_id]
        return None

    @staticmethod
    def get_actor_by_id(actor_id):
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return CarlaDataProvider._carla_actor_pool[actor_id]

        print("Non-existing actor id {}".format(actor_id))
        return None

    @staticmethod
    def remove_actor_by_id(actor_id):
        if actor_id in CarlaDataProvider._carla_actor_pool:
            CarlaDataProvider._carla_actor_pool[actor_id].destroy()
            CarlaDataProvider._carla_actor_pool[actor_id] = None
            CarlaDataProvider._carla_actor_pool.pop(actor_id)
        else:
            print('Missing actor: {}'.format(actor_id))

    @staticmethod
    def remove_actors_in_surrounding(location, distance):
        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            if CarlaDataProvider._carla_actor_pool[actor_id].get_location().distance(location) < distance:
                CarlaDataProvider._carla_actor_pool[actor_id].destroy()
                CarlaDataProvider._carla_actor_pool.pop(actor_id)

        CarlaDataProvider._carla_actor_pool = dict({k: v for k, v in CarlaDataProvider._carla_actor_pool.items() if v})

    @staticmethod
    def get_traffic_manager_port():
        return CarlaDataProvider._traffic_manager_port

    @staticmethod
    def set_traffic_manager_port(tm_port):
        CarlaDataProvider._traffic_manager_port = tm_port

    @staticmethod
    def cleanup():
        DestroyActor = carla.command.DestroyActor
        batch = []

        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            actor = CarlaDataProvider._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                batch.append(DestroyActor(actor))

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._traffic_light_map.clear()
        CarlaDataProvider._map = None
        CarlaDataProvider._world = None
        CarlaDataProvider._sync_flag = False
        CarlaDataProvider._ego_vehicle_route = None
        CarlaDataProvider._all_actors = None
        CarlaDataProvider._carla_actor_pool = {}
        CarlaDataProvider._client = None
        CarlaDataProvider._spawn_points = None
        CarlaDataProvider._spawn_index = 0
        CarlaDataProvider._rng = random.RandomState(CarlaDataProvider._random_seed)
        CarlaDataProvider._grp = None
        CarlaDataProvider._runtime_init_flag = False
