#!/usr/bin/env python


from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface


class Track(Enum):

    SENSORS = 'SENSORS'
    MAP = 'MAP'
    SENSORS_QUALIFIER = 'SENSORS_QUALIFIER'
    MAP_QUALIFIER = 'MAP_QUALIFIER'


class AutonomousAgent(object):


    def __init__(self, carla_host, carla_port, debug=False):
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None

        self.sensor_interface = SensorInterface()

        self.wallclock_t0 = None

        self.get_hero()

    def setup(self, path_to_conf_file):
        pass

    def sensors(self):
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        pass

    def __call__(self):
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp/wallclock_diff

        print('INFERENCE wallclock={} system={} game={} ratio={}x'.format(
            str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')), flush=True)

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    @staticmethod
    def get_ros_version():
        return -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._global_plan_gps_full = global_plan_gps
    
    def get_hero(self):
        hero_actor = None
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        for actor in CarlaDataProvider.get_world().get_actors():
            if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                hero_actor = actor
                break
        self.hero_actor = hero_actor
    
    def get_metric_info(self):
        
        def vector2list(vector, rotation=False):
            if rotation:
                return [vector.roll, vector.pitch, vector.yaw]
            else:
                return [vector.x, vector.y, vector.z]

        output = {}
        output['acceleration'] = vector2list(self.hero_actor.get_acceleration())
        output['angular_velocity'] = vector2list(self.hero_actor.get_angular_velocity())
        output['forward_vector'] = vector2list(self.hero_actor.get_transform().get_forward_vector())
        output['right_vector'] = vector2list(self.hero_actor.get_transform().get_right_vector())
        output['location'] = vector2list(self.hero_actor.get_transform().location)
        output['rotation'] = vector2list(self.hero_actor.get_transform().rotation, rotation=True)
        return output