#!/usr/bin/env python


from __future__ import print_function
import math
import os
import time

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.ros_base_agent import ROSBaseAgent

IS_BENCH2DRIVE = os.environ.get('SAVE_PATH', None)
if IS_BENCH2DRIVE:
    MAX_ALLOWED_RADIUS_SENSOR = 100.0
else:
    MAX_ALLOWED_RADIUS_SENSOR = 3.0

QUALIFIER_SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}
SENSORS_LIMITS = {
    'sensor.camera.rgb': 8,
    'sensor.lidar.ray_cast': 2,
    'sensor.other.radar': 4,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}
ALLOWED_SENSORS = SENSORS_LIMITS.keys()


class AgentError(Exception):

    def __init__(self, message):
        super(AgentError, self).__init__(message)

        
class TickRuntimeError(Exception):
    pass

class AgentWrapperFactory(object):

    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)


def validate_sensor_configuration(sensors, agent_track, selected_track):
    if Track(selected_track) != agent_track:
        raise SensorConfigurationInvalid('Incorrect track [{}]'.format(Track(selected_track)))

    sensor_count = {}
    sensor_ids = []

    for sensor in sensors:

        sensor_id = sensor['id']
        if sensor_id in sensor_ids:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
        else:
            sensor_ids.append(sensor_id)

        if agent_track == Track.SENSORS:
            if sensor['type'].startswith('sensor.opendrive_map'):
                raise SensorConfigurationInvalid('OpenDRIVE sensor prohibited for [{}]'.format(agent_track))

        if sensor['type'] not in ALLOWED_SENSORS:
            raise SensorConfigurationInvalid("Invalid sensor '{}' for [{}]".format(sensor['type'], agent_track))

        if 'x' in sensor and 'y' in sensor and 'z' in sensor:
            if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                raise SensorConfigurationInvalid(
                    "Invalid extrinsics: sensor='{}', radius<={}m".format(sensor['id'], MAX_ALLOWED_RADIUS_SENSOR))

        if sensor['type'] in sensor_count:
            sensor_count[sensor['type']] += 1
        else:
            sensor_count[sensor['type']] = 1

    if agent_track in (Track.SENSORS_QUALIFIER, Track.MAP_QUALIFIER):
        sensor_limits = QUALIFIER_SENSORS_LIMITS
    else:
        sensor_limits = SENSORS_LIMITS

    for sensor_type, max_instances_allowed in sensor_limits.items():
        if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
            raise SensorConfigurationInvalid(
                'Excess {}: maximum={}, requested={}'.format(sensor_type,
                                                                              max_instances_allowed,
                                                                              sensor_count[sensor_type]))


class AgentWrapper(object):

    _agent = None
    _sensors_list = []

    def __init__(self, agent):
        self._agent = agent
        self.wall_start = time.time()

        self._inf_latency_enabled = int(os.environ.get("INFERENCE_LATENCY_ENABLE", "0")) != 0
        self._inf_latency_ms = float(os.environ.get("INFERENCE_LATENCY_MS", "0"))
        self._sim_rate = float(os.environ.get("SIM_RATE", "20"))  
        self._warmup_steps = int(os.environ.get("WARMUP_STEPS", "20"))
        self._latency_mode = os.environ.get("INFERENCE_LATENCY_MODE", "fixed").lower()

        self._step = 0
        self._first_inference = True
        self._next_inference_step = -1

        self._next_control = None
        self._current_control = None
        self._last_measured_inference_time = 0.0

        self._default_control = carla.VehicleControl()
        self._default_control.steer = 0.0
        self._default_control.throttle = 0.0
        self._default_control.brake = 0.0

    def _get_sensor_data(self, input_data: dict, sensor_name: str):
        return input_data.get(sensor_name, input_data.get(sensor_name.upper(), None))

    def _sensor_data_ready(self) -> bool:
        try:
            input_data = self._agent.sensor_interface.get_data(GameTime.get_frame())

            speed_data = self._get_sensor_data(input_data, 'speed')
            speed_ok = speed_data is not None

            gps_data = self._get_sensor_data(input_data, 'gps')
            gps_ok = False
            if gps_data is not None:
                try:
                    val = gps_data[1]
                    if val is not None:
                        if isinstance(val, dict):
                            gps_ok = ('lat' in val and val['lat'] is not None and 'lon' in val and val['lon'] is not None)
                        else:
                            gps_ok = (hasattr(val, '__len__') and len(val) >= 2 and val[0] is not None and val[1] is not None)
                except Exception:
                    gps_ok = False

            return speed_ok and gps_ok
        except Exception:
            return False

    def _run_agent_with_timing(self):
        inference_start_time = time.time()
        control = self._agent()
        self._last_measured_inference_time = time.time() - inference_start_time
        return control

    def _call_measured_latency(self):
        if self._step <= self._warmup_steps or not self._sensor_data_ready():
            if self._step > self._warmup_steps and not self._sensor_data_ready():
                print(f"[INFERENCE LATENCY] step={self._step} WAITING_FOR_SENSOR_DATA", flush=True)
            return self._default_control

        if self._first_inference:
            self._next_control = self._run_agent_with_timing()
            self._next_inference_step = self._step + int(self._last_measured_inference_time * self._sim_rate)
            self._first_inference = False
            print(f"[INFERENCE LATENCY] step={self._step} FIRST_INFERENCE measured_inference_time={self._last_measured_inference_time:.6f}s "
                  f"next_control_step={self._next_inference_step} (result delayed)", flush=True)
            return self._default_control

        if self._step == self._next_inference_step:
            self._current_control = self._next_control
            self._next_control = self._run_agent_with_timing()
            self._next_inference_step = self._step + int(self._last_measured_inference_time * self._sim_rate)
            print(f"[INFERENCE LATENCY] step={self._step} SWITCH_BUFFER measured_inference_time={self._last_measured_inference_time:.6f}s "
                  f"next_control_step={self._next_inference_step}", flush=True)
            return self._current_control

        if self._current_control is not None:
            return self._current_control
        return self._default_control

    def __call__(self):
        self._step += 1

        wallclock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        game_time = GameTime.get_time()
        print(f"=== [Tick] step={self._step} -- Wallclock = {wallclock} -- System time = {time.time() - self.wall_start:.3f} -- Game time = {game_time:.3f} -- Ratio = {game_time / (time.time() - self.wall_start + 1e-6):.3f}x")

        if not self._inf_latency_enabled:
            control = self._agent()
            return control

        if self._latency_mode == "measured":
            return self._call_measured_latency()

        if self._inf_latency_ms <= 0:
            control = self._agent()
            return control

        delay_ticks = int(self._inf_latency_ms * self._sim_rate / 1000.0)

        if self._step <= self._warmup_steps or not self._sensor_data_ready():
            if self._step > self._warmup_steps and not self._sensor_data_ready():
                print(f"[INFERENCE LATENCY] step={self._step} WAITING_FOR_SENSOR_DATA", flush=True)
            return self._default_control

        if self._first_inference:
            self._next_control = self._agent()
            self._next_inference_step = self._step + delay_ticks
            self._first_inference = False
            print(f"[INFERENCE LATENCY] step={self._step} FIRST_INFERENCE delay_ticks={delay_ticks} "
                  f"next_control_step={self._next_inference_step} (result delayed)", flush=True)
            return self._default_control

        if self._step == self._next_inference_step:
            self._current_control = self._next_control
            self._next_control = self._agent()
            self._next_inference_step = self._step + delay_ticks
            print(f"[INFERENCE LATENCY] step={self._step} SWITCH_BUFFER delay_ticks={delay_ticks} "
                  f"next_control_step={self._next_inference_step}", flush=True)
            return self._current_control

        if self._current_control is not None:
            return self._current_control
        else:
            return self._default_control

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

            sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        for _ in range(10):
            world.tick()

    def cleanup(self):
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        CarlaDataProvider.get_world().tick()


class ROSAgentWrapper(AgentWrapper):

    SENSOR_TYPE_REMAPS = {
        "sensor.opendrive_map": "sensor.pseudo.opendrive_map",
        "sensor.speedometer": "sensor.pseudo.speedometer"
    }

    def __init__(self, agent):
        super(ROSAgentWrapper, self).__init__(agent)

    def _preprocess_sensor_spec(self, sensor_spec):
        type_, id_, sensor_transform, attributes = super(ROSAgentWrapper, self)._preprocess_sensor_spec(sensor_spec)
        new_type = self.SENSOR_TYPE_REMAPS.get(type_, type_)
        return new_type, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        for sensor_spec in self._agent.sensors():
            type_, id_, transform, attributes = self._preprocess_sensor_spec(sensor_spec)
            uid = self._agent.spawn_object(type_, id_, transform, attributes, attach_to=vehicle.id)
            self._sensors_list.append(uid)

        CarlaDataProvider.get_world().tick()

    def cleanup(self):
        for uid in self._sensors_list:
            self._agent.destroy_object(uid)
        self._sensors_list.clear()

        CarlaDataProvider.get_world().tick()
