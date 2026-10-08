import copy
import numpy as np
import os
from threading import Thread
from collections import deque
from typing import Dict, Tuple, Any, Optional, List

from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from .config import SensorConfig
from .delay_buffer import DelayBuffer, SensorData
from .temporal_blender import TemporalBlender
from .robustness_processor import RobustnessProcessor
from .compat import (
    CallBack,
    BaseReader,
    SpeedometerReader,
    OpenDriveMapReader,
    GenericMeasurement
)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class SensorInterface(object):

    CAMERA_PREFIXES = ("sensor.camera",)
    EGO_TYPES = ("sensor.other.gnss", "sensor.other.imu", "sensor.speedometer")
    OPENDRIVE_TYPE = "sensor.opendrive_map"

    def __init__(self):
        self._config = SensorConfig.from_env()

        self._buffer = DelayBuffer(self._config)
        self._blender = TemporalBlender(self._config)
        self._robustness = RobustnessProcessor(self._config)

        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._opendrive_tag = None

        self._first_cam_frame: Optional[int] = None
        self._camera_cache: Dict[str, Optional[Tuple[int, Any]]] = {}

        self._speed_cache: Optional[Tuple[int, Any]] = None
        self._speed_sensor_tag: Optional[str] = None

        self._first_tick = True

        self._sensor_tags = set()
        self._camera_tags = set()
        self._non_camera_tags = set()
        self._sensor_groups = {}

    def _get_sensor(self, data_dict: dict, sensor_name: str, default=None):
        return data_dict.get(sensor_name, data_dict.get(sensor_name.upper(), default))

    def _classify_sensor(self, sensor_type: str) -> str:
        if sensor_type.startswith(self.CAMERA_PREFIXES):
            return "cam"
        if sensor_type == self.OPENDRIVE_TYPE:
            return "opendrive"
        if sensor_type in self.EGO_TYPES:
            return "ego"
        return "other"

    def _is_sample_frame(self, frame: int) -> bool:
        if self._first_cam_frame is None:
            return True

        return (frame - self._first_cam_frame) % self._config.rgb_frame_interval == 0


    def register_sensor(self, tag: str, sensor_type: str, sensor) -> None:
        if tag in self._sensors:
            raise ValueError(f"Duplicated sensor tag [{tag}]")

        group = self._classify_sensor(sensor_type)

        self._sensors[tag] = {
            'tag': tag,
            'type': sensor_type,
            'obj': sensor,
            'group': group
        }

        if group == 'opendrive':
            self._opendrive_tag = tag

        if sensor_type == 'sensor.speedometer':
            self._speed_sensor_tag = tag

        self._buffer.register_sensor(tag, group)

        self._sensor_tags.add(tag)
        self._sensor_groups[tag] = group

        if group == 'cam':
            self._camera_tags.add(tag)
        else:
            self._non_camera_tags.add(tag)

    def update_sensor(self, tag: str, data: Any, frame: int) -> None:
        if tag not in self._sensors:
            raise SensorConfigurationInvalid(f"Missing sensor [{tag}]")

        self._buffer.append(tag, frame, data)

    def get_data(self, frame: int) -> Dict[str, Tuple[int, Any]]:
        try:
            if self._first_tick:
                result = self._first_tick_collect(frame)
            else:
                result = self._normal_collect(frame)

            if self._config.robustness_enable:
                for tag in result:
                    group = self._sensor_groups.get(tag, 'other')
                    if self._robustness.should_process(tag, group):
                        f, data = result[tag]
                        processed_f, processed_data = self._robustness.process_sensor_data(
                            tag, group, f, data
                        )
                        result[tag] = (processed_f, processed_data)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise SensorReceivedNoData(f"Error collecting sensor data: {e}")

    def _first_tick_collect(self, frame: int) -> Dict[str, Tuple[int, Any]]:
        import time
        timeout = 300.0
        start_time = time.time()

        first_cam_tag = None
        for tag in self._camera_tags:
            if tag in self._buffer._buffers:
                first_cam_tag = tag
                break

        if first_cam_tag is None:
            first_cam_tag = list(self._camera_tags)[0] if self._camera_tags else None

        latest = None
        if first_cam_tag:
            while time.time() - start_time < timeout:
                latest = self._buffer.get_cache(first_cam_tag)
                if latest is not None:
                    self._first_cam_frame = latest[0]
                    break
                time.sleep(0.01)

        if first_cam_tag and latest is None:
            raise SensorReceivedNoData(f"Camera timeout ({first_cam_tag})")

        speed_tag = self._speed_sensor_tag if self._speed_sensor_tag else 'SPEED'
        required_sensors = self._sensor_tags - {speed_tag}
        data_dict = {}

        while time.time() - start_time < timeout:
            data_dict = {}
            for tag in required_sensors:
                sensor_data = self._buffer.get_latest(tag, frame)
                if sensor_data is not None:
                    if sensor_data.age <= 50:
                        data_dict[tag] = (sensor_data.frame, sensor_data.data)

            if len(data_dict) >= len(required_sensors):
                break

            time.sleep(0.01)

        if len(data_dict) < len(required_sensors):
            missing = required_sensors - set(data_dict.keys())
            raise SensorReceivedNoData(
                f"Sensor timeout: {missing}"
            )

        for tag in self._camera_tags:
            if tag in data_dict:
                self._camera_cache[tag] = data_dict[tag]

        data_dict = self._handle_speed_sensor(frame, data_dict)

        data_dict = self._ensure_all_sensors_present(data_dict, frame)

        self._first_tick = False
        return data_dict

    def _normal_collect(self, frame: int) -> Dict[str, Tuple[int, Any]]:
        raw_dict = {}

        is_sample_frame = self._is_sample_frame(frame)

        speed_tag = self._speed_sensor_tag if self._speed_sensor_tag else 'SPEED'
        if is_sample_frame:
            required_sensors = self._sensor_tags - {speed_tag}
        else:
            required_sensors = self._non_camera_tags.copy()

        for tag in required_sensors:
            sensor_data = self._buffer.get_latest(tag, frame)
            if sensor_data is not None:
                if sensor_data.frame == frame:
                    raw_dict[tag] = (sensor_data.frame, sensor_data.data)
                elif sensor_data.age <= 10:
                    raw_dict[tag] = (sensor_data.frame, sensor_data.data)
                else:
                    pass

        if not is_sample_frame:
            for tag, cached_data in self._camera_cache.items():
                if cached_data is not None:
                    raw_dict[tag] = cached_data
        else:
            for tag in self._camera_tags:
                if tag in raw_dict:
                    self._camera_cache[tag] = raw_dict[tag]

        raw_dict = self._handle_speed_sensor(frame, raw_dict)

        raw_dict = self._ensure_all_sensors_present(raw_dict, frame)

        return raw_dict

    def _handle_speed_sensor(
        self,
        frame: int,
        data_dict: Dict[str, Tuple[int, Any]]
    ) -> Dict[str, Tuple[int, Any]]:
        speed_tag = self._speed_sensor_tag if self._speed_sensor_tag else 'SPEED'

        sensor_data = self._buffer.get_latest(speed_tag, frame)

        if sensor_data is not None:
            if sensor_data.frame == frame:
                self._speed_cache = (sensor_data.frame, sensor_data.data)
            elif sensor_data.age <= 10:
                self._speed_cache = (sensor_data.frame, sensor_data.data)
        elif self._speed_cache is None:
            self._speed_cache = (frame, {'speed': 0.0})

        data_dict[speed_tag] = self._speed_cache

        return data_dict

    def _get_sensor_group(self, tag: str, sensor_type: str) -> str:
        tag_lower = tag.lower()
        if 'cam' in tag_lower or 'bev' in tag_lower:
            return 'cam'
        elif tag_lower in ('gps', 'imu', 'speed'):
            return 'ego'
        elif sensor_type == 'sensor.opendrive_map':
            return 'opendrive'
        else:
            return 'other'

    def _ensure_all_sensors_present(
        self,
        data_dict: Dict[str, Tuple[int, Any]],
        frame: int
    ) -> Dict[str, Tuple[int, Any]]:
        speed_tag = self._speed_sensor_tag if self._speed_sensor_tag else 'SPEED'
        for tag in self._sensor_tags:
            if tag not in data_dict:
                if tag in self._camera_cache and self._camera_cache[tag] is not None:
                    data_dict[tag] = self._camera_cache[tag]
                elif tag == speed_tag and self._speed_cache is not None:
                    data_dict[tag] = self._speed_cache
                else:
                    data_dict[tag] = (frame, None)

        return data_dict
