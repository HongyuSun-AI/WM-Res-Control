import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SensorConfig:

    sim_rate: int = 20

    sensor_delay_ego: int = 0
    sensor_delay_cam: int = 0

    blend_enable: bool = False
    blend_mode: Literal['uniform', 'linear'] = 'uniform'
    blend_num: int = 1

    robustness_enable: bool = False
    burst_probability: float = 0.00
    burst_max_ticks: int = 5

    frame_drop_enable: bool = False

    partial_obs_enable: bool = False
    partial_obs_type: Literal['blur', 'occlusion'] = 'blur'
    partial_obs_ratio: float = 0.3

    gps_drift_enable: bool = False
    gps_drift_mode: Literal['medium', 'severe'] = 'medium'
    gps_drift_seed: Optional[int] = None

    speed_bias_enable: bool = False
    speed_bias_mean: float = 1.0
    speed_bias_std: float = 0.1

    robustness_seed: Optional[int] = None

    queue_timeout: int = 300

    @property
    def rgb_frame_interval(self) -> int:
        return self.sim_rate // 20

    @classmethod
    def from_env(cls) -> 'SensorConfig':
        config = cls()

        config.sim_rate = int(os.environ.get("SIM_RATE", "20"))
        if config.sim_rate <= 0:
            raise ValueError(f"Invalid SIM_RATE: {config.sim_rate}")

        config.sensor_delay_ego = int(os.environ.get("SENSOR_DELAY_EGO", "0"))
        config.sensor_delay_cam = int(os.environ.get("SENSOR_DELAY_CAM", "0"))
        if config.sensor_delay_ego < 0 or config.sensor_delay_cam < 0:
            raise ValueError('Sensor delay requires >=0')

        config.blend_enable = int(os.environ.get("BLEND_ENABLE", "0")) != 0
        config.blend_mode = os.environ.get("BLEND_MODE", "uniform").lower()
        config.blend_num = int(os.environ.get("BLEND_NUM", "1"))
        if config.blend_mode not in ("uniform", "linear"):
            raise ValueError(f"Unsupported BLEND_MODE={config.blend_mode}")
        if config.blend_num < 1:
            raise ValueError(f"Invalid BLEND_NUM: {config.blend_num}")

        config.robustness_enable = int(os.environ.get("ROBUSTNESS_ENABLE", "0")) != 0
        config.burst_probability = float(os.environ.get("BURST_PROBABILITY", "0.01"))
        config.burst_max_ticks = int(os.environ.get("BURST_MAX_TICKS", "5"))

        config.frame_drop_enable = int(os.environ.get("FRAME_DROP_ENABLE", "0")) != 0

        config.partial_obs_enable = int(os.environ.get("PARTIAL_OBS_ENABLE", "0")) != 0
        config.partial_obs_type = os.environ.get("PARTIAL_OBS_TYPE", "blur")
        config.partial_obs_ratio = float(os.environ.get("PARTIAL_OBS_RATIO", "0.3"))

        config.gps_drift_enable = int(os.environ.get("GPS_DRIFT_ENABLE", "0")) != 0
        config.gps_drift_mode = os.environ.get("GPS_DRIFT_MODE", "medium").lower()
        if config.gps_drift_mode not in ("medium", "severe"):
            raise ValueError(f"Unsupported GPS_DRIFT_MODE={config.gps_drift_mode}")

        drift_seed = os.environ.get("GPS_DRIFT_SEED", None)
        config.gps_drift_seed = int(drift_seed) if drift_seed is not None else None

        config.speed_bias_enable = int(os.environ.get("SPEED_BIAS_ENABLE", "0")) != 0
        config.speed_bias_mean = float(os.environ.get("SPEED_BIAS_MEAN", "1.0"))
        config.speed_bias_std = float(os.environ.get("SPEED_BIAS_STD", "0.1"))
        if config.speed_bias_std < 0:
            raise ValueError(f"Invalid SPEED_BIAS_STD: {config.speed_bias_std}")

        seed = os.environ.get("ROBUSTNESS_SEED", None)
        config.robustness_seed = int(seed) if seed is not None else None

        return config

    def get_delay_for_group(self, group: str) -> int:
        delay_map = {
            'ego': self.sensor_delay_ego,
            'cam': self.sensor_delay_cam,
            'other': 0,
            'opendrive': 0,
        }
        return delay_map.get(group, 0)

    def get_buffer_size(self, group: str) -> int:
        delay = self.get_delay_for_group(group)
        history_len = delay + 1
        if self.blend_enable and self.blend_num > 1:
            history_len = max(history_len, delay + self.blend_num)
        return history_len
