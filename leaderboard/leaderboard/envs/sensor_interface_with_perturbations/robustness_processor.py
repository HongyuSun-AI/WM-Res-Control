import random
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None

from .config import SensorConfig


@dataclass
class BurstState:
    active: bool = False
    remaining_ticks: int = 0


class RobustnessProcessor:

    def __init__(self, config: SensorConfig):
        self.config = config
        self._burst_states: Dict[str, BurstState] = {}
        self._last_frame_cache: Dict[str, Optional[Tuple[int, Any]]] = {}

        if config.robustness_seed is not None:
            random.seed(config.robustness_seed)

        self._gps_drift_rng = None
        if config.gps_drift_seed is not None and np is not None:
            self._gps_drift_rng = np.random.RandomState(config.gps_drift_seed)

        self._speed_bias_rng = None
        if config.robustness_seed is not None and np is not None:
            self._speed_bias_rng = np.random.RandomState(config.robustness_seed + 42)

    def should_process(self, tag: str, group: str) -> bool:
        if not self.config.robustness_enable:
            return False

        if group == 'cam':
            if tag not in self._burst_states:
                self._burst_states[tag] = BurstState()

            state = self._burst_states[tag]

            if state.active and self.config.frame_drop_enable:
                return True

            if self.config.burst_probability > 0:
                if self.config.frame_drop_enable:
                    return True

            if not self.config.partial_obs_enable:
                return False

            return True

        if group == 'ego':
            if self.config.gps_drift_enable and tag in ('GPS', 'gps'):
                return True
            if self.config.speed_bias_enable and tag in ('SPEED', 'speed'):
                return True
            return False

        return False

    def process_sensor_data(
        self,
        tag: str,
        group: str,
        frame: int,
        data: Any
    ) -> Tuple[int, Any]:
        if not self.config.robustness_enable:
            return (frame, data)

        if tag not in self._burst_states:
            self._burst_states[tag] = BurstState()
        if tag not in self._last_frame_cache:
            self._last_frame_cache[tag] = None

        cached_result = self._last_frame_cache[tag]

        result = (frame, data)

        if group == 'cam':
            if self._check_burst_trigger(tag):
                data = self._apply_frame_drop(tag, data, cached_result)
                result = (frame, data)

            if self.config.partial_obs_enable:
                data = self._apply_partial_observation(tag, data)
                result = (frame, data)

        elif group == 'ego' and tag in ('GPS', 'gps'):
            data = self._apply_gps_drift(tag, data)
            result = (frame, data)

        elif group == 'ego' and tag in ('SPEED', 'speed'):
            data = self._apply_speed_bias(tag, data)
            result = (frame, data)

        self._last_frame_cache[tag] = result

        return result

    def _check_burst_trigger(self, tag: str) -> bool:
        state = self._burst_states[tag]

        if state.active:
            state.remaining_ticks -= 1
            if state.remaining_ticks <= 0:
                state.active = False
                print(f"[BURST END] tag={tag}", flush=True)
            return True

        if random.random() < self.config.burst_probability:
            burst_duration = random.randint(2, self.config.burst_max_ticks)
            state.active = True
            state.remaining_ticks = burst_duration
            print(f"[BURST START] tag={tag} duration={burst_duration}", flush=True)
            return True

        return False

    def _apply_frame_drop(
        self,
        tag: str,
        data: Any,
        cached: Optional[Tuple[int, Any]]
    ) -> Any:
        if not self.config.frame_drop_enable:
            return data

        if cached is not None:
            print(f"[FRAME DROP] tag={tag}", flush=True)
            return cached[1]

        return data

    def _apply_partial_observation(self, tag: str, data: Any) -> Any:
        if not self.config.partial_obs_enable:
            return data

        if not self._is_rgb_image(data):
            return data

        if self.config.partial_obs_type == "blur":
            return self._apply_partial_blur(data)
        elif self.config.partial_obs_type == "occlusion":
            return self._apply_partial_occlusion(data)

        return data

    def _apply_partial_blur(self, image: Any) -> Any:
        if np is None or cv2 is None:
            return image

        h, w = image.shape[:2]

        mask_h = int(h * self.config.partial_obs_ratio)
        mask_w = int(w * self.config.partial_obs_ratio)
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)

        region = image[top:top+mask_h, left:left+mask_w]
        blurred = cv2.GaussianBlur(region, (31, 31), 0)

        result = image.copy()
        result[top:top+mask_h, left:left+mask_w] = blurred

        print(f"[PARTIAL BLUR] region=({top},{left},{mask_h},{mask_w})", flush=True)
        return result

    def _apply_partial_occlusion(self, image: Any) -> Any:
        if np is None:
            return image

        h, w = image.shape[:2]

        mask_h = int(h * self.config.partial_obs_ratio)
        mask_w = int(w * self.config.partial_obs_ratio)
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)

        result = image.copy()
        result[top:top+mask_h, left:left+mask_w] = 0

        print(f"[PARTIAL OCCLUSION] region=({top},{left},{mask_h},{mask_w})", flush=True)
        return result

    def _is_rgb_image(self, data: Any) -> bool:
        if np is None:
            return False
        return isinstance(data, np.ndarray) and data.ndim == 3

    def _apply_gps_drift(self, tag: str, data: Any) -> Any:
        if np is None:
            return data

        if not self.config.gps_drift_enable:
            return data

        if not isinstance(data, np.ndarray) or data.shape != (3,):
            return data

        drift_std_map = {'medium': 5.0, 'severe': 15.0}
        drift_std = drift_std_map.get(self.config.gps_drift_mode, 5.0)

        lat, lon = data[0], data[1]

        if self._gps_drift_rng is not None:
            lat_drift_m = self._gps_drift_rng.normal(0, drift_std)
            lon_drift_m = self._gps_drift_rng.normal(0, drift_std)
        else:
            lat_drift_m = np.random.normal(0, drift_std)
            lon_drift_m = np.random.normal(0, drift_std)

        lat_drift_deg = lat_drift_m / 111000.0

        lon_scale = 111000.0 * np.cos(np.deg2rad(lat))
        lon_drift_deg = lon_drift_m / lon_scale if lon_scale > 0 else 0

        result = data.copy()
        result[0] = lat + lat_drift_deg
        result[1] = lon + lon_drift_deg

        print(f"[GPS DRIFT] tag={tag} mode={self.config.gps_drift_mode} "
              f"drift_lat={lat_drift_m:.2f}m drift_lon={lon_drift_m:.2f}m", flush=True)

        return result

    def _apply_speed_bias(self, tag: str, data: Any) -> Any:
        if not self.config.speed_bias_enable:
            return data

        if not isinstance(data, dict) or 'speed' not in data:
            return data

        if np is None:
            return data

        mean = self.config.speed_bias_mean
        std = self.config.speed_bias_std

        if self._speed_bias_rng is not None:
            multiplier = self._speed_bias_rng.normal(mean, std)
        else:
            multiplier = np.random.normal(mean, std)

        original_speed = data['speed']
        biased_speed = original_speed * multiplier

        result = data.copy()
        result['speed'] = biased_speed

        bias_pct = (multiplier - 1.0) * 100
        print(f"[SPEED BIAS] tag={tag} mean={mean} std={std} "
              f"original={original_speed:.2f}m/s biased={biased_speed:.2f}m/s (N({mean},{std})={multiplier:.3f}, {bias_pct:+.1f}%)", flush=True)

        return result
