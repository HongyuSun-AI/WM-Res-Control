import numpy as np
from typing import List, Any, Tuple

from .config import SensorConfig


class TemporalBlender:

    def __init__(self, config: SensorConfig):
        self.config = config

    def blend_camera_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        if len(frames) == 1:
            return frames[0]

        arr = np.stack([x.astype(np.float32) for x in frames], axis=0)

        if self.config.blend_mode == "uniform":
            out = arr.mean(axis=0)
        elif self.config.blend_mode == "linear":
            weights = np.arange(1, len(frames) + 1, dtype=np.float32)
            weights = weights / weights.sum()
            out = (arr * weights[:, None, None, None]).sum(axis=0)
        else:
            raise ValueError(f"Unsupported blend mode: {self.config.blend_mode}")

        ref = frames[-1]
        if np.issubdtype(ref.dtype, np.integer):
            out = np.rint(np.clip(out, 0, 255)).astype(ref.dtype)
        else:
            out = out.astype(ref.dtype)
        return out

    def blend_values(self, values: List[Any]) -> Any:
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]

        if self.config.blend_mode == "uniform":
            weights = np.ones(len(values), dtype=np.float32)
        elif self.config.blend_mode == "linear":
            weights = np.arange(1, len(values) + 1, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported blend mode: {self.config.blend_mode}")

        weights = weights / weights.sum()
        return self._weighted_sum(values, weights)

    def _weighted_sum(self, values: List[Any], weights: np.ndarray) -> Any:
        first = values[0]

        if isinstance(first, np.ndarray):
            out = np.zeros_like(first, dtype=np.float32)
            for v, w in zip(values, weights):
                out += v.astype(np.float32) * w

            if np.issubdtype(first.dtype, np.integer):
                out = np.clip(out, 0, 255).astype(first.dtype)
            else:
                out = out.astype(first.dtype)
            return out

        if isinstance(first, (int, float, np.number)):
            return sum(float(v) * float(w) for v, w in zip(values, weights))

        if isinstance(first, tuple):
            return tuple(
                self._weighted_sum(list(items), weights)
                for items in zip(*values)
            )

        if isinstance(first, list):
            return [
                self._weighted_sum(list(items), weights)
                for items in zip(*values)
            ]

        if isinstance(first, dict):
            out = {}
            for k in first.keys():
                out[k] = self._weighted_sum([v[k] for v in values], weights)
            return out

        return values[-1]

    def is_rgb_image(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim == 3

    def apply_blending(
        self,
        tag: str,
        group: str,
        frame_data_list: List[Tuple[int, Any]]
    ) -> Tuple[int, Any]:
        if len(frame_data_list) == 1:
            return frame_data_list[0]

        frames = [item[0] for item in frame_data_list]
        datas = [item[1] for item in frame_data_list]

        if group == 'cam' and all(self.is_rgb_image(d) for d in datas):
            blended = self.blend_camera_frames(datas)
        else:
            blended = self.blend_values(datas)

        return frames[-1], blended
