from collections import deque
from typing import Dict, Tuple, Any, Deque, Optional, List
from dataclasses import dataclass

from .config import SensorConfig


@dataclass
class SensorData:
    frame: int
    data: Any
    age: int = 0


class DelayBuffer:

    def __init__(self, config: SensorConfig):
        self.config = config
        self._buffers: Dict[str, Deque[Tuple[int, Any]]] = {}
        self._latest_frame: Dict[str, int] = {}

    def register_sensor(self, tag: str, group: str) -> None:
        buffer_size = self.config.get_buffer_size(group)
        self._buffers[tag] = deque(maxlen=buffer_size)
        self._latest_frame[tag] = -1

    def append(self, tag: str, frame: int, data: Any) -> None:
        if tag not in self._buffers:
            raise ValueError(f"Unregistered sensor {tag}")

        self._buffers[tag].append((frame, data))
        self._latest_frame[tag] = frame

    def get_latest(self, tag: str, current_frame: int) -> Optional[SensorData]:
        if tag not in self._buffers:
            return None

        buf = self._buffers[tag]
        if len(buf) == 0:
            return None

        latest_frame, latest_data = buf[-1]
        age = current_frame - latest_frame if current_frame >= latest_frame else 0

        return SensorData(frame=latest_frame, data=latest_data, age=age)

    def get_delayed_data(
        self,
        tag: str,
        group: str,
        current_frame: int,
        current_data: Any
    ) -> Tuple[int, Any]:
        if tag not in self._buffers:
            raise ValueError(f"Unregistered sensor {tag}")

        self._buffers[tag].append((current_frame, current_data))
        self._latest_frame[tag] = current_frame

        buf = self._buffers[tag]
        delay = self.config.get_delay_for_group(group)

        if len(buf) <= delay:
            result = (current_frame, current_data)
            return result

        target_idx = len(buf) - delay - 1
        result = buf[target_idx]

        return result

    def get_cache(self, tag: str) -> Optional[Tuple[int, Any]]:
        if tag not in self._buffers:
            return None
        buf = self._buffers[tag]
        if len(buf) == 0:
            return None
        return buf[-1]

    def update_cache(self, tag: str, frame: int, data: Any) -> None:
        pass

    def get_latest_frame(self, tag: str) -> int:
        return self._latest_frame.get(tag, -1)

    def get_buffer_size(self, tag: str) -> int:
        if tag not in self._buffers:
            return 0
        return len(self._buffers[tag])

    def buffer_info(self, tag: str) -> Dict[str, Any]:
        if tag not in self._buffers:
            return {"error": "not_registered"}

        buf = self._buffers[tag]
        return {
            "size": len(buf),
            "maxlen": buf.maxlen,
            "latest_frame": self._latest_frame.get(tag, -1)
        }

    def get_buffer_copy(self, tag: str) -> Optional[List[Tuple[int, Any]]]:
        if tag not in self._buffers:
            return None
        return list(self._buffers[tag])
