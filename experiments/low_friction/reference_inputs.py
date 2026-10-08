import math
import threading
import time


class ExactFrameInputs:
    def __init__(self, timeout=10.):
        self.timeout=timeout
        self.condition=threading.Condition()
        self.buffers={}

    def register_sensor(self, tag, sensor_type, sensor):
        with self.condition:
            if tag in self.buffers: raise ValueError('Duplicate sensor '+tag)
            self.buffers[tag]={}

    def update_sensor(self, tag, data, frame):
        if tag == 'SPEED': return
        self._put(tag,data,frame)

    def _put(self, tag, data, frame):
        with self.condition:
            if tag not in self.buffers: raise ValueError('Unknown sensor '+tag)
            buf=self.buffers[tag]; buf[frame]=data
            for old in sorted(buf)[:-16]: del buf[old]
            self.condition.notify_all()

    def set_snapshot_speed(self, actor_snapshot, frame):
        t=actor_snapshot.get_transform(); v=actor_snapshot.get_velocity()
        yaw=math.radians(t.rotation.yaw); pitch=math.radians(t.rotation.pitch)
        speed=v.x*math.cos(pitch)*math.cos(yaw)+v.y*math.cos(pitch)*math.sin(yaw)+v.z*math.sin(pitch)
        self._put('SPEED',{'speed':speed},frame)

    def get_data(self, frame):
        deadline=time.monotonic()+self.timeout
        with self.condition:
            if not self.buffers: raise RuntimeError('No registered sensors')
            while True:
                missing=[tag for tag,buf in self.buffers.items() if frame not in buf]
                if not missing:
                    result={tag:(frame,buf[frame]) for tag,buf in self.buffers.items()}
                    for buf in self.buffers.values():
                        for old in list(buf):
                            if old<=frame: del buf[old]
                    return result
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    latest={tag:max(self.buffers[tag],default=None) for tag in missing}
                    raise RuntimeError('Exact-frame timeout frame={} missing/latest={}'.format(frame,latest))
                self.condition.wait(remaining)


def gps_to_xy(gps, lat_ref, lon_ref):
    lat,lon=map(float,gps[:2]); radius=6378137.
    scale=math.cos(math.radians(lat_ref))
    x=scale*radius*math.radians(lon-lon_ref)
    y=scale*radius*(math.log(math.tan(math.radians(90+lat_ref)/2))-
                    math.log(math.tan(math.radians(90+lat)/2)))
    if not all(math.isfinite(v) for v in (x,y)): raise ValueError('Nonfinite GNSS projection')
    return [x,y]
