import json
import queue
import cv2
import numpy as np


class DemoCamera:
    def __init__(self,world,vehicle,carla,output):
        self.output=output;self.frames=[];self.queue=queue.Queue();self.actor=None
        self.spectator=world.get_spectator()
        self.original_spectator_transform=self.spectator.get_transform()
        self.writer=cv2.VideoWriter(str(output/'camera.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20.,(960,540))
        if not self.writer.isOpened():raise RuntimeError('Video encoder unavailable')
        bp=world.get_blueprint_library().find('sensor.camera.rgb')
        for k,v in dict(image_size_x='960',image_size_y='540',fov='80',sensor_tick='0.0').items():bp.set_attribute(k,v)
        try:
            self.actor=world.spawn_actor(bp,carla.Transform(carla.Location(x=-7,z=7),carla.Rotation(pitch=-35)),attach_to=vehicle)
            self.actor.listen(self.queue.put)
        except BaseException:self.writer.release();raise

    def capture(self,frame):
        while True:
            image=self.queue.get(timeout=15)
            if image.frame<frame:continue
            if image.frame!=frame:raise RuntimeError('Demo camera frame mismatch')
            self.spectator.set_transform(image.transform)
            pixels=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(540,960,4)[:,:,:3]
            self.writer.write(np.ascontiguousarray(pixels));self.frames.append(frame);return

    def close(self):
        try:
            if self.actor is not None and self.actor.is_alive:
                self.actor.stop();self.actor.destroy()
        finally:
            self.writer.release()
            (self.output/'camera_frames.json').write_text(json.dumps(dict(fps=20,width=960,height=540,frames=self.frames)))
            self.spectator.set_transform(self.original_spectator_transform)
