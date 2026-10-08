import argparse
import json
from pathlib import Path
import numpy as np


def reference(metadata, state=None, sensor_frames=None):
    if metadata.get('agent') != 'only_traj':
        raise ValueError('Expected frozen TCP+PID only_traj metadata')
    points = np.asarray([metadata['wp_'+str(i)] for i in range(1,5)], dtype=float)
    if points.shape != (4,2) or not np.isfinite(points).all():
        raise ValueError('Invalid waypoints')
    control = {k: float(metadata[k]) for k in ('steer','throttle','brake')}
    for k, value in control.items():
        if not np.isfinite(value) or not (-1 if k == 'steer' else 0) <= value <= 1:
            raise ValueError('Invalid final control')
    if control['throttle'] > 1e-4 and control['brake'] > 1e-4:
        raise ValueError('Simultaneous pedals unsupported')
    frames = {} if sensor_frames is None else dict(sensor_frames)
    required = ('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','GPS','IMU','SPEED')
    aligned = state is not None and all(k in frames and frames[k] == state['frame'] for k in required)
    if state is not None:
        for k in ('frame','time','x','y','yaw','vx','vy','r','ax','ay'):
            if not np.isfinite(state[k]):
                raise ValueError('Invalid snapshot state '+k)
    return dict(schema_version=1, waypoints_forward_right=points[:, ::-1].tolist(),
                waypoint_times_seconds=[.5,1.,1.5,2.],
                timing_evidence='TCP/test.py l2_05/l2_1/l2_15/l2_2',
                spatial_origin_verified=False,
                origin_note='GNSS x=-1.4m; snapshot=actor-origin',
                control_final=control, action=[control['steer'],control['throttle']-control['brake']],
                action_semantics='returned command for next transition',
                pid_raw={k:float(metadata[k+'_traj']) for k in ('steer','throttle','brake')},
                state=state, sensor_frames=frames, frame_alignment_verified=bool(aligned),
                ready_for_trajectory_loss=False)

