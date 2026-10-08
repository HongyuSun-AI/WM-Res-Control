import json
import os
import math
from pathlib import Path
from uuid import uuid4

from team_code.tcp_b2d_agent import TCPAgent, PLANNER_TYPE
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from experiments.low_friction.state_monitor import read_state
from experiments.low_friction.tcp_reference import reference
from experiments.low_friction.reference_inputs import gps_to_xy


def get_entry_point():
    return 'TCPReferenceAgent'


class TCPReferenceAgent(TCPAgent):
    def bind_reference_vehicle(self, vehicle):
        if vehicle is None or not vehicle.is_alive or vehicle.attributes.get('role_name') != 'hero':
            raise RuntimeError('Invalid vehicle: id={} alive={} role={}'.format(
                getattr(vehicle, 'id', None), getattr(vehicle, 'is_alive', False),
                getattr(vehicle, 'attributes', {}).get('role_name')))
        self.hero_actor = vehicle

    def setup(self, path_to_conf_file):
        if PLANNER_TYPE != 'only_traj':
            raise ValueError('Reference capture requires PLANNER_TYPE=only_traj')
        directory = Path(os.environ['TCP_REFERENCE_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        super().setup(path_to_conf_file)
        self.net.requires_grad_(False)
        self._reference_file = (directory/('reference_'+uuid4().hex+'.jsonl')).open('x', buffering=1)

    def run_step(self, input_data, timestamp):
        self.latest_reference_row = None
        hero = getattr(self, 'hero_actor', None)
        if hero is None or not hero.is_alive:
            raise RuntimeError('Reference hero unavailable')
        control = super().run_step(input_data, timestamp)
        if self.step >= self.config.seq_len:
            snapshot = CarlaDataProvider.get_world().get_snapshot()
            state = read_state(snapshot, hero.id)
            original_control={k:float(getattr(control,k)) for k in ('steer','throttle','brake')}
            target=getattr(self,'experiment_target_speed_kmh',None)
            uncapped=getattr(self,'experiment_uncapped_tcp',False)
            route_speed=None
            if hasattr(self,'experiment_route_speed'):
                route_speed=self.experiment_route_speed.observe(state)
            if uncapped:
                control.throttle=0. if control.brake>0 else max(0.,min(.75,float(self.pid_metadata['throttle_traj'])))
                if route_speed is not None:
                    route_speed['phase']='arrival' if route_speed['remaining_route_m']<50 else 'cruise'
                    if state['vx']>route_speed['commanded_speed_kmh']/3.6 or route_speed['stopped'] or route_speed['commanded_speed_kmh']==0:
                        control.throttle=0.;control.brake=1.
                self.pid_metadata=dict(self.pid_metadata)
                for k in ('throttle','brake'):self.pid_metadata[k]=float(getattr(control,k))
            if target is not None:
                from experiments.low_friction.fixed_speed import fixed_speed_pedals
                if not hasattr(self,'experiment_speed_pid'):
                    from TCP.model import PIDController
                    self.experiment_speed_pid=PIDController(K_P=self.config.speed_KP,K_I=self.config.speed_KI,K_D=self.config.speed_KD,n=self.config.speed_n)
                effective_target=route_speed['commanded_speed_kmh'] if route_speed is not None else target
                control.throttle,control.brake=fixed_speed_pedals(self.experiment_speed_pid,state['vx'],effective_target,self.config)
                self.pid_metadata=dict(self.pid_metadata)
                for k in ('throttle','brake'):self.pid_metadata[k]=float(getattr(control,k))
            frames = {k:int(v[0]) for k,v in input_data.items()}
            row = reference(self.pid_metadata, state, frames)
            if uncapped:
                row['base_control_mode']='TCP_without_extra_low_speed_throttle_cap'
                row['original_TCP_control']=original_control
                row['TCP_desired_speed_kmh']=float(self.pid_metadata['desired_speed'])*3.6
                if route_speed is not None:row['route_speed']=route_speed
            if target is not None:
                row['base_control_mode']='frozen_TCP_steering+fixed_speed_PID'
                row['target_speed_kmh']=target
                row['original_TCP_control']=original_control
                if route_speed is not None:row['route_speed']=route_speed
            row['agent_step'] = self.step
            row['agent_timestamp'] = float(timestamp)
            gps = input_data['GPS'][1][:2]
            row['gnss_world_xy'] = self.gps_to_location(gps).tolist()
            row['gnss_agent_world_xy'] = row['gnss_world_xy']
            row['gnss_projection_source'] = 'agent_fsolve'
            if hasattr(self, 'reference_georef'):
                row['gnss_world_xy'] = gps_to_xy(gps,*self.reference_georef)
                row['gnss_projection_source'] = 'map_georeference_verified'
                row['map_georeference'] = list(self.reference_georef)
            row['agent_georeference'] = [float(self.lat_ref),float(self.lon_ref)]
            row['gnss_lat_lon'] = [float(v) for v in gps]
            row['speed_source'] = ('actor_snapshot_forward_projection' if hasattr(self,'reference_georef')
                                   else 'sensor_interface')
            row['compass_raw'] = float(input_data['IMU'][1][-1])
            row['tcp_yaw'] = (0.0 if math.isnan(row['compass_raw']) else row['compass_raw'])-math.pi/2
            row['heading_valid'] = math.isfinite(row['compass_raw'])
            if not row['heading_valid']:
                row['compass_raw'] = None
            row['wheel_friction'] = [float(w.tire_friction) for w in hero.get_physics_control().wheels]
            row['map'] = CarlaDataProvider.get_world().get_map().name
            for key in ('steer','throttle','brake'):
                if abs(row['control_final'][key]-float(getattr(control,key))) > 1e-7:
                    raise RuntimeError('Final control metadata mismatch')
            self._reference_file.write(json.dumps(row,allow_nan=False)+'\n')
            self.latest_reference_row = row
        return control

    def destroy(self):
        stream = getattr(self,'_reference_file',None)
        if stream is not None:
            stream.close()
            self._reference_file = None
        if hasattr(self, 'net'):
            super().destroy()
