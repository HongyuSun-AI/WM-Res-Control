import math


def fixed_speed_pedals(pid,speed,target_kmh,config):
    if not math.isfinite(speed) or not math.isfinite(target_kmh) or not 0<=target_kmh<=40:
        raise ValueError('Speed range: [0,40]km/h')
    if target_kmh==0:return 0.,1.
    target=target_kmh/3.6
    brake=target<config.brake_speed or max(0.,speed)/target>config.brake_ratio
    delta=min(config.clip_delta,max(0.,target-speed))
    throttle=min(config.max_throttle,max(0.,float(pid.step(delta))))
    return (0.,1.) if brake else (throttle,0.)
