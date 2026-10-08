import math


def project_xy(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x + s * y, -s * x + c * y


def read_state(snapshot, actor_id):
    actor = snapshot.find(actor_id)
    if actor is None:
        raise RuntimeError("Vehicle missing from frame {}".format(snapshot.frame))
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    vx, vy = project_xy(velocity.x, velocity.y, yaw)
    ax, ay = project_xy(acceleration.x, acceleration.y, yaw)
    state = dict(frame=snapshot.frame, time=snapshot.timestamp.elapsed_seconds,
                 x=transform.location.x, y=transform.location.y, z=transform.location.z,
                 yaw=yaw, vx=vx, vy=vy,
                 r=math.radians(actor.get_angular_velocity().z), ax=ax, ay=ay,
                 beta=math.atan2(vy, vx), beta_valid=vx >= 0.5,
                 world_vx=velocity.x, world_vy=velocity.y)
    if not all(math.isfinite(value) for value in state.values()):
        raise RuntimeError("Nonfinite state at frame {}".format(snapshot.frame))
    return state


def read_action(control):
    action = {name: float(getattr(control, name)) for name in ("steer", "throttle", "brake")}
    for name, lower in (("steer", -1), ("throttle", 0), ("brake", 0)):
        if not math.isfinite(action[name]) or not lower <= action[name] <= 1:
            raise RuntimeError("Invalid control {}={}".format(name, action[name]))
    action["longitudinal"] = action["throttle"] - action["brake"]
    action["reverse"] = bool(control.reverse)
    action["hand_brake"] = bool(control.hand_brake)
    action["gear"] = control.gear
    return action


class StateMonitor:
    def __init__(self, vehicle, snapshot, interval):
        self.vehicle = vehicle
        self.previous = read_state(snapshot, vehicle.id)
        self.interval = interval
        self.count = 0
        self.ranges = {}
        self.valid_beta_count = 0
        self.overlap_count = 0
        self.action_mismatch_count = 0
        self.max_action_delta = dict(steer=0.0, throttle=0.0, brake=0.0)
        self.last_requested = None
        print('STATE: SI, x=forward, y=right, vx>=0.5m/s', flush=True)

    def update(self, snapshot, requested_control):
        current = read_state(snapshot, self.vehicle.id)
        previous = self.previous
        dt = current["time"] - previous["time"]
        if current["frame"] != previous["frame"] + 1 or dt <= 0:
            raise RuntimeError('Nonconsecutive state/action transition')
        applied = read_action(self.vehicle.get_control())
        requested = read_action(requested_control)
        delta = {key: applied[key] - requested[key] for key in self.max_action_delta}
        mismatch = any(abs(value) > 1e-5 for value in delta.values())
        for key, value in delta.items():
            self.max_action_delta[key] = max(self.max_action_delta[key], abs(value))
        if mismatch:
            self.action_mismatch_count += 1
            if self.action_mismatch_count <= 5 or (self.count + 1) % self.interval == 0:
                print('CONTROL_DIAG;frame={};requested={} readback={} delta={} previous_requested={}'.format(
                    current["frame"], requested, applied, delta, self.last_requested), flush=True)
        self.last_requested = requested
        if any(applied[key] != requested[key] for key in ("reverse", "hand_brake")):
            raise RuntimeError("Control mode changed unexpectedly")
        dyaw = math.atan2(math.sin(current["yaw"] - previous["yaw"]),
                          math.cos(current["yaw"] - previous["yaw"]))
        ax_fd, ay_fd = project_xy((current["world_vx"] - previous["world_vx"]) / dt,
                                  (current["world_vy"] - previous["world_vy"]) / dt,
                                  current["yaw"])
        self.count += 1
        self.valid_beta_count += int(current["beta_valid"])
        self.overlap_count += int(applied["throttle"] > 1e-4 and applied["brake"] > 1e-4)
        for key in ("vx", "vy", "r", "ax", "ay", "beta"):
            if key == "beta" and not current["beta_valid"]:
                continue
            low, high = self.ranges.get(key, (current[key], current[key]))
            self.ranges[key] = (min(low, current[key]), max(high, current[key]))
        if self.count == 1 or self.count % self.interval == 0:
            print('STATE;frames={}->{};dt={:.3f}s;pos=({:.2f},{:.2f});yaw={:.3f}rad;vx={:.3f};vy={:.3f}m/s;r={:.4f}rad/s;ax={:.3f} ay={:.3f}m/s2 beta={:.4f}rad({:.2f}deg) valid={}'.format(
                      previous["frame"], current["frame"], dt, current["x"], current["y"], current["yaw"],
                      current["vx"], current["vy"], current["r"], current["ax"], current["ay"],
                      current["beta"], math.degrees(current["beta"]), current["beta_valid"]), flush=True)
            print('ACTION;{}->{}:;steer={:.3f};throttle={:.3f};brake={:.3f};longitudinal={:.3f};reverse={};hand_brake={};gear={};readback={};;FD;r={:.4f}rad/s ax={:.3f} ay={:.3f}m/s2'.format(
                      previous["frame"], current["frame"], applied["steer"], applied["throttle"],
                      applied["brake"], applied["longitudinal"], applied["reverse"], applied["hand_brake"],
                      applied["gear"], "DIFF" if mismatch else "OK", dyaw / dt, ax_fd, ay_fd), flush=True)
        self.previous = current

    def summary(self):
        print('STATE: transitions={} valid_beta={} pedal_overlap={} ranges(SI)={}'.format(
            self.count, self.valid_beta_count, self.overlap_count,
            {key: [round(v, 4) for v in values] for key, values in self.ranges.items()}), flush=True)
        print('CONTROL: mismatch={} tolerance=1e-5 delta={} status={}'.format(
            self.action_mismatch_count, self.max_action_delta,
            "NEEDS_REVIEW" if self.action_mismatch_count else "MATCH"), flush=True)
