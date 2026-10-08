from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from state_monitor import read_action, read_state


class EpisodeRecorder:
    def __init__(self, root, metadata, vehicle, snapshot):
        self.episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(root) / self.episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=1, episode_id=self.episode_id,
                             status="recording", coordinate_frame="yaw-aligned x-forward y-right",
                             units="m, s, rad", beta_min_vx=0.5)
        self.write_metadata()
        self.stream = (self.directory / "transitions.jsonl.partial").open("x", encoding="utf-8")
        self.vehicle = vehicle
        self.previous = read_state(snapshot, vehicle.id)
        self.pending = None
        self.count = 0
        print("RECORD directory={}".format(self.directory.resolve()), flush=True)

    def write_metadata(self):
        temporary = self.directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.directory / "metadata.json")

    def flush_pending(self):
        if self.pending is not None:
            self.stream.write(json.dumps(self.pending, allow_nan=False) + "\n")
            self.stream.flush()
            self.pending = None

    def append(self, snapshot, requested_control, route_id):
        state = read_state(snapshot, self.vehicle.id)
        action = read_action(self.vehicle.get_control())
        requested = read_action(requested_control)
        if any(abs(action[k] - requested[k]) > 1e-5 for k in ("steer", "throttle", "brake")):
            raise RuntimeError("Recording rejected: action readback mismatch")
        if any(action[k] != requested[k] for k in ("reverse", "hand_brake")):
            raise RuntimeError("Recording rejected: control mode mismatch")
        dt = state["time"] - self.previous["time"]
        expected_dt = self.metadata["config"]["fixed_delta_seconds"]
        if state["frame"] != self.previous["frame"] + 1 or abs(dt - expected_dt) > 1e-6:
            raise RuntimeError("Recording rejected: frame/time discontinuity")
        self.flush_pending()
        self.pending = dict(episode_id=self.episode_id, step=self.count, route_id=route_id,
                            dt=dt, state=self.previous, action=action, requested_action=requested,
                            next_state=state, terminated=False, truncated=False, end_reason=None)
        self.previous = state
        self.count += 1

    def close(self, reason, error=None):
        if self.stream.closed:
            return
        if self.pending is not None:
            self.pending.update(terminated=reason == "destination_complete",
                                truncated=reason != "destination_complete", end_reason=reason)
        self.flush_pending()
        self.stream.close()
        (self.directory / "transitions.jsonl.partial").rename(self.directory / "transitions.jsonl")
        self.metadata.update(status="error" if error else "closed", end_reason=reason,
                             error=error, transitions=self.count)
        self.write_metadata()
        print("RECORD closed: transitions={} reason={} directory={}".format(
            self.count, reason, self.directory.resolve()), flush=True)
