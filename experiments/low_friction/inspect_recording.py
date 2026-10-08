import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["schema_version"] == 1, "Unknown schema")
    require(metadata["status"] == "closed", 'Incomplete episode: {}'.format(metadata))
    require(not (directory / "transitions.jsonl.partial").exists(), "Partial recording remains")
    with (directory / "transitions.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) > 0 and len(rows) == metadata["transitions"], "Missing/empty transitions")
    previous = None
    for index, row in enumerate(rows):
        require(finite_tree(row), "Nonfinite number at step {}".format(index))
        require(row["episode_id"] == metadata["episode_id"] and row["step"] == index, "Episode/step mismatch")
        s, n = row["state"], row["next_state"]
        require(n["frame"] == s["frame"] + 1, "Frame discontinuity")
        require(abs(n["time"] - s["time"] - row["dt"]) < 1e-6, "Timestamp mismatch")
        require(abs(row["dt"] - metadata["config"]["fixed_delta_seconds"]) < 1e-6, "Unexpected dt")
        if previous is not None:
            require(previous == s, 'Adjacent transition state mismatch')
        for state in (s, n):
            require(state["beta_valid"] == (state["vx"] >= metadata["beta_min_vx"]), "beta validity mismatch")
            require(abs(state["beta"] - math.atan2(state["vy"], state["vx"])) < 1e-6, "beta mismatch")
        a, requested = row["action"], row["requested_action"]
        for key, lower in (("steer", -1), ("throttle", 0), ("brake", 0)):
            require(lower <= a[key] <= 1 and lower <= requested[key] <= 1, "Action out of bounds")
            require(abs(a[key] - requested[key]) <= 1e-5, "Action alignment mismatch")
        require(all(a[k] == requested[k] for k in ("reverse", "hand_brake")), "Control mode mismatch")
        require(abs(a["longitudinal"] - (a["throttle"] - a["brake"])) < 1e-6, "Longitudinal mismatch")
        if index < len(rows) - 1:
            require(not row["terminated"] and not row["truncated"] and row["end_reason"] is None, "Early episode end")
        else:
            terminated = metadata["end_reason"] == "destination_complete"
            require(row["terminated"] == terminated and row["truncated"] == (not terminated), "End flags mismatch")
            require(row["end_reason"] == metadata["end_reason"], "End reason mismatch")
        previous = n
    return metadata, rows


def plot_episode(directory, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states = [rows[0]["state"]] + [r["next_state"] for r in rows]
    start = states[0]["time"]
    times = [s["time"] - start for s in states]
    action_times = [r["state"]["time"] - start for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes.flat[0]
    ax.plot([s["x"] for s in states], [s["y"] for s in states])
    ax.scatter([states[0]["x"], states[-1]["x"]], [states[0]["y"], states[-1]["y"]], c=["green", "red"])
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title='Trajectory: green=start, red=end')
    ax.axis("equal")
    for ax, keys, label in ((axes.flat[1], ("vx", "vy"), "Velocity (m/s)"),
                            (axes.flat[2], ("ax", "ay"), "Acceleration (m/s²)"),
                            (axes.flat[3], ("r",), "Yaw rate (rad/s)")):
        for key in keys:
            ax.plot(times, [s[key] for s in states], label=key)
        ax.set(xlabel="Simulation time (s)", ylabel=label)
        ax.legend()
    axes.flat[4].plot(times, [math.degrees(s["beta"]) if s["beta_valid"] else float("nan") for s in states])
    axes.flat[4].set(xlabel="Simulation time (s)", ylabel="Valid beta (deg)")
    for key in ("steer", "throttle", "brake"):
        axes.flat[5].step(action_times + [times[-1]],
                          [r["action"][key] for r in rows] + [rows[-1]["action"][key]], where="post", label=key)
    axes.flat[5].set(xlabel="Simulation time (s)", ylabel="Applied control")
    axes.flat[5].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.tight_layout()
    path = Path(directory) / "overview.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("PLOT {}".format(path.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    metadata, rows = validate(args.episode)
    states = [row["next_state"] for row in rows]
    ranges = {}
    for key in ("vx", "vy", "r", "ax", "ay", "beta"):
        values = [s[key] for s in states if key != "beta" or s["beta_valid"]]
        ranges[key] = [min(values), max(values)] if values else None
    print('PASS: episode transitions={} duration={:.2f}s reason={}'.format(
        len(rows), sum(row["dt"] for row in rows), metadata["end_reason"]))
    print("Valid beta={} throttle/brake overlap={} ranges(SI)={}".format(
        sum(s["beta_valid"] for s in states),
        sum(r["action"]["throttle"] > 1e-4 and r["action"]["brake"] > 1e-4 for r in rows), ranges))
    if args.plot:
        plot_episode(args.episode, rows)


if __name__ == "__main__":
    main()
