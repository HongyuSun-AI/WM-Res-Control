import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from inspect_recording import require, validate


STATE_FIELDS = ["vx", "vy", "r", "ax", "ay"]
ACTION_FIELDS = ["steer", "longitudinal"]
TARGET_FIELDS = STATE_FIELDS + ["delta_px", "delta_py", "delta_yaw"]


def construct(rows, history_length):
    require(history_length >= 1 and len(rows) > history_length, "Episode too short for history")
    for row in rows:
        action = row["action"]
        require(not action["reverse"] and not action["hand_brake"], "Unsupported reverse/handbrake control")
        require(not (action["throttle"] > 1e-4 and action["brake"] > 1e-4),
                'Simultaneous throttle/brake unsupported')
    histories, actions, targets, frames = [], [], [], []
    for t in range(history_length, len(rows)):
        history = [[rows[i]["state"][k] for k in STATE_FIELDS] +
                   [rows[i-1]["action"][k] for k in ACTION_FIELDS]
                   for i in range(t-history_length+1, t+1)]
        state, next_state = rows[t]["state"], rows[t]["next_state"]
        dx, dy = next_state["x"] - state["x"], next_state["y"] - state["y"]
        c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
        dyaw = next_state["yaw"] - state["yaw"]
        target = [next_state[k] for k in STATE_FIELDS] + [
            c*dx + s*dy, -s*dx + c*dy, math.atan2(math.sin(dyaw), math.cos(dyaw))]
        histories.append(history)
        actions.append([rows[t]["action"][k] for k in ACTION_FIELDS])
        targets.append(target)
        frames.append([state["frame"], next_state["frame"]])
    result = dict(history=np.asarray(histories, dtype=np.float32),
                  action=np.asarray(actions, dtype=np.float32), target=np.asarray(targets, dtype=np.float32),
                  frames=np.asarray(frames, dtype=np.int64),
                  step=np.arange(history_length, len(rows), dtype=np.int64),
                  history_frames=np.asarray([[rows[i]["state"]["frame"]
                      for i in range(t-history_length+1, t+1)] for t in range(history_length, len(rows))], dtype=np.int64),
                  terminated=np.asarray([r["terminated"] for r in rows[history_length:]], dtype=np.bool_),
                  truncated=np.asarray([r["truncated"] for r in rows[history_length:]], dtype=np.bool_))
    require(all(np.isfinite(a).all() for a in result.values()), "Nonfinite tensor")
    return result


def statistics(values):
    mean, std = values.mean(axis=0), values.std(axis=0)
    scale = np.where(std < 1e-6, 1.0, std)
    return dict(mean=mean.tolist(), std=std.tolist(), scale=scale.tolist())


def build(train, validation, output, history_length):
    require(history_length >= 1, "History length must be positive")
    require(not output.exists(), 'Output already exists')
    groups = {"train": [], "validation": []}
    sources, ids, environment = [], set(), None
    for split, paths in (("train", train), ("validation", validation)):
        for path in paths:
            metadata, rows = validate(path)
            episode_id = metadata["episode_id"]
            require(episode_id not in ids, "Duplicate episode or train/validation leakage")
            ids.add(episode_id)
            signature = {k: metadata["config"][k] for k in
                         ("vehicle", "weather", "fixed_delta_seconds", "tire_friction",
                          "max_substep_delta_time", "max_substeps")}
            if environment is None:
                environment = signature
            require(signature == environment, "Mixed dynamics/environment configuration")
            arrays = construct(rows, history_length)
            arrays["episode_id"] = np.asarray([episode_id]*len(arrays["action"]))
            groups[split].append(arrays)
            sources.append(dict(split=split, episode_id=episode_id, path=str(path.resolve()),
                                transitions=len(rows), samples=len(arrays["action"]),
                                sha256=hashlib.sha256((path / "transitions.jsonl").read_bytes()).hexdigest()))
    require(bool(groups["train"]), 'Training episode required')
    merged = {split: {k: np.concatenate([a[k] for a in items]) for k in items[0]}
              for split, items in groups.items() if items}
    train_states, train_actions = [], []
    for source in sources:
        if source["split"] != "train":
            continue
        _, rows = validate(Path(source["path"]))
        train_states.extend([[r["state"][k] for k in STATE_FIELDS] for r in rows[1:]])
        train_actions.extend([[r["action"][k] for k in ACTION_FIELDS] for r in rows])
    stats = dict(state=statistics(np.asarray(train_states, dtype=np.float64)),
                 action=statistics(np.asarray(train_actions, dtype=np.float64)),
                 target=statistics(merged["train"]["target"].astype(np.float64)))
    manifest = dict(schema_version=1, history_length=history_length,
                    history_fields=STATE_FIELDS+['previous_'+k for k in ACTION_FIELDS],
                    action_fields=ACTION_FIELDS, target_fields=TARGET_FIELDS, environment=environment,
                    validation_available=bool(validation), sources=sources, normalization=stats,
                    note='One-step dynamics samples')
    output.mkdir(parents=True, exist_ok=False)
    for split, arrays in merged.items():
        hm = np.asarray(stats['state']['mean']+stats['action']['mean'])
        hs = np.asarray(stats['state']['scale']+stats['action']['scale'])
        arrays['history_normalized'] = ((arrays['history']-hm)/hs).astype(np.float32)
        for name in ('action','target'):
            arrays[name+'_normalized'] = ((arrays[name]-np.asarray(stats[name]['mean'])) /
                                          np.asarray(stats[name]['scale'])).astype(np.float32)
        np.savez_compressed(output / (split+'.npz'), **arrays)
        print('{}: history={} action={} target={}'.format(split, arrays['history'].shape,
              arrays['action'].shape, arrays['target'].shape))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    if not validation:
        print('Validation dataset unavailable')
    print('SAMPLES {}'.format(output.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--validation', nargs='+', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history-length', type=int, default=10)
    args = parser.parse_args()
    build(args.train, args.validation, args.output, args.history_length)


if __name__ == '__main__':
    main()
