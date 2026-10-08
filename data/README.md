# Data Directory

This directory contains collection and preparation scripts alongside downloaded data.

Dataset: [Google Drive](https://drive.google.com/file/d/1jsaaDUzWepih2cwlcuvvtHdk_RpckVN8/view?usp=sharing). Place `WM_Res_Control_Data.tar.gz` in this directory and run the following from the project root:

```bash
tar -xzf data/WM_Res_Control_Data.tar.gz -C data
```

Allow approximately 60 GB for the archive and extracted files together.

## Dataset Contents

| Directory | Contents | Purpose |
|---|---|---|
| `raw/archive_data/` | Historical dynamics recordings, batch information, and sample arrays | Inspect raw data and sample provenance |
| `raw/archive_runs/` | TCP references, executed-control readbacks, road/lane geometry, and experiment metadata | Inspect the residual-control collection process |
| `raw/legacy_tcp_evaluation/` | 18,578 camera frames and metadata from earlier TCP evaluations | Preserve visual recordings; unused by the default trainers |
| `raw/legacy_deliverable_records/`, `raw/legacy_tcp_summaries/` | Earlier deliverable records and TCP summaries | Source provenance |
| `raw/current_runs/` | Structured recordings from the rainy demos | Compare video and control outcomes |
| `raw/replay_snapshots/` | 158 content-deduplicated snapshots of real and synthetic SAC replay | Preserve the available raw experience |
| `prepared/world/` | Historical world-model samples with relative source paths | Train the Transformer directly |
| `prepared/direct/` | Packed history, TCP references, road geometry, and vehicle geometry | Train the differentiable residual policy directly |
| `prepared/sac/` | Real and synthetic replay from the selected final SAC snapshot | Offline SAC updates on a fixed dataset |
| `maps/Town04.xodr` | Town04 OpenDRIVE map | Prepare geometry for custom recordings |

The dataset contains **151,285 files**, totaling approximately **33.0 GB / 30.7 GiB**. Downloaded data and new collection directories are excluded from Git.

## Default Training Inputs

| Model | Training Data | Validation Data | Entry Point |
|---|---:|---:|---|
| World model | 294 episodes, 986,891 windows | 1 episode, 1,167 windows | `python training/train_world_model.py` |
| Differentiable residual policy | 2,365 windows | 406 windows | `python training/train_direct_residual.py` |
| SAC residual policy | 100,000 real + 4,096 synthetic transitions | No offline driving-success validation split | `python training/train_sac_residual.py` |

The direct-policy dataset separates training and validation by episode and route. Historical lane data is sampled at up to 32 windows per episode and 2,048 windows per split, then combined with the saved 20 km/h selections. Duplicate windows and 15 windows with insufficient initial lane-geometry coverage are excluded.

The fixed SAC pools come from the selected final online-training snapshot and include previously generated world-model experience. Real and synthetic experience remain in separate pools, with 48/16 samples drawn per batch by default. Historical snapshots are kept separate because their reward settings and replay contents may differ.

## Data Fields

**World model:** `history [10,7]` contains `vx, vy, r, ax, ay` and the previous executed steering and longitudinal controls. The current `action [2]` is `[steer, throttle-brake]`. The `target [8]` contains the next five-dimensional state and local `Δx, Δy, Δyaw`. Velocity and acceleration use SI units; angles use radians. Displacement is expressed in the vehicle's heading frame before the transition. Normalization statistics are fitted on training data only.

**Differentiable residual policy:** 70 history features, 2 current-action features, 2 reference-point coordinates, 168 road-geometry features, and 42 validity-mask values make 284 inputs. Three planned-lane features bring the final policy input to 287 dimensions. Packed data also includes vehicle bounding boxes, lane segments, frame IDs, and split membership. Rollouts hold the current base action and do not use future TCP commands.

**SAC:** Each transition contains `obs [287]`, normalized residual `action [2]`, reward, `next_obs [287]`, terminated, truncated, gate, and next_gate. Real collision labels come from the simulator. One-step synthetic transitions carry no inferred collision labels; they are truncated at the model horizon and retain bootstrapping.

## Optional Collection Entry Points

| Command | Default Output | Collection Method |
|---|---|---|
| `python data/collect_dataset.py` | `data/collection/world/train/<timestamp>/` | BehaviorAgent dynamics recordings |
| `python data/collect_tcp.py` | `data/collection/tcp/train/` | Frozen TCP references and control readbacks |
| `python data/collect_sac.py` | `data/collection/sac/` | Online SAC interaction, world-model generation, and updates |
