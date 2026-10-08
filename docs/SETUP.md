# Environment Setup, Rainy Demos, and Retraining

Run all commands from the project root, except the CARLA server startup command. For training only, complete Section 1 and continue to Section 4. Sections 2 and 3 cover CARLA recording. If you already have a working `b2d_robust` environment, start at Section 2. Collected data goes into `data/collection/`; new checkpoints and videos go into `outputs/`.

## 1. Create the Environment

Run from the project root:

```bash
conda env create -f environment.yml
conda activate wm_res_control
```

`environment.yml` installs Python 3.8, the CUDA 12.1 build of PyTorch, and the Python dependencies for demos, training, and evaluation. It targets Linux x86_64 with an NVIDIA GPU. Install the CARLA server separately as described below.

## 2. Install and Start CARLA

Download and extract the Linux binary package from the [CARLA 0.9.15 release page](https://github.com/carla-simulator/carla/releases/tag/0.9.15).

For other CARLA versions, install the corresponding Python client package so that `import carla` succeeds.

Start a dedicated server in a separate terminal:

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -quality-level=Epic -carla-rpc-port=2000
```

CARLA opens a window by default. During the demo, its view follows the recording camera. Add `-RenderOffScreen` for background recording. Keep rendering enabled: TCP requires valid camera images, which are unavailable in `no_rendering_mode`.

## 3. Reproduce the CARLA Routes

After starting the CARLA server, run the rainy collision route from the project root:

```bash
python evaluation/record_demo.py --output outputs/rain_collision
```

The script runs **TCP, TCP + differentiable residual, and TCP + world-model-assisted SAC** sequentially on the same server and produces a three-panel comparison video. The CARLA view follows the current vehicle's recording camera and returns to its original pose after each recording.

The default route is Town04 route 25951. TCP controls the driving speed, with its additional low-speed throttle cap removed and a shared terminal slowdown for all controllers. The environment uses `MidRainyNoon`, a tire-friction coefficient of 0.5 on all four wheels, and synchronous 20 Hz simulation. In the recorded rainy demo, TCP collided and both residual controllers reached the destination.

Video output:

```text
outputs/rain_collision/autonomous/render/comparison_three_way.mp4
```

Run the 20 km/h route separately:

```bash
python evaluation/record_demo.py --scenario speed20 --output outputs/rain_speed20
```

This uses Town04 route 24785 with the same rainy weather and three-controller comparison, a 20 km/h cruise target, and terminal slowdown. The video is saved to `outputs/rain_speed20/speed20/render/comparison_three_way.mp4`.

## 4. Download the Training Data

**Dataset: [Google Drive](https://drive.google.com/file/d/1jsaaDUzWepih2cwlcuvvtHdk_RpckVN8/view?usp=sharing).** Download `WM_Res_Control_Data.tar.gz`, place it in `data/`, and run from the project root:

```bash
tar -xzf data/WM_Res_Control_Data.tar.gz -C data
```

Expected directory structure:

```text
data/
├── collect_dataset.py, collect_tcp.py, collect_sac.py, and other source files
├── raw/                    Raw driving records, geometry, samples, and replay snapshots
├── prepared/
│   ├── world/              train.npz, validation.npz
│   ├── direct/             windows.pt
│   └── sac/                replay.pt
└── maps/Town04.xodr
```

## 5. World Model

### Data Collection

CARLA's BehaviorAgent drives in Town04 with tire friction 0.5 and a 20 Hz simulation rate. Each frame records the vehicle state, executed action, next state, and termination flags. Training and validation are split by episode.

```bash
python data/collect_dataset.py
python data/collect_dataset.py --seed 2027 --record data/collection/world/validation
```

The first command saves to `data/collection/world/train/<timestamp>/`; the second saves to `data/collection/world/validation/<timestamp>/`. Run them sequentially with CARLA running.

For custom collections, build samples using the two episode directories printed by the recorder. Replace `TRAIN_EPISODE` and `VALIDATION_EPISODE` with their actual paths:

```bash
python data/build_samples.py --train TRAIN_EPISODE --validation VALIDATION_EPISODE \
  --output data/prepared/world_custom
```

You can list multiple training episodes after `--train`.

### Train with Downloaded Data

```bash
python training/train_world_model.py
```

By default, the trainer reads 986,891 training windows and 1,167 validation windows from `data/prepared/world/`, trains for 100 epochs with batch size 64, and selects CUDA or CPU automatically. Outputs are saved to `outputs/training/world/`.

- `best.pt`: the model with the best validation metric.
- `last.pt`: the final training checkpoint.
- Normalization uses training-data statistics. The model uses 10 frames of history to predict the next state and motion increments.

Override only the parameters you want to change, for example for a shorter trial or a new output directory:

```bash
python training/train_world_model.py --epochs 10 --output outputs/world_trial
```

Evaluate the new model's multi-step predictions:

```bash
python evaluation/evaluate_world_model.py --data data/prepared/world \
  --checkpoint outputs/training/world/best.pt --output outputs/world_eval
```

## 6. Differentiable Residual Policy

### Data Collection

Drive with frozen TCP and record its base actions, reference trajectory, vehicle states, executed-control readbacks, planned route, and vehicle bounding box.

```bash
python data/collect_tcp.py
python data/collect_tcp.py --route-id 25300 --output data/collection/tcp/validation
```

The default training route is 24795 and the validation route is 25300. Both use rainy weather and friction 0.5, saving to `data/collection/tcp/train/` and `data/collection/tcp/validation/`, respectively.

After a custom collection, read the map and build windows offline:

```bash
python data/prepare_direct.py
```

By default, this uses `data/maps/Town04.xodr`, the two collection directories above, and the released world model. It writes to `data/prepared/direct_custom/`.

### Train with Downloaded Data

```bash
python training/train_direct_residual.py
```

The trainer reads `data/prepared/direct/`, freezes `demo/models/world.pt`, and initializes a new residual policy with a zero-output head. Defaults are 20 epochs, batch size 8, and learning rate 0.0001. It saves `best.pt` and `last.pt` to `outputs/training/direct/`.

During training, the current TCP base action is held for a 0.5-second model rollout. Predicted states and residual actions evolve step by step, with gradients from trajectory deviation, sideslip, and road/lane departure. TCP and world-model parameters remain frozen. Residual limits are 0.05 for steering and 0.02 for longitudinal control.

To use a newly trained world model with matching normalization:

```bash
python training/train_direct_residual.py --world outputs/training/world/best.pt
```

Add `--init-policy demo/models/direct.pt` to fine-tune the released residual policy. The default starts without historical residual weights. The prepared windows combine historical lane-training and 20 km/h fine-tuning data; this simplified recipe does not reproduce every stage that produced the released model.

## 7. World-Model-Assisted SAC Residual Policy

### Data Collection and Online Training

SAC executes residual actions in CARLA and stores real transitions in a real replay buffer. The frozen world model predicts one-step transitions from real states to generate synthetic experience. Collection and updates to the actor, twin critics, and entropy coefficient run together.

```bash
python data/collect_sac.py
```

By default, this uses the released world model and runs two episodes across the demo routes, each capped at 400 steps, saving to `data/collection/sac/`. Each episode's `sac_last.pt` includes model weights, optimizer states, and replay buffers.

For longer online runs, specify the budget explicitly:

```bash
python data/collect_sac.py --episodes 20 --steps 1200 --output data/collection/sac_more
```

This starts online interaction and learning. Resume a full online checkpoint with `--resume PATH`. The released actor is stored at `demo/models/sac.pt`.

### Offline Training with Downloaded Data

```bash
python training/train_sac_residual.py
```

The trainer reads `data/prepared/sac/replay.pt`, containing **100,000 real transitions and 4,096 world-model-generated transitions**. It initializes a new actor and critics and performs 10,000 updates with batch size 64, a 75% / 25% real-to-synthetic mixture, and learning rate 0.0003. It runs on CPU using the saved experience, without CARLA interaction or new synthetic generation.

Outputs in `outputs/training/sac/`:

- `actor.pt`: the actor export used by the existing inference and closed-loop evaluation entry points.
- `training.pt`: an offline training snapshot containing critics and optimizer states, with a format separate from online checkpoints.

To change the update budget:

```bash
python training/train_sac_residual.py --updates 20000 --output outputs/sac_trial
```

## 8. Evaluate Your Trained Policies

Start CARLA for this step. Evaluation writes to a new directory while retaining the released models and videos.

```bash
python evaluation/evaluate_closed_loop.py --policy outputs/training/direct/best.pt \
  --weather MidRainyNoon --output outputs/new_direct_pair --route-id 25951 \
  --uncapped-tcp --extend-route-m 250 --stop-at-route-end --steps 1200 --video
python evaluation/evaluate_closed_loop.py --policy outputs/training/sac/actor.pt --sac-policy \
  --weather MidRainyNoon --output outputs/new_sac_pair --route-id 25951 \
  --uncapped-tcp --extend-route-m 250 --stop-at-route-end --steps 1200 --video
```

Use matching routes, friction, and terminal rules for each comparison. Review the recorded videos for collisions, destination completion, and lane departure.
