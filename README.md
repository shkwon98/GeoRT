# Geometric Retargeting

[![CC BY-NC 4.0 License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the code for the paper “Geometric Retargeting: A
Principled, Ultrafast Neural Hand Retargeting Algorithm.”

![Demo GIF](./images/demo.gif)

## Installation

GeoRT uses CPython 3.12 and [uv](https://docs.astral.sh/uv/). Install only the
extras needed for the current task:

```bash
uv sync --extra training                         # Lightning, SAPIEN, Open3D
uv sync --extra training --extra manus           # Manus + SAPIEN
uv sync --extra training --extra mediapipe       # camera + SAPIEN
uv sync --extra training --extra dexpilot        # DexPilot baseline + SAPIEN
uv sync --all-extras --group dev                 # full development environment
```

ROS 2 and its message packages remain system-provided. `uv.lock` is the
reproducible Python dependency record.

## Workflow

GeoRT has four user-facing commands:

```text
collect   live-record or import a mocap dataset
train     train one GeoRT run
rollout   view replay or live retargeting in SAPIEN
evaluate  run batch inference and write metrics
```

The bundled robot names are `allegro_left`, `allegro_right`, `wuji_left`, and
`wuji_right`. Manus, MediaPipe, and MetaQuest all enter the same canonical
21-landmark coordinate frame before training or inference.

### 1. Configure and collect mocap

Keep machine-local capture settings under the ignored `.geort/configs/`
directory:

```bash
mkdir -p .geort/configs
```

Example `.geort/configs/manus_right.json`:

```json
{
  "mocap": "manus",
  "hand_side": "right",
  "topic": "/manus_quats",
  "calibration": {
    "scale": 1.0,
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "outward_sign": 1
  }
}
```

MediaPipe accepts `"camera": "realsense"` or `"camera": "webcam"` plus an
optional `"device_index"`. MetaQuest accepts a PoseArray `"topic"` and expects
25 poses. Calibrate `scale`, `rotation`, and `outward_sign` for the actual
device instead of assuming the identity example is correct.

Live collection:

```bash
uv run --locked --extra manus geort collect \
  --mocap manus \
  --dataset manus_right_001 \
  --config .geort/configs/manus_right.json
```

Press Ctrl+C to finish Manus or MetaQuest capture. In the MediaPipe window,
press `s` to record, `e` to pause, and `q` to save and quit.

Import an existing NumPy recording with the same command plus `--input`.
Use `--timestamps` when timestamps are stored separately; otherwise import
assumes 30 Hz.

```bash
uv run --locked geort collect \
  --mocap manus \
  --dataset manus_right_001 \
  --config .geort/configs/manus_right.json \
  --input recordings/manus_right.npy
```

### 2. Train GeoRT

```bash
uv run --locked --extra training geort train \
  --dataset manus_right_001 \
  --robot allegro_right \
  --run-id manus_allegro_geort_seed0 \
  --device cuda \
  --epoch 500 \
  --save-every 50
```

For a robot without a complete cache, the first run also generates kinematic
and collision data and trains the reusable FK and collision models. Lightning
stores `best.ckpt` by lowest validation loss, `last.ckpt` for the final state,
and an additional numbered checkpoint every `--save-every N` epochs. Set it to
`0` to disable numbered checkpoints. Inference selects `best.ckpt` and falls
back to `last.ckpt` when validation is disabled.

Resume an interrupted run from `last.ckpt`; `--epoch` is the new total epoch
count, and the dataset, robot, seed, and training settings come from the run:

```bash
uv run --locked --extra training geort train \
  --resume manus_allegro_geort_seed0 \
  --epoch 500 \
  --device cuda
```

### 3. Evaluate

GeoRT evaluation runs inference and metrics together:

```bash
uv run --locked --extra training geort evaluate \
  --run manus_allegro_geort_seed0 \
  --device cuda
```

DexPilot is an online baseline and has no training command. Its evaluation
creates a run directly:

```bash
uv run --locked --extra training --extra dexpilot geort evaluate \
  --dataset manus_right_001 \
  --robot allegro_right \
  --method dexpilot \
  --run-id manus_allegro_dexpilot_seed0
```

### 4. Roll out in SAPIEN

Replay the run's dataset:

```bash
uv run --locked --extra training geort rollout \
  --run manus_allegro_geort_seed0 \
  --source replay \
  --device cuda
```

Use the same learned run with live input:

```bash
uv run --locked --extra training --extra manus geort rollout \
  --run manus_allegro_geort_seed0 \
  --source live \
  --config .geort/configs/manus_right.json \
  --device cuda
```

The main view shows the robot and predicted fingertips. The fixed-scale inset
shows the canonical mocap skeleton, selected landmarks, coordinate axes, and a
50 mm scale bar. This workflow commands only the SAPIEN hand; it does not send
commands to physical hardware or a ROS hand controller.

## Local artifacts

All generated data stays under the project-local, Git-ignored `.geort/`:

```text
.geort/
├── configs/
├── data/
│   └── manus_right_001/
│       ├── raw.npz
│       ├── canonical.npy
│       └── metadata.json
├── cache/
│   └── allegro_right-<fingerprint>/
│       ├── kinematics.npz
│       ├── fk.pth
│       ├── collisions.npz
│       └── collision.pth
└── runs/
    └── manus_allegro_geort_seed0/
        ├── config.json
        ├── checkpoints/
        │   ├── best.ckpt
        │   ├── last.ckpt
        │   └── epoch=0049.ckpt
        └── outputs/
            ├── qpos.npz
            ├── latency.npy
            └── metrics.json
```

A dataset is independent of the target robot and method, so the same capture
can be reused across all robot/method combinations. A run represents one
dataset, robot, method, and seed. The cache fingerprint covers the robot
configuration, URDF, and referenced mesh contents.

## Adding a robot

Use [the Allegro right config](./geort/config/allegro_right.json) or
[the Wuji right config](./geort/config/wuji_right.json) as a reference. Keep a
custom config and its assets in a project-relative location; absolute user
paths are rejected from persisted run metadata. The config must define the
hand side, URDF, base link, unique joint order, and fingertip link/joint groups.

## Notes

- GeoRT assumes the robot fingertip workspace resembles the human hand
  workspace. Keep the configured joint ranges physically meaningful.
- Complex collision meshes can make SAPIEN loading unstable. Prefer simple
  collision geometry when importing a new hand.
- The canonical frame uses +X as the outward palm normal, +Y toward the thumb,
  and +Z from wrist toward the middle-finger MCP. Positions use metres.
- MediaPipe is convenient for debugging but is more sensitive to scale and
  wrist-pose shifts than glove or VR tracking.

## Contact and license

For questions, open an issue or contact `zhaohengyin@cs.berkeley.edu`.
The project is licensed under CC BY-NC 4.0.
