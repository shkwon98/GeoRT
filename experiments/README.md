# Retargeting experiments

The experiment runner keeps mocap, method, and robot selection independent.
Every mocap source is converted to the same 21-landmark canonical frame before
GeoRT or DexPilot sees it. Generated runs live under
`$GEORT_HOME/experiments/<run-id>`.

## Environment

GeoRT uses CPython 3.12. Install simulator dependencies for training and exact
kinematics evaluation, and add the optional upstream DexPilot baseline only
when needed:

```bash
uv sync --extra training
uv sync --extra dexpilot
```

For Wuji experiments, point GeoRT at the existing read-only description:

```bash
export GEORT_HOME=/absolute/path/to/geort-results
export WUJI_HAND_DESCRIPTION=${ROS2_WS}/src/robot/wuji_hand_ros2/wuji_hand_description
```

The example calibration values are placeholders. Measure the rotation, scale,
and palm-normal sign for the physical capture device before collecting data.

## GeoRT workflow

Run collection from the repository root. `raw_webxr.npy` must have shape
`[T, 25, 3]` or `[T, 25, 4, 4]`; an optional timestamp file has shape `[T]`.

```bash
uv run --locked python -m experiments.run collect \
  --config experiments/configs/webxr_wuji_geort.json \
  --input raw_webxr.npy
```

The command prints the created run directory. Use that exact path below:

```bash
uv run --locked --extra training python -m experiments.run train \
  --run-dir "$GEORT_HOME/experiments/<run-id>"
uv run --locked python -m experiments.run export \
  --run-dir "$GEORT_HOME/experiments/<run-id>"
uv run --locked python -m experiments.run infer \
  --run-dir "$GEORT_HOME/experiments/<run-id>"
uv run --locked --extra training python -m experiments.run evaluate \
  --run-dir "$GEORT_HOME/experiments/<run-id>"
```

`<run-id>` is explanatory text, not a shell default. Replace it with the
directory printed by `collect`. `export` and `infer` also accept
`--checkpoint /absolute/checkpoint`.

## DexPilot baseline

DexPilot is an online optimizer and has no training or export step. It uses the
same raw and canonical inputs and emits the same named command artifact:

```bash
uv run --locked --extra dexpilot python -m experiments.run collect \
  --config experiments/configs/webxr_wuji_dexpilot.json \
  --input raw_webxr.npy
uv run --locked --extra dexpilot python -m experiments.run infer \
  --run-dir "$GEORT_HOME/experiments/<dexpilot-run-id>"
uv run --locked --extra training python -m experiments.run evaluate \
  --run-dir "$GEORT_HOME/experiments/<dexpilot-run-id>"
```

## Run artifacts

- `config.json`, `robot.json`, `versions.json`: resolved experiment snapshot
- `raw.npz`, `raw_metadata.json`: immutable device observations and calibration
- `canonical.npy`, `canonical_metadata.json`: derived 21-point frames
- `training.json`, `checkpoints/`: GeoRT training result
- `model.ts`, `model.json`: SAPIEN-independent rollout model and metadata
- `qpos.npz`, `latency.npy`, `metrics.json`: named output and common metrics

Wuji rollout must start its existing launch with `start_retarget:=false`.
Running Wuji's built-in retargeter and the GeoRT bridge together would create
two publishers commanding the same controller and is invalid.

## Wuji ROS 2 rollout

Verify an exported model against Wuji's mock or simulation controller before
enabling real hardware. In one shell, start the existing stack without its
built-in retargeter:

```bash
source /opt/ros/jazzy/setup.bash
source ${ROS2_WS}/install/setup.bash
export WUJI_HAND_DESCRIPTION=${ROS2_WS}/src/robot/wuji_hand_ros2/wuji_hand_description

ros2 launch wuji_hand_demo vr.launch.py start_retarget:=false
```

In another shell, run the GeoRT-owned bridge on the existing topics:

```bash
source /opt/ros/jazzy/setup.bash
source ${ROS2_WS}/install/setup.bash
export WUJI_HAND_DESCRIPTION=${ROS2_WS}/src/robot/wuji_hand_ros2/wuji_hand_description
cd ${GEORT_ROOT}

python3 -m experiments.robots.wuji.ros --ros-args \
  -p model:=/absolute/path/to/model.ts \
  -p metadata:=/absolute/path/to/model.json \
  -p side:=right
```

Use `side:=left` in a separate process for the left hand. Before touching
hardware, inspect both topic endpoints and one command:

```bash
ros2 topic info -v /teleop/human/hand_right/pose
ros2 topic info -v /control/hand_right/hand_right_controller/joint_trajectory
timeout 10 ros2 topic echo --once \
  /control/hand_right/hand_right_controller/joint_trajectory
```

There must be one GeoRT command publisher, no built-in retargeting publisher,
and exactly 20 `right_`-prefixed joint names in the configured order. Every
position must be finite and within `model.json` limits. Move the mock hand
through open, pinch, and closed poses; reject rollout on any name/order error,
non-finite value, duplicate publisher, or uncontrolled motion. Stopping the
GeoRT process must immediately stop new trajectory commands.
