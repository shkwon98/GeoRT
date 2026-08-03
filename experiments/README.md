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
