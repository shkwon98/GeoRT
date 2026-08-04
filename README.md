# Geometric Retargeting

[![CC BY-NC 4.0 License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

Welcome! This repository contains the code for the paper "Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm".

![Demo GIF](./images/demo.gif)
## Installation
Install [uv](https://docs.astral.sh/uv/) first. Choose one environment:

```bash
uv sync --no-dev       # Lightweight core runtime
uv sync --all-extras   # Full research environment, including tests
```

`uv.lock` is the reproducible dependency record. To keep the environment small,
select only the needed extras in one command, for example
`uv sync --extra training --extra mediapipe`. `training` provides SAPIEN/Open3D
(Linux x86_64); `mediapipe` provides the camera demo; `manus` provides the
Manus Python dependency; and `dexpilot` provides the baseline. ROS itself
remains system-provided.

GeoRT supports CPython 3.12.

For experiments across WebXR, Manus, MediaPipe, Allegro, Wuji, GeoRT, and the
DexPilot baseline, see the [multi-device experiment guide](experiments/README.md).

GeoRT keeps generated recordings and checkpoints outside an installed wheel. Set `GEORT_HOME` to choose their writable location:

```bash
export GEORT_HOME="$PWD/.geort"
```

Without `GEORT_HOME`, a source checkout uses its `data/` and `checkpoint/` directories; an installed wheel uses `~/.local/share/geort/` (or `$XDG_DATA_HOME/geort/`).
## Quick Overview
Upon completion, you will be able to train GeoRT and deploy the checkpoint in a clean and straightforward way. 
### Training (1-2min):
```bash
uv run python -m geort.trainer --hand allegro_right --human-data human_alex \
  --tag geort_1 --device cpu --epoch 50
```

This writes an experiment directory such as `$GEORT_HOME/checkpoint/allegro_right_YYYY-MM-DD_HH-MM-SS_geort_1/`.

### Deploy in code
```python
import geort

model = geort.load_model(
    "/absolute/path/to/allegro_right_YYYY-MM-DD_HH-MM-SS_geort_1",
    epoch=None,  # last.pth; use epoch=0 for epoch_0.pth
    device="cpu",
)
mocap = ...
qpos = model.forward(mocap.get())
```
But before this, we need to complete some one-time system setup steps outlined below.

**Useful Links**: [Notes and Troubleshooting](#notes-and-troubleshooting)
## Getting Started
We use the native Allegro Hand as an example. 

### Step 1: Import your robot hand (one-time setup).
Note: For the Allegro Hand, you can actually skip this step. However, please follow it if you want to import a customized robot hand.

We just need to complete a quick setup process outlined below:

1. Keep your custom robot URDF and meshes in a directory you control. The built-in Allegro assets are bundled with the package.
2. Create a config file such as ``your_robot_name.json``. Pass its path with ``--hand /path/to/your_robot_name.json``; do not write into an installed package directory. Below is an abbreviated example; use [the Allegro right-hand config](./geort/config/allegro_right.json) as the complete reference.

```
{
    "name": "allegro_right",  
    "urdf_path": "/absolute/path/to/your_robot.urdf",
    "base_link": "base_link",
    "joint_order": [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    ],
    "fingertip_link": [
        {
            "name": "index",
            "link": "link_4.0_tip",
            "joint": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "center_offset": [0.0, 0.0, 0.0],
            "human_hand_id": 8,
        },
        ...
    ]
}

```
Now, you can run this command to visualize your hand.
```bash
uv run python -m geort.env.hand --hand /path/to/your_robot_name.json
```
such as 
```bash
uv run python -m geort.env.hand --hand allegro_right
```
<span style="color:red"> If there is any segmentation error, please simplify the collision meshes or just remove all the `<collision>` fields in your URDF. </span> See the [Notes and Troubleshooting](#notes-and-troubleshooting) section.

### Step 2: Collect human hand mocap data.
Now we need to collect some human hand data for training the retargeting model. ``human_alex`` is bundled as an example. New recordings saved through GeoRT go to the writable data directory described above.

```
import geort
import time

# Dataset Name
data_output_name = "human" # TODO(): Specify a name for this (e.g. your name)

# Your data collection loop.
mocap = YourAwesomeMocap() # TODO(): your mocap system.
                           # Define a mocap.get() method.
                           # Apologies, you still have to do this...
 
data = []

for step in range(5000):       # collect 5000 data points.
    hand_keypoint = mocap.get() # mocap.get() return [N, 3] numpy array.
    data.append(hand_keypoint)
    
    time.sleep(0.01)            # take a short break.

# finish data collection.
geort.save_human_data(data, data_output_name)
```
Use ``geort.save_human_data`` API -- this can simplify your effort in specifying the path. This dataset can be reloaded later using **data_output_name**. 

During the data collection process, try to 1. fully stretch each finger and explore its fingertip moving range and 2. perform pinch grasps. Ensure that your fingers feel natural and comfortable—since during teleoperation deployment, you will use these recorded gestures to control the robot! Please avoid any unnatural or strained movements.

We understand that most users likely have their own mocap systems. However, for demonstration purposes, we provide a simple mocap solution based on MediaPipe. Please note, this is intended only for demo use and not for deployment; we will explain this in more detail later.

```bash
uv run python -m geort.mocap.mediapipe_mocap --name human
```
Run `uv sync --extra mediapipe` first. The command generates a dataset named ``human``. Refer to the file for instructions. When you see the pop-up window, press ``s`` to start recording and ``q`` to finish.

**Note:** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 

### Step 3: Train the Model
Assuming you saved ``your_robot_name.json`` somewhere writable as described in Step 1, and set ``data_output_name`` to ``human`` in Step 2, run the following command. ``TAG`` is appended to the generated experiment directory.

```bash
uv run python -m geort.trainer --hand /path/to/your_robot_name.json --human-data human \
  --tag TAG --device cuda --epoch 50
```

Let it train for about 30–50 epochs (approximately 1–2 minutes). You can press Ctrl+C to stop early if you wish. 

If this is the first time you’re training for a new hand, an additional 5 minutes will be needed to train the neural FK model — this only happens once. The bundled Allegro FK checkpoint loads directly.

For demo purpose, ``human_alex`` is bundled with GeoRT. For adapting it to a right Allegro hand, just run

```bash
uv run python -m geort.trainer --hand allegro_right --human-data human_alex \
  --tag geort_1 --device cpu --epoch 50
```
This creates ``<checkpoint-root>/allegro_right_<timestamp>_geort_1/``. Resume the same experiment (with the same hand, data, seed, validation split, and loss settings) with:

```bash
uv run python -m geort.trainer --hand allegro_right --human-data human_alex \
  --resume /absolute/path/to/allegro_right_<timestamp>_geort_1 \
  --device cpu --epoch 100
```

### Step 4: Deploy!
Ok, now we are all set. Use the following code to import and deploy the trained model. 

```python
import geort

experiment_dir = '/absolute/path/to/allegro_right_<timestamp>_geort_1'
model = geort.load_model(experiment_dir, epoch=None, device='cpu')

mocap = YourAwesomeMocap()      # TODO: your mocap.
robot = YourRobustRobotHand()   # TODO: your robot.

while True:
    qpos = model.forward(mocap.get()) # This is the retargeted qpos. 
                                      # (Note: unnormalized joint angle)
    robot.command(qpos)               # execute!

```
The unified evaluator supports replay, MediaPipe, and Manus inputs. We
recommend a glove-based mocap system instead of MediaPipe because vision-based
mocap has significant input distribution shift during deployment.

The simplest way for testing is to use the replay evaluation as below. This will show the retargeted trajectory in the viewer. 
```bash
uv run python -m geort.mocap.evaluation --mocap replay \
  --hand allegro_right --ckpt-tag /absolute/path/to/YOUR_EXPERIMENT \
  --data YOUR_TRAINING_DATA
```
For instance, if ``human`` is in the configured writable data directory
```bash
uv run python -m geort.mocap.evaluation --mocap replay \
  --hand allegro_right --ckpt-tag /absolute/path/to/YOUR_EXPERIMENT \
  --data human
```
## Contributing
Feel free to contribute your robot model and mocap system to the GeoRT repository!

## [Notes and Troubleshooting](#notes-and-troubleshooting)
1. **Note:Joint Range Clipping.** One core assumption of GeoRT is that the motion range of robot fingertips resembles that of human hands. To maintain realistic fingertip poses, please clip your robot's joint movement ranges appropriately and avoid unnatural configurations.

2. **Simulation Errors with New Hands?** Simulation errors (segmentation fault) may occur when importing new robotic hands (e.g. [this issue](https://github.com/facebookresearch/GeoRT/issues/7)), and this is usually caused by collision meshes. To avoid this, ensure that the collision meshes defined in your URDF are simple—such as boxes or basic convex shapes. Alternatively, you can remove all <collision> elements from the URDF to eliminate these issues entirely. 

3. **Hand Coordinate System (Frame) Convention** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 


## Contact Us
For any inquiries, please open an issue or contact the authors via email at ``zhaohengyin@cs.berkeley.edu``
<!-- ## Bibliography -->

## License
CC-by-NC license
