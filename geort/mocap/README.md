# Mocap Instructions
In this folder, we provide manus gloves mocap code and a very simple vision based mocap system based on mediapipe for code debugging and quick-start purpose only. DO NOT use Mediapipe (+GeoRT) for real teleop! Mediapipe is very unreliable under wrist movement. The hand scale is constantly changing throughout deployment and this will feed out-of-distribution keypoints to GeoRT!!

## Our Vision-based Mocap Recommendations:
Please use a mocap glove. If you insist on using a vision-based mocap system, consider:
1. Use a more advanced 3D hand detector (e.g. Hamer).
2. Probably mount the camera on your wrist in a fixed, standard way through training and deployment.


## Manus Gloves Mocap.
GeoRT works best with the glove-based mocap. We provide an example based on the Manus gloves.

### Installation
Install the Manus Python dependency:
```
uv sync --extra manus
```
ROS itself remains system-provided.

We need to get Manus gloves and its mocap server installed on a separate windows laptop. Make sure that windows laptop and the host is in the same LAN an ping each other. Then, follows the readme in the manus_client folder to setup. This will build a ROS2 node that can 

After this step, run the following for the right hand in one terminal.
```
ros2 run manus_client manus_right
```

This publishes the existing Manus quaternion stream on `/manus_quats`.
### Deployment

With `manus_right` running, start a live rollout from an existing run:
```
uv run geort rollout --run YOUR_RUN --source live \
  --config .geort/configs/manus_right.json
```
The rollout subscribes to `/manus_quats` directly. No localhost relay process
is required.
