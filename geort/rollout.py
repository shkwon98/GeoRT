import time

import numpy as np

from geort.artifacts import load_dataset, load_run
from geort.methods import make_method
from geort.mocap.adapters import adapt_observation
from geort.robots import load_robot
from geort.schema import validate_calibration
from geort.utils.config_utils import load_json


def _live_source(mocap, config):
    if mocap == "manus":
        from geort.mocap.manus_mocap import ManusMocap

        return ManusMocap(topic=config.get("topic", "/manus_quats"))
    if mocap == "mediapipe":
        from geort.mocap.mediapipe_mocap import MediaPipeMocap

        return MediaPipeMocap(
            camera=config.get("camera", "realsense"),
            device_index=int(config.get("device_index", 0)),
            hand_side=config.get("hand_side", "right"),
        )
    if mocap == "metaquest":
        from geort.mocap.metaquest_mocap import MetaQuestMocap

        hand_side = config.get("hand_side", "right")
        topic = config.get(
            "topic", f"/teleop/human/hand_{hand_side}/pose"
        )
        return MetaQuestMocap(topic)
    raise ValueError(f"unsupported mocap: {mocap}")


def collect_live(mocap, config_path):
    config = load_json(config_path)
    source = _live_source(mocap, config)
    observations = []
    timestamps = []
    try:
        while True:
            result = source.get()
            if result["status"] == "quit":
                break
            if result["status"] == "no data":
                time.sleep(0.001)
                continue
            if result["status"] == "recording" and result["result"] is not None:
                observations.append(np.asarray(result["result"]).copy())
                timestamps.append(float(result.get("timestamp", time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    if not observations:
        raise ValueError("no mocap frames were recorded")
    return np.stack(observations), np.asarray(timestamps, dtype=np.float64)


def apply_frame(frame, method, hand, viewer, human_ids):
    if not frame.valid:
        viewer.hide_mocap_overlay()
        return None
    command = method.infer(frame)
    hand.set_qpos_target(command.qpos)
    robot_points = hand.keypoint_from_qpos(command.qpos, ret_vec=True)
    viewer.set_mocap_overlay(frame.points, human_ids, robot_points)
    return command


def _replay_interval(frames, index):
    following = (index + 1) % len(frames)
    if following == 0:
        return 1 / 30
    return max(1 / 240, float(frames[following].timestamp - frames[index].timestamp))


def _live_calibration(config_path, metadata):
    config = load_json(config_path)
    if config.get("mocap", metadata["mocap"]) != metadata["mocap"]:
        raise ValueError("live mocap does not match the run dataset")
    if config.get("hand_side") != metadata["hand_side"]:
        raise ValueError("live hand_side does not match the run dataset")
    calibration = config.get("calibration", {})
    scale, rotation, outward_sign = validate_calibration(
        calibration.get("scale"),
        calibration.get("rotation"),
        calibration.get("outward_sign"),
    )
    return config, {
        "scale": scale,
        "rotation": rotation,
        "outward_sign": outward_sign,
    }


def run(run_id, source, device=None, config_path=None):
    run_dir, config = load_run(run_id)
    robot = load_robot(config["robot"])
    if config.get("robot_fingerprint") != robot["robot_fingerprint"]:
        raise ValueError("robot configuration or assets changed after run creation")
    _, frames, metadata = load_dataset(config["dataset"])
    if metadata["hand_side"] != robot["hand_side"]:
        raise ValueError("dataset hand_side does not match robot")

    from geort.env.hand import HandKinematicModel

    method = make_method(config, robot, run_dir, device)
    hand = HandKinematicModel.build_from_config(robot, render=True)
    tips = robot["fingertip_link"]
    hand.initialize_keypoint(
        [tip["link"] for tip in tips],
        [tip["center_offset"] for tip in tips],
    )
    human_ids = [tip["human_hand_id"] for tip in tips]
    viewer = hand.get_viewer_env()
    mocap = None
    try:
        if source == "replay":
            index = 0
            next_frame = time.perf_counter()
            while not viewer.viewer.closed:
                viewer.update()
                now = time.perf_counter()
                if now < next_frame:
                    continue
                apply_frame(frames[index], method, hand, viewer, human_ids)
                following = (index + 1) % len(frames)
                next_frame = now + _replay_interval(frames, index)
                index = following
        else:
            if not config_path:
                raise ValueError("live rollout requires --config")
            capture, calibration = _live_calibration(config_path, metadata)
            mocap = _live_source(metadata["mocap"], capture)
            while not viewer.viewer.closed:
                viewer.update()
                result = mocap.get()
                if result["status"] == "quit":
                    break
                if result["status"] == "no data":
                    continue
                if result["status"] != "recording" or result["result"] is None:
                    viewer.hide_mocap_overlay()
                    continue
                frame = adapt_observation(
                    metadata["mocap"],
                    result["result"],
                    result.get("timestamp", time.monotonic()),
                    metadata["hand_side"],
                    calibration,
                )
                apply_frame(frame, method, hand, viewer, human_ids)
    finally:
        if mocap is not None:
            mocap.close()
        viewer.viewer.close()
