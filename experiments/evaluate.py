import itertools

import numpy as np


def evaluate_trajectory(frames, qpos, joint_names, latencies, robot_spec, hand):
    frames = list(frames)
    qpos = np.asarray(qpos)
    latencies = np.asarray(latencies, dtype=np.float64)
    expected_names = tuple(robot_spec["joint_order"])
    if tuple(joint_names) != expected_names:
        raise ValueError("joint_names must exactly match the robot specification")
    if qpos.shape != (len(frames), len(expected_names)):
        raise ValueError("qpos must have shape (frames, joints)")
    if (
        latencies.shape != (len(frames),)
        or not np.isfinite(latencies).all()
        or np.any(latencies < 0)
    ):
        raise ValueError("latencies must be finite, non-negative, and match frames")

    lower, upper = (np.asarray(limit) for limit in hand.get_joint_limit())
    if lower.shape != (len(expected_names),) or upper.shape != lower.shape:
        raise ValueError("hand joint limits must match joint_names")
    finite_values = np.isfinite(qpos)
    finite_frames = finite_values.all(axis=1)
    valid_frames = finite_frames & np.array([frame.valid for frame in frames])
    limit_violations = finite_values & ((qpos < lower) | (qpos > upper))

    tips = robot_spec["fingertip_link"]
    hand.initialize_keypoint(
        [tip["link"] for tip in tips],
        [tip["center_offset"] for tip in tips],
    )
    human_ids = [tip["human_hand_id"] for tip in tips]
    coverage_errors = []
    pinch_errors = []
    collisions = []
    for index in np.flatnonzero(valid_frames):
        human_tips = frames[index].points[human_ids]
        robot_tips = np.asarray(
            hand.keypoint_from_qpos(qpos[index], ret_vec=True)
        )
        if robot_tips.shape != human_tips.shape or not np.isfinite(robot_tips).all():
            raise ValueError("robot fingertip positions must match configured tips")
        coverage_errors.extend(np.linalg.norm(robot_tips - human_tips, axis=1))
        for left, right in itertools.combinations(range(len(tips)), 2):
            human_distance = np.linalg.norm(human_tips[left] - human_tips[right])
            if human_distance < 0.015:
                robot_distance = np.linalg.norm(
                    robot_tips[left] - robot_tips[right]
                )
                pinch_errors.append(abs(robot_distance - human_distance))
        collisions.append(bool(hand.is_self_collision(qpos[index])))

    finite_qpos = qpos[finite_frames]
    total_variation = (
        np.abs(np.diff(finite_qpos, axis=0)).sum() / (len(finite_qpos) - 1)
        if len(finite_qpos) > 1
        else 0.0
    )
    return {
        "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
        "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
        "joint_limit_violations": int(limit_violations.sum()),
        "nonfinite_violations": int((~finite_values).sum()),
        "fingertip_coverage_error_m": float(np.mean(coverage_errors))
        if coverage_errors
        else 0.0,
        "pinch_preservation_error_m": float(np.mean(pinch_errors))
        if pinch_errors
        else 0.0,
        "self_collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "joint_total_variation": float(total_variation),
    }
