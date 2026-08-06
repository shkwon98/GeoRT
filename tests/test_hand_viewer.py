from types import SimpleNamespace

import numpy as np
import sapien

from geort.env import hand


def test_camera_faces_and_fits_hand_bounds():
    bounds = np.array([
        [[-0.08, -0.06, 0.32], [0.04, 0.03, 0.41]],
        [[0.01, -0.02, 0.35], [0.10, 0.09, 0.47]],
    ])

    pose, near, far = hand._fit_camera_to_bounds(bounds)
    lower = bounds[:, 0].min(axis=0)
    upper = bounds[:, 1].max(axis=0)
    center = (lower + upper) / 2
    radius = np.linalg.norm(upper - lower) / 2
    center_in_camera = (pose.inv() * sapien.Pose(center)).p

    np.testing.assert_allclose(
        center_in_camera[[0, 2]], [3 * radius, 0], atol=1e-7
    )
    horizontal_ndc = -center_in_camera[1] / (
        center_in_camera[0] * np.tan(0.5) * (1920 / 1080)
    )
    np.testing.assert_allclose(horizontal_ndc, 400 / 1920, atol=1e-7)
    assert 0 < near < 2 * radius
    assert far > 4 * radius


def test_mocap_overlay_uses_hand_base_frame():
    scene = sapien.Scene([
        sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()
    ])
    base_pose = sapien.Pose([1, 2, 3], [1, 0, 0, 0])
    link = SimpleNamespace(get_entity_pose=lambda: base_pose)
    model = SimpleNamespace(
        hand=SimpleNamespace(get_links=lambda: [link]),
        base_link_idx=0,
    )
    viewer = hand.HandViewerEnv.__new__(hand.HandViewerEnv)
    viewer.model = model
    viewer.scene = scene
    viewer._mocap_inset = hand.MocapInset()
    viewer._robot_cloud = None
    human_points = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
    robot_points = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    viewer.set_mocap_overlay(human_points, [4, 8], robot_points)

    robot = viewer._robot_cloud
    bone = viewer._mocap_inset._bones[0]
    used = viewer._mocap_inset._used
    np.testing.assert_allclose(
        bone.entity.pose.p, (human_points[0] + human_points[1]) / 2
    )
    np.testing.assert_allclose(
        [marker.entity.pose.p for marker in used], human_points[[4, 8]]
    )
    assert isinstance(robot, sapien.render.RenderPointCloudComponent)
    np.testing.assert_allclose(robot.get_vertices(), robot_points)
    np.testing.assert_allclose(robot.entity.pose.p, base_pose.p)
    np.testing.assert_allclose(robot.entity.pose.q, base_pose.q)


def test_robot_fingertips_render_on_first_frame():
    scene = sapien.Scene([sapien.render.RenderSystem()])
    camera = scene.add_camera("test", 128, 128, 1.0, 0.01, 1.0)
    camera.set_entity_pose(sapien.Pose())
    link = SimpleNamespace(get_entity_pose=sapien.Pose)
    viewer = hand.HandViewerEnv.__new__(hand.HandViewerEnv)
    viewer.model = SimpleNamespace(
        hand=SimpleNamespace(get_links=lambda: [link]),
        base_link_idx=0,
    )
    viewer.scene = scene
    viewer._mocap_inset = SimpleNamespace(set_points=lambda *_: None)
    viewer._robot_cloud = None

    viewer.set_mocap_overlay(
        np.zeros((21, 3), dtype=np.float32),
        [4],
        np.array([[0.2, 0, 0]], dtype=np.float32),
    )
    scene.update_render()
    camera.take_picture()
    rgb = camera.get_picture("Color")[..., :3]

    assert np.any(
        (rgb[..., 0] > 0.9)
        & (rgb[..., 1] < 0.4)
        & (rgb[..., 2] > 0.7)
    )


def test_mocap_bones_are_smooth_meshes_between_landmarks():
    inset = hand.MocapInset()
    points = np.column_stack(
        (
            np.arange(21, dtype=np.float32) * 0.003,
            np.arange(21, dtype=np.float32) * 0.002,
            np.arange(21, dtype=np.float32) * 0.004,
        )
    )

    inset.set_points(points, [4, 8])

    bone = inset._bones[0]
    shape, = bone.render_shapes
    assert isinstance(shape, sapien.render.RenderShapeTriangleMesh)
    axis = bone.entity.pose.to_transformation_matrix()[:3, 0]
    endpoints = np.stack(
        (
            bone.entity.pose.p - shape.scale[0] * axis,
            bone.entity.pose.p + shape.scale[0] * axis,
        )
    )
    np.testing.assert_allclose(endpoints, points[[0, 1]], atol=1e-7)


def test_mocap_camera_matches_robot_projection_from_hand_base():
    inset = hand.MocapInset()
    main_camera = sapien.Pose(
        [0.4, -0.2, 0.7], [0.363401, -0.3801039, 0.16556499, 0.83429549]
    )
    base_pose = sapien.Pose(
        [0.1, 0.2, 0.3], [0.695, 0.0, -0.718, 0.0]
    )
    points = np.array([[0.01, 0.02, 0.03], [-0.04, 0.05, 0.08]])

    inset.set_view(main_camera, base_pose)

    main_projection = np.stack([
        (main_camera.inv() * base_pose * sapien.Pose(point)).p
        for point in points
    ])
    inset_camera = inset.camera.entity_pose
    inset_projection = np.stack([
        (inset_camera.inv() * sapien.Pose(point)).p
        for point in points
    ])
    np.testing.assert_allclose(
        np.diff(inset_projection, axis=0),
        np.diff(main_projection, axis=0),
        atol=1e-7,
    )


def test_mocap_inset_has_fixed_metric_scale():
    inset = hand.MocapInset()
    before = (
        inset.camera.ortho_left,
        inset.camera.ortho_right,
        inset.camera.ortho_bottom,
        inset.camera.ortho_top,
    )

    inset.set_points(np.full((21, 3), 10, dtype=np.float32), [4, 8])

    after = (
        inset.camera.ortho_left,
        inset.camera.ortho_right,
        inset.camera.ortho_bottom,
        inset.camera.ortho_top,
    )
    assert inset.camera.get_mode() == "orthographic"
    np.testing.assert_allclose(before, [-0.14, 0.14, -0.14, 0.14])
    np.testing.assert_allclose(after, before)
    scale_bar = inset._scale_bar
    shape, = scale_bar.render_shapes
    axis = scale_bar.entity.pose.to_transformation_matrix()[:3, 0]
    endpoints = np.stack((
        scale_bar.entity.pose.p - shape.scale[0] * axis,
        scale_bar.entity.pose.p + shape.scale[0] * axis,
    ))
    np.testing.assert_allclose(
        np.linalg.norm(endpoints[1] - endpoints[0]), 0.05, atol=1e-7
    )


def test_mocap_overlay_hides_stale_points():
    cloud = sapien.render.RenderPointCloudComponent(1)
    viewer = hand.HandViewerEnv.__new__(hand.HandViewerEnv)
    viewer._robot_cloud = cloud
    viewer._mocap_inset = hand.MocapInset()
    viewer._mocap_inset.set_points(np.zeros((21, 3), dtype=np.float32), [4])

    viewer.hide_mocap_overlay()

    assert not cloud.is_enabled
    assert not viewer._mocap_inset.visible


def test_viewer_uses_slow_camera_controls(monkeypatch):
    class Window:
        def set_camera_parameters(self, **kwargs):
            self.parameters = kwargs

    class Viewer:
        def __init__(self, plugins):
            self.plugins = plugins
            self.window = Window()

        def set_scene(self, scene):
            self.scene = scene

        def set_camera_pose(self, pose):
            self.camera_pose = pose

    scene = sapien.Scene([
        sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()
    ])
    body = sapien.render.RenderBodyComponent()
    body.attach(sapien.render.RenderShapeSphere(
        0.01, sapien.render.RenderMaterial()
    ))
    sapien.Entity().add_component(body).add_to_scene(scene)
    monkeypatch.setattr(hand, "Viewer", Viewer)

    viewer = hand.HandViewerEnv(SimpleNamespace(scene=scene))

    control = next(
        plugin for plugin in viewer.viewer.plugins
        if isinstance(plugin, hand.ControlWindow)
    )
    assert control.move_speed <= 0.01
    assert control.rotate_speed <= 0.001
    assert control.scroll_speed <= 0.1
