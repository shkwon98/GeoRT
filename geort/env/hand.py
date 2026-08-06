# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien
from sapien.utils import Viewer
from sapien.utils.viewer.control_window import ControlWindow
from sapien.utils.viewer.plugin import Plugin

from geort.utils.config_utils import get_config
from geort.utils.hand_utils import (
    check_contact,
    get_active_joint_indices,
    get_active_joints,
    get_entity_by_name,
)
from geort.utils.path import resolve_resource_path

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
)

_CAMERA_ROTATION = np.array(
    [0.363401, -0.3801039, 0.16556499, 0.83429549])
_CONTROL_WIDTH = 400
_VIEW_RESOLUTION = (1920, 1080)
_VIEW_FOVY = 1.0


def _make_render_body(scene, shape):
    body = sapien.render.RenderBodyComponent()
    body.attach(shape)
    sapien.Entity().add_component(body).add_to_scene(scene)
    return body


def _set_segment(body, start, end):
    start = np.asarray(start)
    end = np.asarray(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        body.disable()
        return
    body.entity.set_pose(sapien.Pose(
        (start + end) / 2,
        sapien.math.shortest_rotation([1, 0, 0], direction),
    ))
    body.enable()


def _fit_camera_to_bounds(bounds):
    """Fit a +X-forward SAPIEN camera to bounds shaped (N, 2, 3)."""
    lower = bounds[:, 0].min(axis=0)
    upper = bounds[:, 1].max(axis=0)
    center = (lower + upper) / 2
    radius = np.linalg.norm(upper - lower) / 2
    rotation = _CAMERA_ROTATION
    camera_axes = sapien.Pose(q=rotation).to_transformation_matrix()[:3, :3]
    forward, left = camera_axes[:, :2].T
    distance = 3 * radius
    panel_offset = (
        distance
        * np.tan(_VIEW_FOVY / 2)
        * _CONTROL_WIDTH
        / _VIEW_RESOLUTION[1]
    )
    pose = sapien.Pose(
        center - distance * forward + panel_offset * left, rotation
    )
    return pose, max(0.001, 0.1 * radius), max(1.0, 10 * radius)


class MocapInset(Plugin):
    def __init__(self):
        self.scene = sapien.Scene([sapien.render.RenderSystem()])
        self.scene.set_ambient_light([1, 1, 1])
        cyan = sapien.render.RenderMaterial(
            base_color=[0.05, 0.75, 1, 1], emission=[0.02, 0.3, 0.4, 1]
        )
        green = sapien.render.RenderMaterial(
            base_color=[0.2, 1, 0.35, 1], emission=[0.08, 0.4, 0.14, 1]
        )
        white = sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 1], emission=[0.4, 0.4, 0.4, 1]
        )
        axis_materials = [
            sapien.render.RenderMaterial(
                base_color=color, emission=np.asarray(
                    color) * [0.4, 0.4, 0.4, 1]
            )
            for color in (
                [1, 0.2, 0.2, 1],
                [0.2, 1, 0.2, 1],
                [0.2, 0.4, 1, 1],
            )
        ]
        primitive = sapien.render.RenderShapeCylinder(1, 1, cyan)
        cylinder = (
            primitive.vertices,
            primitive.triangles,
            primitive.vertex_normal,
            primitive.vertex_uv,
        )

        self._cylinder = cylinder
        self._bone_material = cyan
        self._bones = [None] * len(_HAND_CONNECTIONS)
        self._joints = [
            _make_render_body(
                self.scene, sapien.render.RenderShapeSphere(0.0025, cyan)
            )
            for _ in range(21)
        ]
        self._used = None
        self._used_material = green
        self._axes = [
            self._make_segment(material, 0.025, 0.0012)
            for material in axis_materials
        ]
        self._scale_bar = self._make_segment(white, 0.05, 0.0015)

        self.camera = self.scene.add_camera(
            "mocap", 640, 640, 1.0, 0.01, 2.0
        )
        self.camera.set_orthographic_parameters(0.01, 2.0, 0.14)
        self.set_view(sapien.Pose(q=_CAMERA_ROTATION), sapien.Pose())
        self.visible = False
        self.ui_window = None
        self.ui_picture = None

    def _make_segment(self, material, length, radius):
        shape = sapien.render.RenderShapeTriangleMesh(
            *self._cylinder, material
        )
        shape.scale = [length / 2, radius, radius]
        return _make_render_body(self.scene, shape)

    def set_view(self, main_camera_pose, base_pose):
        rotation = (base_pose.inv() * main_camera_pose).q
        rotation_matrix = sapien.Pose(
            q=rotation).to_transformation_matrix()[:3, :3]
        forward, left, up = rotation_matrix.T
        center = np.array([0, 0, 0.1])
        self.camera.set_entity_pose(sapien.Pose(
            center - 0.5 * forward, rotation
        ))

        axis_origin = center + 0.09 * left - 0.09 * up
        for body, direction in zip(self._axes, np.eye(3) * 0.025):
            _set_segment(body, axis_origin, axis_origin + direction)
        bar_center = center - 0.075 * left - 0.11 * up
        _set_segment(
            self._scale_bar,
            bar_center - 0.025 * left,
            bar_center + 0.025 * left,
        )

    def set_points(self, human_points, human_ids):
        human_points = np.asarray(human_points, dtype=np.float32)
        human_ids = np.asarray(human_ids, dtype=np.int64)
        for index, (start, end) in enumerate(_HAND_CONNECTIONS):
            length = np.linalg.norm(human_points[end] - human_points[start])
            if self._bones[index] is None and length >= 1e-8:
                # ponytail: reuse first valid bone length; rebuild only if
                # per-frame landmark scale changes become visually significant.
                self._bones[index] = self._make_segment(
                    self._bone_material, length, 0.0015
                )
            if self._bones[index] is not None:
                _set_segment(
                    self._bones[index], human_points[start], human_points[end]
                )
        for marker, point in zip(self._joints, human_points):
            marker.entity.set_pose(sapien.Pose(point))
            marker.enable()
        if self._used is None:
            self._used = [
                _make_render_body(
                    self.scene,
                    sapien.render.RenderShapeSphere(
                        0.004, self._used_material),
                )
                for _ in human_ids
            ]
        if len(self._used) != len(human_ids):
            raise ValueError(
                "human_ids cannot change after inset initialization")
        for marker, point in zip(self._used, human_points[human_ids]):
            marker.entity.set_pose(sapien.Pose(point))
            marker.enable()
        self.visible = True

    def hide(self):
        self.visible = False

    def before_render(self):
        if self.visible:
            self.scene.update_render()
            self.camera.take_picture()

    def get_ui_windows(self):
        if not self.visible:
            return []
        if self.ui_window is None:
            from sapien import internal_renderer as R

            self.ui_picture = R.UIPicture()
            self.ui_window = (
                R.UIWindow()
                .Label("Mocap input")
                .Size(340, 380)
                .append(
                    R.UIDisplayText().Text(
                        "Matched view | fixed 280 mm | white = 50 mm"),
                    self.ui_picture,
                )
            )
        self.ui_window.Pos(10, 420)
        self.ui_picture.Size(320, 320).Picture(
            self.camera._internal_renderer, "Color")
        return [self.ui_window]


class HandKinematicModel:
    def __init__(self,
                 scene=None,
                 render=False,
                 hand=None,
                 hand_urdf='',
                 base_link='base_link',
                 joint_names=None,
                 # Ideally, these two guys (PD controller args) shouldn't be here.
                 # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
                 # If you see your hand model doing something weird (in the simulation viewer below), tune them.
                 kp=400.0,
                 kd=10):

        if joint_names is None:
            joint_names = []
        if scene is None:
            sapien.physx.set_default_material(1.0, 1.0, 0.0)
            sapien.physx.set_shape_config(contact_offset=0.02)
            sapien.physx.set_body_config(
                solver_position_iterations=25,
                solver_velocity_iterations=1,
            )
            scene_config = sapien.physx.get_scene_config()
            scene_config.enable_pcm = False
            sapien.physx.set_scene_config(scene_config)

            systems = [
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem(),
            ]
            if render:
                print("Enable Render Mode.")
            scene = sapien.Scene(systems)

        self.scene = scene

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose(
                [0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        base_link_entity = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(base_link_entity)

        # Setup hand dofs.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [user_idx_to_sim_idx.index(
            i) for i in range(len(user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        # this is in user specified "joint_name" order
        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])
        # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order(
            (self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, joint_names[i], joint,
                  self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=10)

    def get_n_dof(self):
        '''
            number of dof.
        '''
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        '''
            Setup keypoints to track.
        '''
        keypoint_links = [get_entity_by_name(
            self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(
            keypoint_links[i]), i) for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        '''
            Get keypoints from hand qpos. qpos is specified using the user order.
        '''
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {}
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(p=pose.p + (pose.to_transformation_matrix()[
                                   :3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1), q=pose.q)

            x = (base_pose.inv() * new_pose).p  # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    def is_self_collision(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.shape != (self.get_n_dof(),):
            raise ValueError(
                f"Expected qpos with shape ({self.get_n_dof()},), got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos must contain only finite values")

        qpos = self.convert_user_order_to_sim_order(qpos)
        self.hand.set_qpos(qpos)
        self.hand.set_qvel(np.zeros_like(qpos))
        self.scene.step()
        links = self.hand.get_links()
        return check_contact(self.scene, links, links)

    @staticmethod
    def build_from_config(config, **kwargs):
        '''
            Build a kinematic model from user config.
        '''
        render = kwargs.get("render", False)
        urdf_path = resolve_resource_path(config["urdf_path"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(hand_urdf=str(
            urdf_path), render=render, base_link=base_link, joint_names=joint_order)
        for link_name, group in config.get(
                "collision_group_overrides", {}).items():
            if (
                isinstance(group, bool)
                or not isinstance(group, int)
                or not 0 <= group <= 0xffffffff
            ):
                raise ValueError(
                    f"Invalid collision group for link '{link_name}': {group}")
            link = get_entity_by_name(model.hand.get_links(), link_name)
            if link is None:
                raise ValueError(
                    f"Collision group link not found: {link_name}")
            for shape in link.get_collision_shapes():
                shape.set_collision_groups([group, group, 0, 0])
        return model

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def set_qpos_target(self, qpos):
        '''
            This function is only used during visualization
        '''
        qpos = np.asarray(qpos)
        if qpos.shape != (self.get_n_dof(),):
            raise ValueError(
                f"Expected qpos with shape ({self.get_n_dof()},), got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos must contain only finite values")

        user_qpos = np.clip(
            qpos,
            self.joint_lower_limit + 1e-3,
            self.joint_upper_limit - 1e-3,
        )
        self.qpos_target = self.convert_user_order_to_sim_order(user_qpos)
        self.hand.set_qpos(self.qpos_target)
        self.hand.set_qvel(np.zeros_like(self.qpos_target))
        for joint, target in zip(self.all_joints, user_qpos):
            joint.set_drive_target(target)


class HandViewerEnv:
    def __init__(self, model):
        scene = model.scene
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

        control_window = ControlWindow()
        control_window.move_speed = 0.01
        control_window.rotate_speed = 0.001
        control_window.scroll_speed = 0.1
        mocap_inset = MocapInset()
        viewer = Viewer(plugins=[control_window, mocap_inset])
        viewer.set_scene(scene)
        scene.update_render()
        bounds = np.array([
            body.get_global_aabb_fast()
            for body in scene.render_system.render_bodies
        ])
        camera_pose, near, far = _fit_camera_to_bounds(bounds)
        viewer.set_camera_pose(camera_pose)
        viewer.window.set_camera_parameters(near=near, far=far, fovy=1)

        self.model = model
        self.scene = scene
        self.viewer = viewer
        self._mocap_inset = mocap_inset
        self._robot_cloud = None

    def set_mocap_overlay(self, human_points, human_ids, robot_points):
        human_points = np.asarray(human_points, dtype=np.float32)
        human_ids = np.asarray(human_ids, dtype=np.int64)
        robot_points = np.asarray(robot_points, dtype=np.float32)
        if human_points.shape != (21, 3) or not np.isfinite(human_points).all():
            raise ValueError("human_points must be finite with shape (21, 3)")
        if (
            human_ids.ndim != 1
            or len(human_ids) == 0
            or np.any(human_ids < 0)
            or np.any(human_ids >= len(human_points))
        ):
            raise ValueError("human_ids must contain valid landmark indices")
        if (
            robot_points.shape != (len(human_ids), 3)
            or not np.isfinite(robot_points).all()
        ):
            raise ValueError("robot_points must be finite and match human_ids")

        self._mocap_inset.set_points(human_points, human_ids)
        if self._robot_cloud is None:
            self._robot_cloud = sapien.render.RenderPointCloudComponent(
                len(robot_points)
            )
            self._robot_cloud.set_attribute(
                "color",
                np.tile([1, 0.2, 0.8, 1], (len(robot_points), 1)).astype(
                    np.float32
                ),
            )
            self._robot_cloud.set_attribute(
                "scale", np.full(len(robot_points), 0.004, dtype=np.float32)
            )
            sapien.Entity().add_component(
                self._robot_cloud).add_to_scene(self.scene)
        elif self._robot_cloud.get_vertices().shape != robot_points.shape:
            raise ValueError(
                "human_ids cannot change after overlay initialization")
        self._robot_cloud.set_vertices(robot_points)
        base_pose = self.model.hand.get_links()[
            self.model.base_link_idx].get_entity_pose()
        self._robot_cloud.entity.set_pose(base_pose)
        self._robot_cloud.enable()

    def hide_mocap_overlay(self):
        if self._robot_cloud is not None:
            self._robot_cloud.disable()
        self._mocap_inset.hide()

    def update(self):
        base_pose = self.model.hand.get_links()[
            self.model.base_link_idx].get_entity_pose()
        self._mocap_inset.set_view(
            self.viewer.window.get_camera_pose(), base_pose
        )
        self.scene.update_render()
        self.viewer.render()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro_right')

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()

    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    while True:
        viewer_env.update()

        steps += 1
        if steps % 30 == 0:
            targets = np.random.uniform(
                0, 1, n_dof) * (dof_upper - dof_lower - 1e-7) + dof_lower + 1e-7
            model.set_qpos_target(targets)
