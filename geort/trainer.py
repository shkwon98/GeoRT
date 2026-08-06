# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from geort.dataset import (
    CollisionDataset,
    GestureDataset,
    MultiPointDataset,
    RobotKinematicsDataset,
)
from geort.formatter import HandFormatter
from geort.loss import chamfer_distance, collision_free_loss, pinch_correspondence_loss
from geort.model import CollisionClassifier, FKModel, IKModel
from geort.utils.config_utils import get_config, load_json, save_json
from geort.utils.path import (
    get_bundled_fk_checkpoint,
    get_checkpoint_root,
    get_human_data,
    get_robot_cache_root,
)

PAPER_DEFAULTS = {
    "w_chamfer": 80.0,
    "w_curvature": 1.0,
    "w_pinch": 1000.0,
    "w_collision": 1e-4,
}

RESUME_HAND_KEYS = ("name", "urdf_path", "base_link",
                    "joint_order", "fingertip_link", "joint")


def validate_resume_config(current_config, saved_config, expected_training):
    for key in RESUME_HAND_KEYS:
        if saved_config.get(key) != current_config.get(key):
            raise ValueError(f"Resume configuration mismatch for '{key}'")
    saved_training = saved_config.get("training", {})
    for key, value in expected_training.items():
        if saved_training.get(key) != value:
            raise ValueError(f"Resume configuration mismatch for '{key}'")


def _freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def split_aligned_frames(frames, val_fraction, seed):
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError("Expected aligned frames with shape [T, F, 3]")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    if not val_fraction:
        return frames, frames[:0]
    if len(frames) < 2:
        raise ValueError("Validation requires at least two frames")

    validation_count = min(
        len(frames) - 1, max(1, round(len(frames) * val_fraction)))
    validation_indices = np.sort(np.random.default_rng(
        seed).permutation(len(frames))[:validation_count])
    train_mask = np.ones(len(frames), dtype=bool)
    train_mask[validation_indices] = False
    return frames[train_mask], frames[validation_indices]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["state"].cpu().numpy(),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


@contextmanager
def fixed_rng(seed):
    state = capture_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def build_checkpoint_callbacks(save_dir, save_every, has_validation=True):
    if save_every < 0:
        raise ValueError("save_every must be non-negative")

    callbacks = [ModelCheckpoint(
        dirpath=save_dir,
        filename="best",
        monitor="validation/total" if has_validation else None,
        mode="min",
        save_last=True,
        save_top_k=1 if has_validation else 0,
        save_on_train_epoch_end=not has_validation,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )]
    if save_every:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                filename="epoch={epoch:04d}",
                every_n_epochs=save_every,
                save_on_train_epoch_end=True,
                save_top_k=-1,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
    return callbacks


def merge_dict_list(dl):
    keys = dl[0].keys()

    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])

    result = {k: np.array(v) for k, v in result.items()}
    return result


def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list


def generate_current_timestring():
    """
        Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class GeoRTLightningModule(LightningModule):
    def __init__(
        self,
        ik_model,
        fk_model,
        collision_classifier,
        robot_points,
        weights,
        direction_sigma,
        flatness_sigma,
        validation_seed,
    ):
        super().__init__()
        self.ik_model = ik_model
        self.fk_model = _freeze_model(fk_model)
        self.collision_classifier = _freeze_model(collision_classifier)
        self.register_buffer("robot_points", robot_points, persistent=False)
        self.weights = weights
        self.direction_sigma = direction_sigma
        self.flatness_sigma = flatness_sigma
        self.validation_seed = validation_seed
        self._validation_rng_state = None

    def train(self, mode=True):
        super().train(mode)
        self.fk_model.eval()
        self.collision_classifier.eval()
        return self

    def _compute_losses(self, coverage_points, gesture_frames):
        coverage_embedded = self.fk_model(self.ik_model(coverage_points))
        gesture_joints = self.ik_model(gesture_frames)
        gesture_embedded = self.fk_model(gesture_joints)

        flatness_delta = torch.randn_like(gesture_frames) * self.flatness_sigma
        embedded_positive = self.fk_model(
            self.ik_model(gesture_frames + flatness_delta))
        embedded_negative = self.fk_model(
            self.ik_model(gesture_frames - flatness_delta))
        curvature = ((embedded_positive + embedded_negative -
                      2 * gesture_embedded) ** 2).mean()

        selected = torch.randint(
            self.robot_points.shape[1],
            (coverage_points.size(0),),
            device=self.device,
        )
        target = self.robot_points[:, selected].permute(1, 0, 2)
        chamfer = sum(
            chamfer_distance(
                coverage_embedded[:, i].unsqueeze(0),
                target[:, i].unsqueeze(0),
            )
            for i in range(coverage_points.size(1))
        )

        direction_delta = torch.randn_like(
            gesture_frames) * self.direction_sigma
        embedded_delta = self.fk_model(
            self.ik_model(gesture_frames + direction_delta))
        d1 = direction_delta.reshape(-1, 3)
        d2 = (embedded_delta - gesture_embedded).reshape(-1, 3)
        direction = -(F.normalize(d1, dim=-1, eps=1e-5) *
                      F.normalize(d2, dim=-1, eps=1e-5)).sum(-1).mean()
        pinch = pinch_correspondence_loss(gesture_frames, gesture_embedded)
        collision = collision_free_loss(
            self.collision_classifier(gesture_joints))
        total = (
            direction
            + self.weights["w_chamfer"] * chamfer
            + self.weights["w_curvature"] * curvature
            + self.weights["w_pinch"] * pinch
            + self.weights["w_collision"] * collision
        )
        return {
            "total": total,
            "direction": direction,
            "chamfer": chamfer,
            "curvature": curvature,
            "pinch": pinch,
            "collision": collision,
        }

    def _shared_step(self, batch, stage):
        losses = self._compute_losses(batch["coverage"], batch["gesture"])
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "total",
                batch_size=1,
                sync_dist=True,
            )
        return losses["total"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "validation")

    def configure_optimizers(self):
        return optim.AdamW(self.ik_model.parameters(), lr=1e-4)

    def on_validation_epoch_start(self):
        self._validation_rng_state = capture_rng_state()
        set_seed(self.validation_seed)

    def on_validation_epoch_end(self):
        restore_rng_state(self._validation_rng_state)
        self._validation_rng_state = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint["rng_state"] = capture_rng_state()

    def on_load_checkpoint(self, checkpoint):
        restore_rng_state(checkpoint["rng_state"])


class GeoRTTrainer:
    def __init__(self, config, device=None, data_dir=None, checkpoint_dir=None):
        from geort.env.hand import HandKinematicModel

        self.config = config
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.data_dir = Path(
            data_dir
        ) if data_dir is not None else get_robot_cache_root(self.config)
        self.checkpoint_dir = Path(
            checkpoint_dir) if checkpoint_dir is not None else Path(get_checkpoint_root())
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_robot_pointcloud(self, keypoint_names):
        '''
            Utility getter function. Return the robot fingertip point cloud.
        '''
        kinematics_dataset = self.get_robot_kinematics_dataset()
        return kinematics_dataset.export_robot_pointcloud(keypoint_names)

    def get_robot_kinematics_dataset(self):
        '''
            Utility getter function. Return the robot kinematics dataset
        '''
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not dataset_path.exists():
            self.generate_robot_kinematics_dataset(n_total=100000, save=True)

        keypoint_names = self.get_keypoint_info()["link"]

        kinematics_dataset = RobotKinematicsDataset(
            dataset_path, keypoint_names=keypoint_names)
        return kinematics_dataset

    def get_robot_kinematics_dataset_path(self, postfix=False):
        '''
            Utility getter function. Return the path to the robot kinematics dataset.
        '''
        data_name = self.config["name"]

        return self.data_dir / (f"{data_name}.npz" if postfix else data_name)

    def get_keypoint_info(self):
        keypoint_links = []
        keypoint_offsets = []
        keypoint_joints = []
        keypoint_human_ids = []

        joint_order = self.config["joint_order"]

        for info in self.config["fingertip_link"]:
            keypoint_links.append(info["link"])
            keypoint_offsets.append(info['center_offset'])
            keypoint_human_ids.append(info['human_hand_id'])

            keypoint_joint = []
            for joint in info["joint"]:
                keypoint_joint.append(joint_order.index(joint))

            keypoint_joints.append(keypoint_joint)

        out = {
            "link": keypoint_links,
            "offset": keypoint_offsets,
            "joint": keypoint_joints,
            "human_id": keypoint_human_ids,
        }

        return out

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        '''
            This function will generate a (joint position, keypoint position) dataset. 
            - The joint order is specified by "joint_order" in configuration.
            - The keypoint order is specified by "fingertip_link" field in configuration.
        '''
        info = self.get_keypoint_info()

        self.hand.initialize_keypoint(
            keypoint_link_names=info["link"], keypoint_offsets=info["offset"])

        # joint order is based on user config specification.
        joint_range_low, joint_range_high = self.hand.get_joint_limit()
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []

        for _ in tqdm(range(n_total)):
            qpos = np.random.uniform(0, 1, len(
                joint_range_low)) * (joint_range_high - joint_range_low) + joint_range_low
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)

        all_data_keypoint = merge_dict_list(all_data_keypoint)

        dataset = {"qpos": np.asarray(
            all_data_qpos, dtype=np.float32), "keypoint": all_data_keypoint}

        if save:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / f"fk_model_{name}.pth"

    def get_robot_neural_fk_model(self, force_train=False):
        '''
            This function will return a forward kinematics model.
            If the fk model does not exist, this function will train one first.
        '''

        # Normalizer.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        # Model.
        print(self.get_keypoint_info()["joint"])
        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()[
                           "joint"]).to(self.device)

        # If the model exists, load it.
        fk_checkpoint_path = self.get_fk_checkpoint_path()
        load_path = fk_checkpoint_path if fk_checkpoint_path.exists() else None
        if load_path is None and not force_train:
            try:
                load_path = get_bundled_fk_checkpoint(self.config["name"])
            except FileNotFoundError:
                pass
        if load_path is not None and not force_train:
            fk_model.load_state_dict(torch.load(
                load_path, map_location=self.device, weights_only=True))
        else:
            # If the model does not exist, train it.
            print("Train Neural Forward Kinematics (FK) from Scratch")

            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(
                fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)

            criterion_fk = nn.MSELoss()
            for epoch in range(50):
                all_fk_error = 0
                for batch_idx, batch in enumerate(fk_dataloader):
                    keypoint = batch["keypoint"].to(self.device).float()
                    qpos = batch["qpos"].to(self.device).float()
                    qpos = qpos_normalizer.normalize_torch(qpos)
                    predicted_keypoint = fk_model(qpos)
                    fk_optim.zero_grad()
                    loss = criterion_fk(predicted_keypoint, keypoint)
                    loss.backward()
                    fk_optim.step()

                    all_fk_error += loss.item()

                avg_fk_error = all_fk_error / (batch_idx + 1)
                print(
                    f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}")

            torch.save(fk_model.state_dict(), fk_checkpoint_path)

        return _freeze_model(fk_model)

    def get_collision_dataset_path(self):
        return self.data_dir / f"{self.config['name']}_collision.npz"

    def generate_collision_dataset(self, save=True):
        qpos = self.get_robot_kinematics_dataset().qpos.astype(np.float32)
        collision = np.asarray([self.hand.is_self_collision(q)
                               for q in tqdm(qpos)], dtype=np.float32)
        dataset = {"qpos": qpos, "collision": collision}
        if save:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.get_collision_dataset_path(), **dataset)
        return dataset

    def get_collision_dataset(self):
        path = self.get_collision_dataset_path()
        if not path.exists():
            self.generate_collision_dataset(save=True)
        return CollisionDataset(path)

    def get_collision_checkpoint_path(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / f"collision_model_{self.config['name']}.pth"

    def get_collision_classifier(self, force_train=False):
        lower, upper = self.hand.get_joint_limit()
        normalizer = HandFormatter(lower, upper)
        model = CollisionClassifier(len(lower)).to(self.device)
        path = self.get_collision_checkpoint_path()

        if path.exists() and not force_train:
            model.load_state_dict(torch.load(
                path, map_location=self.device, weights_only=True))
        else:
            dataset = self.get_collision_dataset()
            if len(dataset) < 2:
                raise ValueError(
                    "Collision classifier training requires at least two samples")
            batch_size = min(256, len(dataset))
            while len(dataset) % batch_size == 1 and batch_size > 2:
                batch_size -= 1
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            criterion = nn.BCEWithLogitsLoss()
            model.train()
            for _ in range(50):
                for batch in loader:
                    qpos = normalizer.normalize_torch(
                        batch["qpos"].to(self.device).float())
                    labels = batch["collision"].to(self.device).float()
                    loss = criterion(model(qpos), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            torch.save(model.state_dict(), path)

        return _freeze_model(model)

    @staticmethod
    def _build_streams(
        frames,
        shuffle,
        coverage_samples=20000,
        coverage_batch_size=2048,
        gesture_batch_size=2048,
    ):
        if min(coverage_samples, coverage_batch_size, gesture_batch_size) <= 0:
            raise ValueError(
                "training sample and batch sizes must be positive")
        coverage = MultiPointDataset.from_points(
            frames.transpose(1, 0, 2), n=coverage_samples)
        gestures = GestureDataset(frames)
        gesture_batch_size = min(gesture_batch_size, len(gestures))
        while len(gestures) % gesture_batch_size == 1 and gesture_batch_size > 2:
            gesture_batch_size -= 1
        return (
            DataLoader(coverage, batch_size=coverage_batch_size,
                       shuffle=shuffle),
            DataLoader(gestures, batch_size=gesture_batch_size,
                       shuffle=shuffle),
        )

    def train(
        self,
        human_data_path,
        tag="",
        epoch=50,
        save_every=50,
        seed=0,
        val_fraction=0.1,
        resume=None,
        direction_sigma=0.005,
        flatness_sigma=0.002,
        coverage_samples=20000,
        coverage_batch_size=2048,
        gesture_batch_size=2048,
        **loss_weights,
    ):
        weights = {name: loss_weights.get(name, default)
                   for name, default in PAPER_DEFAULTS.items()}
        human_data_path = Path(human_data_path).resolve()
        lower, upper = self.hand.get_joint_limit()
        export_config = self.config.copy()
        export_config["joint"] = {"lower": get_float_list_from_np(
            lower), "upper": get_float_list_from_np(upper)}
        training_metadata = {
            "human_data": str(human_data_path),
            "seed": seed,
            "val_fraction": val_fraction,
            **weights,
            "direction_sigma": direction_sigma,
            "flatness_sigma": flatness_sigma,
            "coverage_samples": coverage_samples,
            "coverage_batch_size": coverage_batch_size,
            "gesture_batch_size": gesture_batch_size,
            "save_every": save_every,
        }
        export_config["training"] = training_metadata

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            resume_path = Path(resume)
            if resume_path.is_file():
                save_dir = resume_path.parent
                resume_checkpoint = resume_path
            else:
                save_dir = resume_path if resume_path.is_dir() else self.checkpoint_dir / \
                    resume_path
                resume_checkpoint = save_dir / "last.ckpt"
            if not save_dir.is_dir():
                raise FileNotFoundError(
                    f"Resume experiment directory not found: {save_dir}")
            config_path = save_dir / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Resume configuration not found: {config_path}")
            validate_resume_config(export_config, load_json(
                config_path), training_metadata)
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Lightning checkpoint not found: {resume_checkpoint}")
        else:
            resume_checkpoint = None
            suffix = f"_{tag}" if tag else ""
            save_dir = self.checkpoint_dir / \
                f"{self.config['name']}_{generate_current_timestring()}{suffix}"
            save_dir.mkdir(parents=True)
            save_json(export_config, save_dir / "config.json")

        set_seed(seed)
        human_raw = np.load(human_data_path)
        human_ids = self.get_keypoint_info()["human_id"]
        if human_raw.ndim != 3 or human_raw.shape[-1] < 3 or max(human_ids) >= human_raw.shape[1]:
            raise ValueError(
                "Human recordings must have shape [T, landmarks, >=3] and contain configured human ids")
        human_frames = human_raw[:, human_ids, :3].astype(np.float32)
        train_frames, validation_frames = split_aligned_frames(
            human_frames, val_fraction, seed)
        if len(train_frames) < 2:
            raise ValueError("Training requires at least two aligned frames")
        train_coverage, train_gestures = self._build_streams(
            train_frames,
            shuffle=True,
            coverage_samples=coverage_samples,
            coverage_batch_size=coverage_batch_size,
            gesture_batch_size=gesture_batch_size,
        )
        validation_streams = self._build_streams(
            validation_frames,
            shuffle=False,
            coverage_samples=coverage_samples,
            coverage_batch_size=coverage_batch_size,
            gesture_batch_size=gesture_batch_size,
        ) if len(validation_frames) else None

        fk_model = self.get_robot_neural_fk_model()
        collision_classifier = self.get_collision_classifier()
        set_seed(seed)
        ik_model = IKModel(keypoint_joints=self.get_keypoint_info()[
                           "joint"]).to(self.device)
        robot_points = torch.as_tensor(
            self.get_robot_pointcloud(self.get_keypoint_info()["link"]), dtype=torch.float32, device=self.device
        )
        module = GeoRTLightningModule(
            ik_model,
            fk_model,
            collision_classifier,
            robot_points,
            weights,
            direction_sigma,
            flatness_sigma,
            seed,
        )
        train_loader = CombinedLoader(
            {"coverage": train_coverage, "gesture": train_gestures},
            mode="max_size_cycle",
        )
        validation_loader = CombinedLoader(
            {"coverage": validation_streams[0],
                "gesture": validation_streams[1]},
            mode="max_size_cycle",
        ) if validation_streams is not None else None
        accelerator = "gpu" if self.device.type == "cuda" else self.device.type
        devices = [
            self.device.index] if self.device.type == "cuda" and self.device.index is not None else 1
        lightning_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=epoch,
            callbacks=build_checkpoint_callbacks(
                save_dir, save_every, validation_streams is not None
            ),
            default_root_dir=save_dir,
            deterministic=True,
            enable_model_summary=False,
            logger=CSVLogger(save_dir=save_dir, name="logs"),
            log_every_n_steps=1,
            num_sanity_val_steps=0,
        )
        lightning_trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
            ckpt_path=resume_checkpoint,
        )

        return save_dir


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-hand", "--hand", default="allegro_right")
    parser.add_argument("-human_data", "--human-data", default="human")
    parser.add_argument("-ckpt_tag", "--tag", default="")
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume")
    parser.add_argument("--direction-sigma", type=float, default=0.005)
    parser.add_argument("--flatness-sigma", type=float, default=0.002)
    parser.add_argument("--coverage-samples", type=int, default=20000)
    parser.add_argument("--coverage-batch-size", type=int, default=2048)
    parser.add_argument("--gesture-batch-size", type=int, default=2048)
    for name, default in PAPER_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}",
                            f"--{name}", dest=name, type=float, default=default)
    return parser


def _resolve_human_data(name_or_path, data_dir=None):
    path = Path(name_or_path)
    if path.is_file():
        return path
    if data_dir is not None:
        path = Path(data_dir) / path
        if not path.suffix:
            path = path.with_suffix(".npy")
        if path.is_file():
            return path
        raise FileNotFoundError(f"Human data file not found: {path}")
    return get_human_data(name_or_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    config = get_config(args.hand)
    trainer = GeoRTTrainer(
        config,
        device=args.device,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    human_data_path = _resolve_human_data(args.human_data, args.data_dir)
    print("Training with human data:", human_data_path.as_posix())
    trainer.train(
        human_data_path,
        tag=args.tag,
        epoch=args.epoch,
        save_every=args.save_every,
        seed=args.seed,
        val_fraction=args.val_fraction,
        resume=args.resume,
        direction_sigma=args.direction_sigma,
        flatness_sigma=args.flatness_sigma,
        coverage_samples=args.coverage_samples,
        coverage_batch_size=args.coverage_batch_size,
        gesture_batch_size=args.gesture_batch_size,
        w_chamfer=args.w_chamfer,
        w_curvature=args.w_curvature,
        w_collision=args.w_collision,
        w_pinch=args.w_pinch,
    )
