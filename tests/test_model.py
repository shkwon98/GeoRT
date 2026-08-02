import pytest
import torch

from geort.model import CollisionClassifier, FingerIK, IKModel


JOINT_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_finger_ik_maps_one_fingertip_to_its_joint_vector():
    model = FingerIK(num_joints=2, hidden_dim=8).eval()

    output = model(torch.randn(3, 3))

    assert output.shape == (3, 2)


def test_finger_ik_rejects_non_fingertip_shape():
    model = FingerIK(num_joints=2, hidden_dim=8)

    with pytest.raises(ValueError, match="Expected fingertip with shape"):
        model(torch.randn(3, 4))


def test_ik_model_rejects_wrong_number_of_fingertips():
    model = IKModel(JOINT_GROUPS, hidden_dim=8)

    with pytest.raises(ValueError, match="Expected keypoints with shape"):
        model(torch.randn(2, 3, 3))


def test_ik_model_preserves_input_dtype():
    model = IKModel(JOINT_GROUPS, hidden_dim=8).double().eval()

    output = model(torch.randn(2, 4, 3, dtype=torch.float64))

    assert output.shape == (2, 8)
    assert output.dtype == torch.float64


def test_ik_model_supports_explicit_total_joint_count():
    model = IKModel(JOINT_GROUPS, num_joints=10, hidden_dim=8).eval()

    output = model(torch.randn(2, 4, 3))

    assert output.shape == (2, 10)


def test_ik_model_keeps_legacy_checkpoint_key_layout():
    model = IKModel(JOINT_GROUPS, hidden_dim=8)

    assert "nets.0.0.weight" in model.state_dict()


def test_collision_classifier_returns_one_logit_per_joint_vector():
    model = CollisionClassifier(num_joints=8, hidden_dim=8).eval()

    logits = model(torch.rand(3, 8) * 2 - 1)

    assert logits.shape == (3,)


def test_collision_classifier_rejects_wrong_joint_shape():
    model = CollisionClassifier(num_joints=8, hidden_dim=8)

    with pytest.raises(ValueError, match="Expected normalized joints with shape"):
        model(torch.randn(3, 7))


def test_collision_classifier_rejects_unnormalized_joints():
    model = CollisionClassifier(num_joints=8, hidden_dim=8)

    with pytest.raises(ValueError, match=r"normalized to \[-1, 1\]"):
        model(torch.full((3, 8), 1.1))
