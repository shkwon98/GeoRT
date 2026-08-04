import pytest
import torch

import geort.model as model_module
from geort.model import CollisionClassifier, FingerIK, FKModel, IKModel

JOINT_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_ik_hierarchy_maps_fingertips_and_keeps_checkpoint_keys():
    finger = FingerIK(num_joints=2, hidden_dim=8).eval()
    model = IKModel(JOINT_GROUPS, hidden_dim=8).double().eval()

    assert finger(torch.randn(3, 3)).shape == (3, 2)
    output = model(torch.randn(2, 4, 3, dtype=torch.float64))
    assert output.shape == (2, 8)
    assert output.dtype == torch.float64
    assert "nets.0.0.weight" in model.state_dict()


def test_fk_hierarchy_uses_finger_modules_and_keeps_checkpoint_keys():
    finger = model_module.FingerFK(num_joints=2, hidden_dim=8).double().eval()
    model = FKModel(JOINT_GROUPS, hidden_dim=8).double().eval()

    assert finger(torch.randn(3, 2, dtype=torch.float64)).shape == (3, 3)
    assert all(isinstance(net, model_module.FingerFK) for net in model.nets)
    assert model(torch.randn(2, 8, dtype=torch.float64)).shape == (2, 4, 3)
    assert "nets.0.0.weight" in model.state_dict()
    assert not hasattr(model, "n_total_joint")


def test_collision_classifier_returns_one_logit_per_joint_vector():
    model = CollisionClassifier(num_joints=8, hidden_dim=8).eval()

    assert model(torch.rand(3, 8) * 2 - 1).shape == (3,)


@pytest.mark.parametrize("groups", [[], [[]], [[0, 1], [1, 2]], [[0], [2]]])
def test_joint_groups_reject_empty_duplicate_or_gapped_indices(groups):
    with pytest.raises(ValueError, match="joint groups"):
        IKModel(groups)
    with pytest.raises(ValueError, match="joint groups"):
        FKModel(groups)


def test_ik_supports_five_complete_joint_groups():
    groups = [list(range(start, start + 4)) for start in range(0, 20, 4)]
    model = IKModel(groups, hidden_dim=8).eval()

    assert model(torch.randn(2, 5, 3)).shape == (2, 20)
