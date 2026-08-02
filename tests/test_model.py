import torch

from geort.model import CollisionClassifier, FingerIK, IKModel, get_finger_ik


JOINT_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_ik_hierarchy_maps_fingertips_and_keeps_checkpoint_keys():
    finger = FingerIK(num_joints=2, hidden_dim=8).eval()
    model = IKModel(JOINT_GROUPS, hidden_dim=8).double().eval()

    assert finger(torch.randn(3, 3)).shape == (3, 2)
    output = model(torch.randn(2, 4, 3, dtype=torch.float64))
    assert output.shape == (2, 8)
    assert output.dtype == torch.float64
    assert "nets.0.0.weight" in model.state_dict()
    assert "0.weight" in get_finger_ik(n_joint=2, hidden=8).state_dict()


def test_collision_classifier_returns_one_logit_per_joint_vector():
    model = CollisionClassifier(num_joints=8, hidden_dim=8).eval()

    assert model(torch.rand(3, 8) * 2 - 1).shape == (3,)
