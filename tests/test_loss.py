import torch

from geort.loss import collision_free_loss


def test_collision_free_loss_is_lower_for_safer_logits():
    safer_logits = torch.tensor([-4.0, -2.0])
    riskier_logits = torch.tensor([2.0, 4.0])

    assert collision_free_loss(safer_logits) < collision_free_loss(riskier_logits)
