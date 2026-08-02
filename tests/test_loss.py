import torch

from geort.loss import collision_free_loss, pinch_correspondence_loss


def test_paper_auxiliary_losses_preserve_safety_and_pinch_signals():
    safer_logits = torch.tensor([-4.0, -2.0])
    riskier_logits = torch.tensor([2.0, 4.0])
    human = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]]
    )
    robot = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]]
    )

    assert collision_free_loss(
        safer_logits) < collision_free_loss(riskier_logits)
    assert torch.isclose(pinch_correspondence_loss(
        human, robot), torch.tensor(2.0))
