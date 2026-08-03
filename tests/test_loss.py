import torch

from geort.loss import chamfer_distance, collision_free_loss, pinch_correspondence_loss


def _repeat_chamfer(left, right):
    squared = ((left[:, :, None] - right[:, None, :]) ** 2).sum(-1)
    return (squared.min(2).values.mean(1) + squared.min(1).values.mean(1)).mean()


def test_chamfer_matches_repeat_reference_values_and_gradients():
    torch.manual_seed(3)
    actual_left = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    actual_right = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    expected_left = actual_left.detach().clone().requires_grad_()
    expected_right = actual_right.detach().clone().requires_grad_()

    actual = chamfer_distance(actual_left, actual_right)
    expected = _repeat_chamfer(expected_left, expected_right)
    actual_grad = torch.autograd.grad(actual, (actual_left, actual_right))
    expected_grad = torch.autograd.grad(expected, (expected_left, expected_right))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad[0], expected_grad[0])
    torch.testing.assert_close(actual_grad[1], expected_grad[1])


def test_chamfer_does_not_repeat_point_tensors(monkeypatch):
    def reject_repeat(*args, **kwargs):
        raise RuntimeError("point tensor repeat")

    monkeypatch.setattr(torch.Tensor, "repeat", reject_repeat)

    assert torch.isfinite(
        chamfer_distance(torch.randn(1, 2, 3), torch.randn(1, 3, 3))
    )


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
