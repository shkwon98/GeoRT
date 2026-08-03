import sys

import pytest
import torch


def test_runtime_uses_python_312():
    assert sys.version_info[:2] == (3, 12)


def test_cuda_build_supports_the_detected_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    major, minor = torch.cuda.get_device_capability()
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    assert torch.ones(1, device="cuda").item() == 1.0


def test_installed_training_stack_supports_python_312():
    sapien = pytest.importorskip("sapien")
    open3d = pytest.importorskip("open3d")

    assert int(sapien.__version__.split(".", 1)[0]) >= 3
    assert tuple(map(int, open3d.__version__.split(".")[:2])) >= (0, 19)


def test_installed_training_stack_loads_the_bundled_hand():
    pytest.importorskip("sapien")
    from geort.env.hand import HandKinematicModel
    from geort.utils.config_utils import get_config

    hand = HandKinematicModel.build_from_config(get_config("allegro_right"))

    assert hand.get_n_dof() == 16
