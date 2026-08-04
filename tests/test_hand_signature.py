import importlib
import inspect
import sys
from types import SimpleNamespace


def test_hand_does_not_use_a_shared_joint_names_default(monkeypatch):
    original_module = sys.modules.pop("geort.env.hand", None)
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sapien.utils", SimpleNamespace(Viewer=object))
    try:
        module = importlib.import_module("geort.env.hand")
        parameter = inspect.signature(module.HandKinematicModel).parameters[
            "joint_names"
        ]

        assert parameter.default is None
    finally:
        sys.modules.pop("geort.env.hand", None)
        if original_module is not None:
            sys.modules["geort.env.hand"] = original_module
