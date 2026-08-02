import numpy as np


def test_collision_dataset_loads_qpos_and_labels(tmp_path):
    from geort.dataset import CollisionDataset

    path = tmp_path / "collision.npz"
    qpos = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    collision = np.array([False, True])
    np.savez(path, qpos=qpos, collision=collision)

    dataset = CollisionDataset(path)

    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset[1]["qpos"], qpos[1])
    assert dataset[1]["collision"] == np.float32(1.0)
