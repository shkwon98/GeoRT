from pathlib import Path

from geort.robots import load_robot, robot_fingerprint


def test_bundled_robots_are_complete_and_self_contained():
    for name, joint_count in (("allegro_right", 16), ("wuji_right", 20)):
        robot = load_robot(name)

        assert robot["name"] == name
        assert robot["hand_side"] == "right"
        assert Path(robot["urdf_path"]).is_file()
        assert len(robot["robot_fingerprint"]) == 64
        assert len(robot["joint_order"]) == joint_count


def test_robot_fingerprint_covers_config_urdf_and_mesh(tmp_path):
    mesh = tmp_path / "finger.stl"
    mesh.write_bytes(b"mesh-a")
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        '<robot name="hand"><link name="palm"><visual><geometry>'
        '<mesh filename="finger.stl"/>'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    config = {"name": "hand", "collision_ignore_pairs": []}

    original = robot_fingerprint(config, urdf)
    config["collision_ignore_pairs"] = [["palm", "finger"]]
    config_changed = robot_fingerprint(config, urdf)
    mesh.write_bytes(b"mesh-b")
    mesh_changed = robot_fingerprint(config, urdf)

    assert len({original, config_changed, mesh_changed}) == 3


def test_robot_fingerprint_depends_on_contents_not_local_urdf_path(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text('<robot name="hand"/>', encoding="utf-8")

    first = robot_fingerprint(
        {"name": "hand", "urdf_path": "/machine-a/hand.urdf"}, urdf
    )
    second = robot_fingerprint(
        {"name": "hand", "urdf_path": "/machine-b/hand.urdf"}, urdf
    )

    assert first == second
