from pathlib import Path

from experiments.robots import load_robot_spec


def test_allegro_and_wuji_specs_have_complete_named_joint_groups():
    wuji = load_robot_spec("wuji_right", env={})
    allegro = load_robot_spec("allegro_right")

    assert Path(wuji["urdf_path"]).is_file()
    assert len(wuji["urdf_sha256"]) == 64
    assert len(wuji["joint_order"]) == 20
    assert len(
        {name for tip in wuji["fingertip_link"] for name in tip["joint"]}
    ) == 20
    assert wuji["ros"]["input_topic"] == "/teleop/human/hand_right/pose"
    assert wuji["ros"]["output_topic"] == (
        "/control/hand_right/hand_right_controller/joint_trajectory"
    )
    assert wuji["geort_config"] == "wuji_right"
    assert allegro["geort_config"] == "allegro_right"


def test_bundled_wuji_left_does_not_require_external_description():
    wuji = load_robot_spec("wuji_left", env={})

    assert Path(wuji["urdf_path"]).is_file()
    assert wuji["geort_config"] == "wuji_left"
