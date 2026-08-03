import pytest

from geort.mocap.evaluation import parse_args


def test_evaluation_parser_accepts_replay_data():
    args = parse_args([
        "--mocap", "replay",
        "--hand", "allegro_right",
        "--ckpt-tag", "/tmp/checkpoint",
        "--data", "human",
    ])

    assert args.mocap == "replay"
    assert args.data == "human"


def test_evaluation_parser_requires_replay_data():
    with pytest.raises(SystemExit):
        parse_args([
            "--mocap", "replay",
            "--hand", "allegro_right",
            "--ckpt-tag", "/tmp/checkpoint",
        ])
