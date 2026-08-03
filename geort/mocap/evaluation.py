# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mocap", choices=("manus", "mediapipe", "replay"), required=True)
    parser.add_argument("--hand", default="allegro_right")
    parser.add_argument("--ckpt-tag", default="alex")
    parser.add_argument("--data")
    args = parser.parse_args(argv)
    if args.mocap == "replay" and not args.data:
        parser.error("--data is required when --mocap replay")
    return args


def _create_mocap(args):
    if args.mocap == "manus":
        from geort.mocap.manus_mocap import ManusMocap
        return ManusMocap()
    if args.mocap == "mediapipe":
        from geort.mocap.mediapipe_mocap import MediaPipeMocap
        return MediaPipeMocap()

    from geort.mocap.replay_mocap import ReplayMocap
    return ReplayMocap(args.data)


def main(argv=None):
    args = parse_args(argv)

    from geort import get_config, load_model
    from geort.env.hand import HandKinematicModel

    model = load_model(args.ckpt_tag)
    mocap = _create_mocap(args)
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer = hand.get_viewer_env()
    viewer_updates = 10 if args.mocap == "replay" else 1

    try:
        while True:
            for _ in range(viewer_updates):
                viewer.update()

            result = mocap.get()
            if result["status"] == "recording" and result["result"] is not None:
                hand.set_qpos_target(model.forward(result["result"]))
            if result["status"] == "quit":
                break
    finally:
        close = getattr(mocap, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
