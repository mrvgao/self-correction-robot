import argparse
import os
import time
import datetime

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

def scan_datasets(folder, postfix=".h5"):
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths


def get_generator(algo_name, config_file, args, algo_name_short=None, pt=False):
    if args.wandb_proj_name is None:
        strings = [
            algo_name_short if (algo_name_short is not None) else algo_name,
            args.name,
            args.env,
            args.mod,
        ]
        args.wandb_proj_name = '_'.join([str(s) for s in strings if s is not None])

    if args.script is not None:
        generated_config_dir = os.path.join(os.path.dirname(args.script), "json")
    else:
        curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%y-%H-%M-%S')
        generated_config_dir=os.path.join(
            '~/', 'tmp/autogen_configs/ril', algo_name, args.env, args.mod, args.name, curr_time, "json",
        )

    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=generated_config_dir,
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    args.algo_name = algo_name
    args.pt = pt

    return generator


def set_env_settings(generator, args):
    if args.env in ["r2d2"]:
        assert args.mod == "im"
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot_state/cartesian_position", "robot_state/gripper_position"]
                ],
            )
        if "observation.modalities.obs.rgb" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["camera/image/hand_camera_image", "camera/image/varied_camera_left_image", "camera/image/varied_camera_right_image"]
                ],
            )
        if "observation.encoder.rgb.obs_randomizer_kwargs.crop_height"  not in generator.parameters:
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
        generator.add_param(
            key="train.data_format",
            name="",
            group=-1,
            values=[
                "r2d2"
            ],
        )
        generator.add_param(
            key="train.action_keys",
            name="",
            group=-1,
            values=[
                ["action/cartesian_velocity", "action/gripper_velocity"],
            ],
        )
        # generator.add_param(
        #     key="train.hdf5_cache_mode",
        #     name="",
        #     group=-1,
        #     values=[None],
        # )
    elif args.env == 'calvin':
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            if args.mod == 'ld':
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["scene_obs", "env_id", "robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "scene_proprio",
                    ],
                    hidename=True,
                )
            else:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "proprio",
                    ],
                    hidename=True,
                )

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[["rgb_static"]],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[["rgb_gripper"]],
            )

        if "experiment.rollout.horizon" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[1000],
            )

        # no batching yet for calvin
        generator.add_param(
            key="experiment.rollout.batched",
            name="",
            group=-1,
            values=[False],
        )

    elif args.env == 'franka_kitchen':
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["flat"]
            ],
        )
        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot_joints"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image", "eye_in_hand_image"]
                ],
            )

        if "experiment.rollout.horizon" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[280],
            )
    elif args.env in ["kitchen", "kitchen_mm"]:
        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_base_pos",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["robot0_agentview_left_image",
                     "robot0_agentview_right_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot0_base_pos",
                     "object",
                    ]
                ],
            )
    elif args.env in ['square', 'lift', 'place_close']:
        # # set videos off
        # args.no_video = True

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "object"]
                ],
            )
    elif args.env == 'transport':
        # set videos off
        args.no_video = True

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot1_eef_pos",
                     "robot1_eef_quat",
                     "robot1_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["shouldercamera0_image",
                     "robot0_eye_in_hand_image",
                     "shouldercamera1_image",
                     "robot1_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot1_eef_pos",
                     "robot1_eef_quat",
                     "robot1_gripper_qpos",
                     "object"]
                ],
            )

        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )
    elif args.env == 'tool_hang':
        # set videos off
        args.no_video = True

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["sideview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "object"]
                ],
            )

        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )
    elif args.env in ['real_breakfast', 'real_lift', 'real_cook']:
        assert args.mod == "im"
        generator.add_param(
            key="experiment.save.enabled",
            name="",
            group=-1,
            values=[
                True
            ],
        )
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["ee_pos", "ee_quat", "gripper_states"]
            ],
        )
        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="",
            group=-1,
            values=[
                ["agentview_rgb", "eye_in_hand_rgb"]
            ],
        )
    else:
        raise ValueError


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
    )

    parser.add_argument(
        "--env",
        type=str,
        default='r2d2',
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        default='im',
    )

    parser.add_argument(
        "--ckpt_mode",
        type=str,
        choices=["off", "all", "best_only"],
        default=None,
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        '--no_video',
        action='store_true'
    )

    parser.add_argument(
        "--tmplog",
        action="store_true",
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
    )

    parser.add_argument(
        "--no_rollout",
        action="store_true",
    )

    parser.add_argument(
        "--r3m",
        action="store_true",
    )

    parser.add_argument(
        "--pl",
        action="store_true",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None
    )

    parser.add_argument(
        "--num_cmd_groups",
        type=int,
        default=None
    )

    return parser


def make_generator(args, make_generator_helper):
    if args.tmplog or args.debug and args.name is None:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + str(args.name)

    if args.debug or args.tmplog:
        args.no_wandb = True

    if args.wandb_proj_name is not None:
        # prepend data to wandb name
        # time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        # args.wandb_proj_name = time_str + args.wandb_proj_name
        pass

    if (args.debug or args.tmplog) and (args.wandb_proj_name is None):
        args.wandb_proj_name = 'debug'

    if not args.debug:
        assert args.name is not None

    # make config generator
    generator = make_generator_helper(args)

    if args.ckpt_mode is None:
        if args.pt:
            args.ckpt_mode = "all"
        else:
            args.ckpt_mode = "best_only"

    set_env_settings(generator, args)
    # set_mod_settings(generator, args)

    # set the debug settings last, to override previous setting changes
    # set_debug_mode(generator, args)

    """ misc settings """
    generator.add_param(
        key="experiment.validate",
        name="",
        group=-1,
        values=[
            False,
        ],
    )

    # generate jsons and script
    generator.generate()