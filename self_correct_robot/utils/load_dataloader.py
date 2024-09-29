# first set seeds
import json
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader

import self_correct_robot.utils.train_utils as TrainUtils
import self_correct_robot.utils.obs_utils as ObsUtils
import self_correct_robot.utils.env_utils as EnvUtils
import self_correct_robot.utils.file_utils as FileUtils
import self_correct_robot.utils.lang_utils as LangUtils
from self_correct_robot.utils.log_utils import PrintLogger, DataLogger, flush_warnings


def load_dataloader(config, device):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    # if config.progress_model_path:
    #     progress_model = ProgressResNetModel()
    #     progress_model.load_state_dict(torch.load(config.progress_model_path))
    #     progress_model.to(device)
    # else:
    #     progress_model = None

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============: dataset_path: ", dataset_path + "\n")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from self_correct_robot.utils.script_utils import deep_update

        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        do_eval = dataset_cfg.get("do_eval", True)
        if do_eval is not True:
            continue
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        # horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        horizon = max(dataset_cfg.get('horizon', 0), config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)


    # create environments
    def env_iterator():
        for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):
            def create_env_helper(env_i=0):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                    seed=config.train.seed * 1000 + env_i,
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

                return env

            if config.experiment.rollout.batched:
                from tianshou.env import SubprocVectorEnv
                env_fns = [lambda env_i=i: create_env_helper(env_i) for i in
                           range(config.experiment.rollout.num_batch_envs)]
                env = SubprocVectorEnv(env_fns)
                # env_name = env.get_env_attr(key="name", id=0)[0]
            else:
                env = create_env_helper()
                # env_name = env.name
            print(env)
            yield env


    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    # shape_meta_list[0]['all_shapes']['value'] = [1]
    # shape_meta_list[0]['all_obs_keys'].append('value')


    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # if checkpoint is specified, load in model weights

    # load training data
    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    return_tuples = (
        train_loader,
        valid_loader,
        eval_env_name_list,
        obs_normalization_stats,
        data_logger,
        obs_normalization_stats,
        action_normalization_stats,
        lang_encoder,
        env_iterator,
        video_dir,
        eval_env_horizon_list,
        vis_dir,
        trainset,
        validset,
        env_meta,
        shape_meta,
        ckpt_dir,
        shape_meta_list
    )

    return return_tuples
