"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import numpy as np
from copy import deepcopy
import torch
import self_correct_robot.utils.tensor_utils as TensorUtils
import self_correct_robot.utils.log_utils as LogUtils
import self_correct_robot.utils.file_utils as FileUtils

from self_correct_robot.utils.dataset import SequenceDataset, R2D2Dataset, MetaDataset


def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    #if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to self_correct_robot module location
    #    base_output_dir = os.path.join(self_correct_robot.__path__[0], base_output_dir)
    #base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    #if os.path.exists(base_output_dir):
    #    if not auto_remove_exp_dir:
    #        ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
    #    else:
    #        ans = "y"
    #    if ans == "y":
    #        print("REMOVING")
    #        shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)

    # vis directory
    vis_dir = os.path.join(base_output_dir, time_str, "vis")
    os.makedirs(vis_dir)
    
    return log_dir, output_dir, video_dir, vis_dir


def load_data_for_training(config, obs_keys, lang_encoder=None):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=valid_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
    else:
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None, lang_encoder=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=True,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        shuffled_obs_key_groups=config.train.shuffled_obs_key_groups,
        lang_encoder=lang_encoder,
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.train.data]
    ds_kwargs["config"] = config
    ds_kwargs["filter_by_attribute"] = [ds_cfg.get("filter_key", filter_by_attribute) for ds_cfg in config.train.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.train.data]
    ds_langs = [ds_cfg.get("lang", None) for ds_cfg in config.train.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=R2D2Dataset if config.train.data_format == "r2d2" else SequenceDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_langs=ds_langs,
        normalize_weights_by_ds_size=False,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_langs,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]

        ds_kwargs_copy["dataset_lang"] = ds_langs[i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def batchify_obs(obs_list):
    """
    TODO: add comments
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None, action_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

        action_normalization_stats (dict): TODO
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    if action_normalization_stats is not None:
        action_normalization_stats = deepcopy(action_normalization_stats)
        params["action_normalization_stats"] = TensorUtils.to_list(action_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None, config=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)

    epoch_avg_value_losses = []

    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)

        # Check if there is any False value in the tensor

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        PROGRESS = 'progress'
        input_batch[PROGRESS] = batch[PROGRESS]

        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    mean_value_loss = np.mean(epoch_avg_value_losses)
    print(f'mean value loss of epoch {epoch} is {mean_value_loss}')
    model.scheduler.step(mean_value_loss)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) if not k.startswith('Parameters') else (k, np.mean(v, axis=0)) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
