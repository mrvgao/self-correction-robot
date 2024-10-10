import json
import numpy as np
import os
import sys
import socket
import traceback
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import self_correct_robot
import self_correct_robot.utils.train_utils as TrainUtils
import self_correct_robot.utils.torch_utils as TorchUtils
import self_correct_robot.utils.obs_utils as ObsUtils
import self_correct_robot.utils.env_utils as EnvUtils
import self_correct_robot.utils.file_utils as FileUtils
import self_correct_robot.utils.lang_utils as LangUtils
from self_correct_robot.config import config_factory
from self_correct_robot.algo import algo_factory, RolloutPolicy
from self_correct_robot.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from collections import Counter
from self_correct_robot.tasl_exp.task_mapping import TASK_MAPPING_50_DEMO,TASK_PATH_MAPPING
import time
from multiprocessing import Pool
from functools import partial

def write_image_with_name(image, dir_name, step, complete_rate, task_description):
    image_path = os.path.join(dir_name, f'{step}_{task_description}_{complete_rate}.png')
    image = np.array(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image)

def write_task_emb_with_name(task_emb, dir_name, task_desp):
    task_emb_path = os.path.join(dir_name, f'{task_desp}.npy')
    np.save(task_emb_path, task_emb)

def process_single_item(i, exporting_dataset, eye_names, dir_name_left, dir_name_hand, dir_name_right, dir_name_task_emb):
    left_image = exporting_dataset[i]['obs'][eye_names[0]][0]
    hand_image = exporting_dataset[i]['obs'][eye_names[1]][0]
    right_image = exporting_dataset[i]['obs'][eye_names[2]][0]
    task_emb = exporting_dataset[i]['obs']['lang_emb'][0]

    demo_id = exporting_dataset._index_to_demo_id[i]
    demo_start_index = exporting_dataset._demo_id_to_start_indices[demo_id]
    demo_length = exporting_dataset._demo_id_to_demo_length[demo_id]

    # start at offset index if not padding for frame stacking
    demo_index_offset = 0 if exporting_dataset.pad_frame_stack else (exporting_dataset.n_frame_stack - 1)
    index_in_demo = i - demo_start_index + demo_index_offset

    complete_rate = round(index_in_demo / demo_length, 4)

    task_description = exporting_dataset._demo_id_to_demo_lang_str[demo_id]
    task_description = '_'.join(task_description.split())

    write_image_with_name(left_image, dir_name_left, i, complete_rate, task_description)
    write_image_with_name(hand_image, dir_name_hand, i, complete_rate, task_description)
    write_image_with_name(right_image, dir_name_right, i, complete_rate, task_description)
    write_task_emb_with_name(task_emb, dir_name_task_emb, task_description)

def extract_and_export_image(demo_dataset, task_name):
    task_name = '_'.join(task_name.split())
    dir_name_left = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/left_images/'
    dir_name_right = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/right_images/'
    dir_name_hand = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/hand_images/'
    dir_name_task_emb = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/task_emb/'

    dirs_need_to_create = [dir_name_left, dir_name_right, dir_name_hand, dir_name_task_emb]

    for d in dirs_need_to_create:
        if not os.path.exists(d):
            os.makedirs(d)

    eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

    # Partial function to fix the arguments except for the index `i`
    partial_process_item = partial(process_single_item,
                                   exporting_dataset=demo_dataset,
                                   eye_names=eye_names,
                                   dir_name_left=dir_name_left,
                                   dir_name_hand=dir_name_hand,
                                   dir_name_right=dir_name_right,
                                   dir_name_task_emb=dir_name_task_emb)

    # Set the number of workers (processes) based on available CPU cores
    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress while processing the dataset in parallel
        list(tqdm(executor.map(partial_process_item, range(len(demo_dataset))), total=len(demo_dataset)))

def generate_concated_images_from_demo_path(task_name, file_path):
    config_path_compsoite = "/home/minquangao/completion-infuse-robot/robomimic/scripts/run_configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    ext_cfg['train']['data'][0]['path'] = file_path
    # print('loading from path ', TASK_PATH_MAPPING[task_name])
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda, cuda_mark=config.cuda_mark)

    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # print(config)
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

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
        # print("\n============= Loaded Environment Metadata =============")
        # print(dataset_path)
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
            verbose=False
        )
        shape_meta_list.append(shape_meta)

    # if config.experiment.env is not None:
    #     env_meta["env_name"] = config.experiment.env
    #     print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    # for (dataset_i, dataset_cfg) in enumerate(config.train.data):
    #     do_eval = dataset_cfg.get("do_eval", True)
    #     if do_eval is not True:
    #         continue
    #     eval_env_meta_list.append(env_meta_list[dataset_i])
    #     eval_shape_meta_list.append(shape_meta_list[dataset_i])
    #     eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
    #     horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
    #     eval_env_horizon_list.append(horizon)

    # save the config as a json file
    # with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
    #     json.dump(config, outfile, indent=4)

    # if checkpoint is specified, load in model weights

    # load training data
    # lang_encoder = LangUtils.LangEncoder(
    #     device=device,
    # )
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=None)

    # TODO: combine trainset and validset
    demo_dataset = trainset

    extract_and_export_image(demo_dataset, task_name=task_name)


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    parser.add_argument('--task_id', type=int, required=True, help='specify the task id to expoert')

    task_id = parser.parse_args().task_id

    task_path_mapping = list(TASK_PATH_MAPPING.items())

    # for key, value in TASK_PATH_MAPPING.items():
    #     print('PROCESSING.... ', key)
    #     print('FROM PATH.... ', value)
    generate_concated_images_from_demo_path(task_name=task_path_mapping[task_id][0], file_path=task_path_mapping[task_id][1])

