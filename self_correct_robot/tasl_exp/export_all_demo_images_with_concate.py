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
import random

# Global function to extract gripper_db from a single dataset entry
def get_gripper_db(exporting_dataset, i):
    print('running', i)
    return exporting_dataset[i]['obs']['robot0_gripper_qpos']


# Step 1: Precompute all the gripper_db values for the entire dataset using parallel processing
def prefetch_gripper_db(exporting_dataset):
    # Use partial to fix the first argument (exporting_dataset) for the function get_gripper_db
    get_gripper_db_partial = partial(get_gripper_db, exporting_dataset)

    with Pool() as pool:
        # Use parallel map with the partial function that includes exporting_dataset
        gripper_db_mapping = list(
            tqdm(pool.imap(get_gripper_db_partial, range(len(exporting_dataset))), total=len(exporting_dataset)))
    return gripper_db_mapping


def find_overlap_length(list1, list2, max_length):
    """
    Find the maximum overlap length between the end of list1 and the beginning of list2.
    """
    for overlap_length in range(1, max_length+1):
        if np.array_equal(list1[-overlap_length:], list2[:overlap_length]):
            return overlap_length
    return 0


def combine_images_horizen(images):
    pil_images = [Image.fromarray((image).astype(np.uint8)) for image in images]

    # Combine images horizontally
    total_width = sum(image.width for image in pil_images)
    max_height = max(image.height for image in pil_images)

    combined_image = Image.new('RGB', (total_width, max_height))

    current_width = 0
    for image in pil_images:
        combined_image.paste(image, (current_width, 0))
        current_width += image.width

    return combined_image

def extract_task_name(file_path):
    # Split the path into parts
    path_parts = file_path.split(os.sep)

    # Find the index of "single_stage" to locate the part you want
    single_stage_index = path_parts.index("single_stage")

    # Extract the task name part (the one after "single_stage")
    task_name = path_parts[single_stage_index + 1]

    return task_name

def extract_and_export_image(all_demo_dataset):

    # import pdb; pdb.set_trace()
    # exporting_dataset = demo_dataset


    # dir_name = f'/home/ubuntu/robocasa-statics/export-images-from-demo/{_tmp_task_name}'
    # dir_name = f'/home/ubuntu/robocasa-statics/export-multi-tasks/{_tmp_task_name}'

    for demo_dataset in all_demo_dataset.datasets:
        exporting_dataset = demo_dataset
        task_name = extract_task_name(exporting_dataset.hdf5_path)
        task_name = '_'.join(task_name.split())

        dir_name_left = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/left_images/'
        dir_name_right = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/right_images/'
        dir_name_hand = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/hand_images/'
        dir_name_task_emb = f'/home/minquangao/robocasa-statics/export-images-from-demo-3k/{task_name}/task_emb/'

        dirs_need_to_create = [dir_name_left, dir_name_right, dir_name_hand, dir_name_task_emb]

        for d in dirs_need_to_create:
            if not os.path.exists(d):
                os.makedirs(d)

        def write_image_with_name(image, dir_name, step, complete_rate, task_description):
            image_path = os.path.join(dir_name, f'{step}_{task_description}_{complete_rate}.png')

            image = np.array(image)
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image)

        def write_task_emb_with_name(task_emb, dir_name, task_desp):
            task_emb_path = os.path.join(dir_name, f'{task_desp}.npy')
            np.save(task_emb_path, task_emb)

        eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

        print('PROCESSING....', task_name)

        for i in tqdm(range(len(exporting_dataset))):
            if random.random() > 0.1: continue

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

            # get three images
            # get task embedding
            # save three images into three folders, left_image, hand_image, right_image
            # each_file_name_will be '{task}_{i}_{complete_rate}.png'
            # save the task embedding into a folder, and the file named '{task}_{i}_{complete_rate}.npy'



def generate_concated_images_from_demo_path(task_name=None, file_path=None):
    config_path_compsoite = "/home/minquangao/completion-infuse-robot/robomimic/scripts/run_configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    if task_name and file_path:
        ext_cfg['train']['data'].append({'path':file_path})
        # print('loading from path ', TASK_PATH_MAPPING[task_name])
    else:
        for path in TASK_MAPPING_50_DEMO.values():
            ext_cfg['train']['data'].append({'path': path})

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

    extract_and_export_image(demo_dataset)


if __name__ == '__main__':
    # import argparse
    #
    #
    # parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    # parser.add_argument('--task_id', type=int, required=True, help='specify the task id to expoert')
    #
    # task_id = parser.parse_args().task_id

    # task_path_mapping = list(TASK_PATH_MAPPING.items())

    # for key, value in TASK_PATH_MAPPING.items():
    #     print('PROCESSING.... ', key)
    #     print('FROM PATH.... ', value)
        generate_concated_images_from_demo_path()

