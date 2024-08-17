
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import json
from robomimic.config import config_factory
import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

TASK_PATH_MAPPING = {
      'restock pantry': '/data3/mgao/robocasa/datasets/v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10/demo_im128.hdf5',
      'microwave thawing': '/data3/mgao/robocasa/datasets/v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11/demo_im128.hdf5',
      'arrange vegetables' : '/data3/mgao/robocasa/datasets/v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11/demo_im128.hdf5',
      'prepare for socking pan': '/data3/mgao/robocasa/datasets/v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10/demo_im128.hdf5',
      'prepare coffee' : '/data3/mgao/robocasa/datasets/v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07/demo_im128.hdf5',
      'turn sink spout': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29/demo_gentex_im128_randcams.hdf5',
      'turn on sink faucent': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'turn off sink faucet': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'turn on stove': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5',
      'turn off stove': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5',
      'close drswer': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams.hdf5',
      'open drawer': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5',
      'pick from cabinet and place to counter': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5',
      'pick from counter and place to stove' : '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams.hdf5',
      'pick from counter and place to cabinet': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams.hdf5',
      'pick from microwave and place to counter': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5',
      'pick from stove and place to counter': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams.hdf5',
      'pick from counter and place to sink': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'pick from sink and place to counter': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5',
      'pick from counter and place to microwave': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5',
      'open double door': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5',
      'open single door': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5',
      'close double door': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5',
      'close single door': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5',
      'setup a coffee mug': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'press coffee maker button': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'serving coffee in a mug': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5',
      'turn off microwave': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5',
      'turn on microwave': '/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5'
}

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


def extract_and_export_image(demo_dataset, task_name):

    max_check_length = 50  # Maximum length to check for overlap

    exporting_dataset = demo_dataset

    eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

    _tmp_task_name = '-'.join(task_name.split())
    dir_name = f'/data3/mgao/export-images-from-demo/{_tmp_task_name}'
    # dir_name = f'/home/minquangao/export-images-from-demo/{_tmp_task_name}'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    PNG_ID = 1
    TASK_ID = 1

    def write_image_with_name(image1, image2, image3, task_id, png_id):
        task_dir = os.path.join(dir_name, str(task_id))

        if not os.path.exists(task_dir): os.mkdir(task_dir)

        whole_image = combine_images_horizen([image1, image2, image3])
        whole_image = np.array(whole_image)
        whole_image = whole_image.astype(np.uint8)
        whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(task_dir, f'{png_id}.png')
        cv2.imwrite(image_path, whole_image)

    def write_several_images(images1, images2, image3, task_id, start_id):
        for l, h, r in zip(images1, images2, image3):
            write_image_with_name(l, h, r, task_id, start_id)
            start_id += 1

    # MAX_TRY = 10
    previous_delta = None

    for i in tqdm(range(len(exporting_dataset))):
        left_db = exporting_dataset[i]['obs'][eye_names[0]]
        hand_db = exporting_dataset[i]['obs'][eye_names[1]]
        right_db = exporting_dataset[i]['obs'][eye_names[2]]
        gripper_db = exporting_dataset[i]['obs']['robot0_gripper_qpos']

        delta = np.mean(gripper_db[-1] - gripper_db[-2])

        if previous_delta == 0 and delta != 0:
            TASK_ID += 1
            print(f'{task_name}: task_name NEW sub-TASK: {TASK_ID}')
            PNG_ID = 1

        previous_delta = delta

        if i == 0:
            write_several_images(left_db, hand_db, right_db, task_id=TASK_ID, start_id=1)
        else:
            write_image_with_name(left_db[-1], right_db[-1], hand_db[-1], task_id=TASK_ID, png_id=PNG_ID)
            PNG_ID += 1

    if TASK_ID != 50:
        print(f'WARNING!!! task: {task_name} did not get 50 tasks, it got : {TASK_ID} tasks')


def generate_concated_images_from_demo_path(task_name):
    config_path_compsoite = "/data2/mgao/pretrained-models/configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))
    ext_cfg['train']['data'][0]['path'] = TASK_PATH_MAPPING[task_name]
    print('loading from path ', TASK_PATH_MAPPING[task_name])
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

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    print(config)
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
        print("\n============= Loaded Environment Metadata =============")
        print(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
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
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)

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

    # TODO: combine trainset and validset
    demo_dataset = trainset

    extract_and_export_image(demo_dataset, task_name=task_name)


if __name__ == '__main__':
    for key, value in tqdm(TASK_PATH_MAPPING.items()):
        generate_concated_images_from_demo_path(task_name=key)

