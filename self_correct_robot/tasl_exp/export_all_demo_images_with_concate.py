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

    # dir_name = f'/home/ubuntu/robocasa-statics/export-images-from-demo/{_tmp_task_name}'
    # dir_name = f'/home/ubuntu/robocasa-statics/export-multi-tasks/{_tmp_task_name}'

    task_name = '_'.join(task_name.split())

    dir_name = f'/home/ubuntu/robocasa-statics/export-images-from-demo/{task_name}'

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    PNG_ID = 1
    TASK_ID = 1

    def write_image_with_name(image1, image2, image3, task_ds_dir, png_id):
        task_dir = os.path.join(dir_name, task_ds_dir)

        if not os.path.exists(task_dir): os.mkdir(task_dir)

        whole_image = combine_images_horizen([image1, image2, image3])
        whole_image = np.array(whole_image)
        whole_image = whole_image.astype(np.uint8)
        whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(task_dir, f'{png_id}.png')
        cv2.imwrite(image_path, whole_image)

    def write_several_images(images1, images2, image3, task_ds_dir, start_id):
        for l, h, r in zip(images1, images2, image3):
            write_image_with_name(l, h, r, task_ds_dir, start_id)
            start_id += 1

    # MAX_TRY = 10
    previous_delta = None

    for i in tqdm(range(len(exporting_dataset))):
        left_db = exporting_dataset[i]['obs'][eye_names[0]]
        hand_db = exporting_dataset[i]['obs'][eye_names[1]]
        right_db = exporting_dataset[i]['obs'][eye_names[2]]
        gripper_db = exporting_dataset[i]['obs']['robot0_gripper_qpos']
        task_ds = exporting_dataset[i]['task_ds']

        # dir_name = f'/home/minquangao/export-images-from-demo/{task_ds_dir}'

        delta = np.mean(gripper_db[-1] - gripper_db[-2])

        change_task = (previous_delta == 0 and delta != 0)

        if i == 0 or change_task:
            task_ds_dir = '_'.join(task_ds.split()) + '_ID_' + str(TASK_ID)
            TASK_ID += 1
            PNG_ID = 1
            print(f'task name: {task_name}: NEW sub-TASK: {task_ds_dir}')

        previous_delta = delta

        if i == 0:
            write_several_images(left_db, hand_db, right_db, task_ds_dir, start_id=1)
            PNG_ID += len(left_db)
        else:
            write_image_with_name(left_db[-1], right_db[-1], hand_db[-1], task_ds_dir, png_id=PNG_ID)
            PNG_ID += 1


def generate_concated_images_from_demo_path(task_name):
    config_path_compsoite = "/home/ubuntu/robocasa-statics/configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    ext_cfg['train']['data'][0]['path'] = TASK_MAPPING_50_DEMO[task_name]
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


def process_task(task_name):
    print(f'Processing... {task_name}')
    generate_concated_images_from_demo_path(task_name=task_name)


if __name__ == '__main__':
    import concurrent.futures

    max_workers = min(30, len(TASK_PATH_MAPPING))  # Assuming 30 CPUs available

    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the pool
        futures = {executor.submit(process_task, task_name): task_name for task_name in TASK_PATH_MAPPING}

        # Ensure all tasks are completed
        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                future.result()
                print(f'{task_name} processed successfully.')
            except Exception as e:
                print(f'Error processing {task_name}: {e}')