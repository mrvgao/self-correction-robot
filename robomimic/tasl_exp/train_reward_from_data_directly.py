from torch import nn
import sys
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
from torch.utils.data import DataLoader, random_split
from robomimic.config import config_factory
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import robomimic.utils.lang_utils as LangUtils
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import random
import wandb
from robomimic.tasl_exp.reward_basic_models import ValueDetrModel, ValueViTModel, ValueResNetModelWithText
from robomimic.tasl_exp.task_mapping import TASK_PATH_MAPPING
from collections import namedtuple
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.log_utils import PrintLogger
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
# Mapping task names to file paths

def set_seed(seed):
    random.seed(seed)
    np.random

# Image processing functions
def find_overlap_length(list1, list2, max_length):
    """
    Find the maximum overlap length between the end of list1 and the beginning of list2.
    """
    for overlap_length in range(1, max_length + 1):
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


# Custom dataset that loads images directly from the demo dataset
class CustomImageDataset(Dataset):
    def __init__(self, demo_dataset, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.demo_dataset = demo_dataset
        self.lang_encoder = LangUtils.LangEncoder(device=device)
        self.image_pairs = []
        self._process_demo_dataset()

    def _process_demo_dataset(self):
        eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

        for i in tqdm(range(len(self.demo_dataset))):
            left_db = self.demo_dataset[i]['obs'][eye_names[0]]
            hand_db = self.demo_dataset[i]['obs'][eye_names[1]]
            right_db = self.demo_dataset[i]['obs'][eye_names[2]]
            # task_ds = self.demo_dataset[i]['task_ds']
            # task_name = '_'.join(task_ds.split())
            concatenated_image = combine_images_horizen([left_db[-1], hand_db[-1], right_db[-1]])

            task_embedding = self.demo_dataset[i]['obs']['lang_emb']
            self.image_pairs.append((concatenated_image, task_embedding, i))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image, task_embedding, label = self.image_pairs[idx]

        # Preprocess the image
        image = self.feature_extractor(image)

        return image, task_embedding, torch.tensor(label, dtype=torch.float32)


# Function to generate the demo dataset
def generate_concated_images_from_demo_path():
    config_path_compsoite = "/home/ubuntu/robocasa-statics/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    for i, task_name in enumerate(TASK_PATH_MAPPING):
        if i + 1 > len(ext_cfg['train']['data']):
            ext_cfg['train']['data'].append(ext_cfg['train']['data'][0].copy())
        ext_cfg['train']['data'][i]['path'] = TASK_PATH_MAPPING[task_name]

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda, cuda_mark=config.cuda_mark)

    # Set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(1)
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
        from robomimic.utils.script_utils import deep_update
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

    demo_dataset = trainset

    return demo_dataset


# Main function for training
def main(args):
    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")

    resnet_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Generate the demo dataset
    demo_dataset = generate_concated_images_from_demo_path()

    # Create the dataset
    dataset = CustomImageDataset(demo_dataset, resnet_transformer, device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ValueResNetModelWithText().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    task_name = wandb.run.name
    wandb.watch(model)

    save_dir = f'value_models/{task_name}'
    os.makedirs(save_dir, exist_ok=True)

    early_stopping_patience = 3
    best_val_loss = float('inf')
    early_stopping_counter = 0

    total_steps = 0

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Training]", leave=True)

        for i, (images, task_embs, labels) in enumerate(progress_bar):
            images, task_embs, labels = images.to(device), task_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task_embs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            total_steps += 1

            if total_steps % 100 == 0:
                wandb.log({"total-steps": total_steps, "train_loss": loss.item()})

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}")

        # Validation and checkpointing
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Validation]",
                                    leave=True)
                for i, (images, task_embs, labels) in enumerate(progress_bar):
                    images, task_embs, labels = images.to(device), task_embs.to(device), labels.to(device)
                    outputs = model(images, task_embs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                scheduler.step(avg_val_loss)
                wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    wandb.finish()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    # parser.add_argument('--name', type=str, required=False, help='Name for the training task.')
    # parser.add_argument('--model', type=str, required=False, choices=['detr', 'vit', 'resnet'], help='Name for the selection model')
    # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    # parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
    # parser.add_argument('--cuda', type=str, required=True, help='the No of cuda')
    # parser.add_argument('--seed', type=int, required=True, help='training random seed')
    # parser.add_argument('--task_dir', type=str, default=None, required=False, help='specify a task')
    # args = parser.parse_args()

    name = 'all-tasks-in-one-with-full-batch'
    model = 'resnet'
    lr = 1e-5
    bs = 2048
    num_epochs = 1000
    cuda = 0
    seed = 999

    # sub_tasks = [
    #     'close-double-door', 'close-single-door', 'close-single-door',  'open-double-door',
    #     'open-drawer', 'open-single-door', 'pick-from-counter-and-place-to-microwave',
    #     'pick-from-counter-and-place-to-sink', 'pick-from-counter-and-place-to-stove',
    #     'pick-from-microwave-and-place-to-counter', 'pick-from-sink-and-place-to-counter',
    #     'pick-from-stove-and-place-to-counter', 'press-coffee-maker-button',
    #     'serving-coffee-in-a-mug', 'setup-a-coffee-mug', 'turn-off-microwave',
    #     'turn-off-sink-faucet', 'turn-off-stove', 'turn-on-microwave', 'turn-on-sink-faucent',
    #     'turn-on-stove', 'turn-sink-spout'
    # ]
    #
    # assert len(sub_tasks) == 22
    #
    Args = namedtuple(
        'Args',
        ['name', 'model', 'lr',  'batch_size', 'num_epochs', 'cuda', 'seed']
    )
    #
    # for i, task_dir in enumerate(sub_tasks):
    args = Args(name, model, lr, bs, num_epochs, cuda, seed)
    run_name = f"all_task_model_{model}_lr_{lr}_bs_{bs}_epochs_{num_epochs}_seed_{seed}"
    wandb.init(project="value-model-for-all-single-tasks", entity="minchiuan", name=run_name, config={
        "system_metrics": True  # Enable system metrics logging
    })
    main(args)
