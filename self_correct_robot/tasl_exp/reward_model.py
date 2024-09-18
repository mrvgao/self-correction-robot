import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from self_correct_robot.utils.lang_utils import LangEncoder
import random
import numpy as np
from collections import namedtuple
from self_correct_robot.tasl_exp.reward_basic_models import ValueDetrModel, ValueViTModel, ValueResNetModelWithText, ValueResNetModelWithTextWithAttnAndResidual
import argparse
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
import time
import hashlib
from self_correct_robot.utils.load_dataloader import load_dataloader
import json
from self_correct_robot.config import config_factory


torch.backends.cudnn.benchmark = True

try:
    scaler = torch.amp.GradScaler()
except AttributeError:
    scaler = torch.cuda.amp.GradScaler()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, device, target_task=None):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.image_pairs = []
        self.lang_encoder = LangEncoder(device=device)
        self.target_task = target_task
        self.tasks = []
        self.task_embeddings = []

        # Prepare a list of all image pairs and corresponding labels
        for ti, task_name_dir in enumerate(os.listdir(root_dir)):
            if target_task is not None and task_name_dir != target_task: continue

            for task_ds_str in os.listdir(os.path.join(root_dir, task_name_dir)):

                task_path = os.path.join(root_dir, task_name_dir, task_ds_str) # export/prepare-coffee/1

                rm_index = task_ds_str.find('_ID_')
                task_ds_str = task_ds_str[:rm_index]

                current_task_name = ' '.join(task_ds_str.split('_'))
                print('init task: ', current_task_name)
                if os.path.isdir(task_path):
                    images = sorted([f for f in os.listdir(task_path) if f.endswith('.png')])
                    # Get the numeric parts of the filenames
                    image_numbers = sorted([int(os.path.splitext(f)[0]) for f in images])

                    N = image_numbers[-1]

                    for i in range(len(image_numbers)):
                        image_path = os.path.join(task_path, f'{image_numbers[i]}.png')

                        # Calculate the label
                        # label = -1 * (N - image_numbers[i]) + 1
                        label = image_numbers[i] / N
                        self.image_pairs.append((image_path, label))
                        self.tasks.append(current_task_name)

        self._batch_process_tasks()

    def _batch_process_tasks(self):
        """
        Batch process task embeddings for all task names.
        """
        print("Batch processing task embeddings...")
        feature_batch_size = 1024
        for i in tqdm(range(0, len(self.tasks), feature_batch_size)):
            batch_task_names = self.tasks[i:i + feature_batch_size]
            task_embeddings = self.lang_encoder.get_lang_emb(batch_task_names)
            self.task_embeddings.extend(task_embeddings)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_path, label = self.image_pairs[idx]

        # Load the images
        image = Image.open(image_path).convert('RGB')
        image = self.feature_extractor(image)

        task_embedding = self.task_embeddings[idx]

        return image, task_embedding, torch.tensor(label, dtype=torch.float32)


class NumpyToTensor:
    """Custom transform to convert a NumPy array directly to a PyTorch tensor."""
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # Convert NumPy array (H x W x C) to PyTorch tensor (C x H x W)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img


resnet_transformer = transforms.Compose([
    NumpyToTensor(),  # C
    transforms.Resize((224, 224), antialias=True),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Custom Dataset that applies the transformer
class RoboCustomDataset(Dataset):
    def __init__(self, dataset, transformer):
        self.dataset = dataset  # Original dataset from the DataLoader
        self.transformer = transformer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Assuming the dataset is structured as dataset[idx]['obs']
        obs_data = self.dataset[idx]['obs']
        progress_label = torch.tensor(self.dataset[idx]['progress'], dtype=torch.float32)
        left_image = obs_data['robot0_agentview_left_image'][0]
        hand_image = obs_data['robot0_eye_in_hand_image'][0]
        right_image = obs_data['robot0_agentview_right_image'][0]
        task_emb = torch.tensor(obs_data['lang_emb'][0], dtype=torch.float32)

        # Extract the three necessary features, you can customize these keys
        # Apply the resnet_transformer to each feature
        feature_left = self.transformer(left_image)
        feature_hand = self.transformer(hand_image)
        feature_right = self.transformer(right_image)

        return feature_left, feature_hand, feature_right, task_emb, progress_label


def split_valid_test_from_robo_config_dataset(config, batch_size):

    my_dataloader = load_dataloader(config, device='cuda')[0]

    dataset = my_dataloader.dataset  # Retrieve the dataset from the existing DataLoader

    # Step 1: Define the lengths for train, validation, and test splits
    dataset_size = len(dataset)
    train_size = int(0.80 * dataset_size)  # 70% training
    val_size = int(0.10 * dataset_size)  # 15% validation
    test_size = dataset_size - train_size - val_size  # Remaining for test

    # Step 2: Split the dataset into train, validation, and test
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Step 3: Wrap the datasets into the custom dataset class that applies the transformation
    train_dataset = RoboCustomDataset(train_dataset, resnet_transformer)
    val_dataset = RoboCustomDataset(val_dataset, resnet_transformer)
    test_dataset = RoboCustomDataset(test_dataset, resnet_transformer)

    # Step 4: Create new DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main(args):
    set_seed(args.seed)

    # device = torch.device(f'cuda:{args.cuda}')
    # print(f"Using device: {device}")

    train_dataloader, val_dataloader, test_dataloader = split_valid_test_from_robo_config_dataset(args.config, batch_size=args.batch_size)

    # Create the dataloader
    # Example training loop
    num_epochs = args.num_epochs
    # model = ProgressViTModel().to(device)
    if args.model == 'detr':
        model = ValueDetrModel().to(device)
    elif args.model == 'vit':
        model = ValueViTModel().to(device)
    elif args.model == 'resnet':
        model = ValueResNetModelWithText().to(device)
    elif args.model == 'attn':
        model = ValueResNetModelWithTextWithAttnAndResidual().to(device)
    else:
        raise ValueError("unsupported model name, ", args.model)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6], output_device=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)

    task_name = wandb.run.name
    wandb.watch(model)

    save_dir = f'value_models/{task_name}'
    os.makedirs(save_dir, exist_ok=True)

    early_stopping_patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)
        batch_loss = 0

        for i, (img_1, img_2, img_3, task_embs, labels) in enumerate(progress_bar):
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            img_3 = img_3.to(device)
            task_embs, labels = task_embs.to(device), labels.to(device)

            # if args.task_dir:
            #     task_embs = None
            optimizer.zero_grad()
            # outputs = model(image1, image2)
            with autocast():
                outputs = model(img_1, img_2, img_3, task_embs)
                loss = criterion(outputs, labels.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": avg_loss})
                running_loss = 0.0

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {batch_loss / len(train_dataloader):.4f}")

        # Save the model

        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=True)
                for i, (images, task_embs, labels) in enumerate(progress_bar):
                    images, task_embs, labels = images.to(device), task_embs.to(device), labels.to(device)
                    if args.task_dir:
                        task_embs = None

                    with autocast():
                        outputs = model(images, task_embs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss, 'loss': current_lr})
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    print("Evaluating the model on the test dataset...")

    # Load the best model if it exists
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for test evaluation.")
    else:
        print("Best model not found, using the last saved model.")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Evaluating on Test Dataset", leave=True)
        for i, (img_1, img_2, img_3, task_embs, labels) in enumerate(progress_bar):
            img_1, img_2, img_3 = img_1.to(device), img_2.to(device), img_3.to(device)
            task_embs, labels = task_embs.to(device), labels.to(device)

            with autocast():
                outputs = model(img_1, img_2, img_3, task_embs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Final Test Loss: {avg_test_loss:.4f}")
        wandb.log({"test_loss": avg_test_loss})

    wandb.finish()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,7"

    device = torch.device('cuda')  # This will refer to the first available GPU in your restricted list

    parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    parser.add_argument('--config', type=str, required=True, help='give the data path config')
    parser.add_argument('--tag', type=str, required=True, help='Add a tag to make logger easier')
    parser.add_argument('--model', type=str, required=False, choices=['attn', 'resnet'], help='Name for the selection model')
    parser.add_argument('--seed', type=int, required=True, help='training random seed')
    args = parser.parse_args()

    # config_path = '/home/ubuntu/self-correction-robot/self_correct_robot/scripts/running_configs/lambda_multi-task-with-value-correct-seed-999.json'

    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    split_valid_test_from_robo_config_dataset(config=config, batch_size=512)
    # mp.set_start_method('spawn', force=True)  # Set multiprocessing to use 'spawn'
    #
    # parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    # # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    # # parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    # # parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
    # # parser.add_argument('--cuda', type=str, required=True, help='the No of cuda')
    # # parser.add_argument('--task_dir', type=str, default=None, required=False, help='specify a task')
    #
    timestamp = str(int(time.time()))
    signature = hashlib.sha256(timestamp.encode()).hexdigest()

    name = f'{args.tag}_{str(signature)}_all-tasks-in-one-predicate-progress'
    model = args.model
    lr = 1e-4
    # num_epochs = 1000
    num_epochs = 10
    cuda = 0
    # seed = 999
    batch_size = 128

    Args = namedtuple(
        'Args',
        ['name', 'model', 'lr',  'batch_size', 'num_epochs', 'cuda', 'seed', 'task_dir', 'config']
    )
    # #
    # # for i, task_dir in enumerate(sub_tasks):
    args = Args(name, model, lr, batch_size, num_epochs, cuda, args.seed, None, config)
    run_name = f"{name}_{model}_lr_{lr}_bs_{batch_size}_epochs_{num_epochs}_seed_{args.seed}"
    # wandb.init(project="value-model-trained-by-3k", entity="minchiuan", name=run_name, config={
    #     ""
    #     "system_metrics": True  # Enable system metrics logging
    # })
    main(args)
