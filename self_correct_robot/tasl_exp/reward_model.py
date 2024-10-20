import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from self_correct_robot.utils.lang_utils import LangEncoder
import random
import numpy as np
from collections import namedtuple
from self_correct_robot.tasl_exp.reward_basic_models import ValueDetrModel, ValueViTModel, ValueResNetWithAttnPerformance, ValueResNetModelWithTextWithAttnAndResidual
import argparse
from torch.cuda.amp import autocast
from torch.utils.data import Subset
import torch.multiprocessing as mp
import time
import hashlib
from self_correct_robot.utils.load_dataloader import load_dataloader
import json

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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transformer = transform
        self.lang_embedding = dict()
        self.data = []

        for sub_task in os.listdir(root_dir):
            if len(self.data) > 500: # For testing
                break
            sub_task_dir = os.path.join(root_dir, sub_task)

            for task_desc in os.listdir(os.path.join(sub_task_dir, 'task_emb')):
                task_emb_path = os.path.join(sub_task_dir, 'task_emb', task_desc)
                if not os.path.isfile(task_emb_path):
                    print(f"Task embedding not found for {task_emb_path}, skipping.")
                    continue
                else:
                    print('load task embedding for ', task_desc)
                    task_desc_name = task_desc.split('.')[0]
                    self.lang_embedding[task_desc_name] = np.load(task_emb_path)

            image_dirs = ['left_images', 'hand_images', 'right_images']

            image_paths = {img_dir: [] for img_dir in image_dirs}

            for img_dir in image_dirs:
                print('load image for ', img_dir)
                for img_name in os.listdir(os.path.join(sub_task_dir, img_dir)):
                    img_path = os.path.join(sub_task_dir, img_dir, img_name)
                    if not os.path.isfile(img_path):
                        continue

                    tokens = img_path.split('/')[-1].split('_')
                    label = float(tokens[-1].replace('.png', ''))
                    task_name = '_'.join(tokens[1:-1])
                    image_paths[img_dir].append((img_path, task_name, label))

            num_images = min(len(image_paths[img_dir]) for img_dir in image_dirs)

            for i in range(num_images):
                img_1_path, task_name, label = image_paths['left_images'][i]
                img_2_path, task_name_1, label_1 = image_paths['hand_images'][i]
                img_3_path, task_name_2, label_2 = image_paths['right_images'][i]

                assert task_name == task_name_1 == task_name_2
                assert label == label_1 == label_2

                self.data.append((img_1_path, img_2_path, img_3_path, task_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        left_image, hand_image, right_image, task_name, label = self.data[idx]

        # Load images
        left_image = Image.open(left_image).convert('RGB')
        hand_image = Image.open(hand_image).convert('RGB')
        right_image = Image.open(right_image).convert('RGB')

        if self.transformer:
            left_image = self.transformer(left_image)
            hand_image = self.transformer(hand_image)
            right_image = self.transformer(right_image)

        task_emb = self.lang_embedding[task_name]
        task_emb = torch.tensor(task_emb, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)

        return left_image, hand_image, right_image, task_emb, label


def create_dataloaders(my_dataset, batch_size, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Create train, validation, and test dataloaders from a dataset.

    Args:
        dataset (Dataset): The dataset to split.
        batch_size (int): The batch size for the dataloaders.
        train_split (float): Fraction of the dataset to use for training.
        val_split (float): Fraction of the dataset to use for validation.
        test_split (float): Fraction of the dataset to use for testing.

    Returns:
        train_loader, val_loader, test_loader: DataLoader for each split.
    """
    # Ensure the splits sum to 1
    assert train_split + val_split + test_split == 1, "Splits must sum to 1."

    # Calculate the number of samples in each split
    total_size = len(my_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size  # Ensure all samples are used

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])

    # Create DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main(args):
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")

    train_dataset = CustomDataset('/home/minquangao/robocasa-statics/export-images-from-demo-3k', transform)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, args.batch_size)

    # Create the dataloader
    # Example training loop
    num_epochs = args.num_epochs
    # model = ProgressViTModel().to(device)
    if args.model == 'detr':
        model = ValueDetrModel().to(device)
    elif args.model == 'vit':
        model = ValueViTModel().to(device)
    # elif args.model == 'resnet':
    #     model = ValueResNetModelWithText().to(device)
    elif args.model == 'attn':
        model = ValueResNetWithAttnPerformance().to(device)
    else:
        raise ValueError("unsupported model name, ", args.model)
    if args.ckpt:
        print('load pretrainde model, ', args.ckpt)
        model.load_state_dict(torch.load(args.ckpt))
        model = model.to(device)

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

    # test_dataloader_iter = iter(test_dataloader)
    # train_dataloader_iter = iter(train_dataloader)
    # val_dataloader_iter = iter(val_dataloader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_loss = 0

        batch_step = 0

        for img_1, img_2, img_3, task_embs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True):

            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            img_3 = img_3.to(device)
            task_embs, labels = task_embs.to(device), labels.to(device)

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
            if batch_step % 5 == 0:  # Log every 10 batches
                avg_loss = running_loss / 5
                wandb.log({"epoch": epoch + 1, "batch": batch_step + 1, "train_loss": avg_loss})
                running_loss = 0.0

            batch_step += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {batch_loss / len(train_dataloader):.4f}")

        # Save the model

        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img_1, img_2, img_3, task_embs, labels in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=True):
                    img_1, img_2, img_3 = img_1.to(device), img_2.to(device), img_3.to(device)
                    task_embs, labels = task_embs.to(device), labels.to(device)

                    if args.task_dir:
                        task_embs = None

                    with autocast():
                        outputs = model(img_1, img_2, img_3, task_embs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss, 'lr': current_lr})
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

    # Move the model to CPU
    model = model.to('cpu')
    torch.cuda.empty_cache()

    # Load the best model if it exists
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        with torch.no_grad():
            model = model.to(device)
        print(f"Loaded best model from {best_model_path} for test evaluation.")
    else:
        print("Best model not found, using the last saved model.")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for img_1, img_2, img_3, task_embs, labels in tqdm(test_dataloader, desc="Evaluating on Test Dataset", leave=True):
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
    device = torch.device('cuda')  # This will refer to the first available GPU in your restricted list

    parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    parser.add_argument('--tag', type=str, required=True, help='Add a tag to make logger easier')
    parser.add_argument('--model', type=str, required=False, choices=['attn', 'resnet'], help='Name for the selection model')
    parser.add_argument('--seed', type=int, required=True, help='training random seed')
    parser.add_argument('--ckpt', type=str, default=None, required=False, help='specify a pretraind model path')
    args = parser.parse_args()

    timestamp = str(int(time.time()))
    signature = hashlib.sha256(timestamp.encode()).hexdigest()

    name = f'{args.tag}_{str(signature)}_all-tasks-in-one-predicate-progress'
    model = args.model
    lr = 1e-4
    # num_epochs = 1000
    num_epochs = 100
    cuda = 0
    # seed = 999
    batch_size = 128

    Args = namedtuple(
        'Args',
        ['name', 'model', 'lr',  'batch_size', 'num_epochs',
                    'cuda', 'seed', 'task_dir', 'ckpt']
    )
    # #
    # # for i, task_dir in enumerate(sub_tasks):
    mock_args = Args(name, model, lr, batch_size, num_epochs, cuda, args.seed, None, args.ckpt)
    run_name = f"{name}_{model}_lr_{lr}_bs_{batch_size}_epochs_{num_epochs}_seed_{args.seed}"

    wandb.init(project="value-model-trained-by-3k", entity="minchiuan", name=run_name, config={
        "system_metrics": True  # Enable system metrics logging
    })

    main(mock_args)
