import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from robomimic.utils.lang_utils import LangEncoder
import random
import numpy as np
from collections import namedtuple
from robomimic.tasl_exp.reward_basic_models import ValueDetrModel, ValueViTModel, ValueResNetModelWithText, ValueResNetModelWithTextWithAttnAndResidual
import argparse
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from torchvision.transforms.functional import resize, normalize



torch.backends.cudnn.benchmark = True

scaler = torch.amp.GradScaler()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, device, target_task=None, batch_size=1024):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.image_pairs = []
        self.lang_encoder = LangEncoder(device=device)
        self.target_task = target_task
        self.images = []
        self.task_embeddings = []
        self.labels = []
        self.batch_size = batch_size
        self.device = device

        image_paths = []
        task_names = []

        # Prepare a list of all image pairs and corresponding labels
        for ti, task_name_dir in enumerate(os.listdir(root_dir)):
            if target_task is not None and task_name_dir != target_task: continue

            for task_ds_str in os.listdir(os.path.join(root_dir, task_name_dir)):
                task_path = os.path.join(root_dir, task_name_dir, task_ds_str)
                rm_index = task_ds_str.find('_ID_')
                task_ds_str = task_ds_str[:rm_index]
                current_task_name = ' '.join(task_ds_str.split('_'))
                print(f'{ti}/{len(os.listdir(root_dir))} init task: {current_task_name}')

                if os.path.isdir(task_path):
                    images = sorted([f for f in os.listdir(task_path) if f.endswith('.png')])
                    image_numbers = sorted([int(os.path.splitext(f)[0]) for f in images])
                    N = image_numbers[-1]

                    for i in range(len(image_numbers)):
                        image_path = os.path.join(task_path, f'{image_numbers[i]}.png')
                        label = image_numbers[i] / N
                        image_paths.append(image_path)
                        task_names.append(current_task_name)
                        self.labels.append(label)

        # Perform batched image feature extraction
        self._batch_process_images(image_paths)

        # Perform batched task embedding extraction
        self._batch_process_tasks(task_names)

    def _batch_process_images(self, image_paths):
        """
        Batch process image preprocessing (resize, normalize, etc.) for all images and save them on CPU.
        """
        print("Batch processing images...")
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_image_paths = image_paths[i:i + self.batch_size]

            # Load the batch of images
            images = [Image.open(p).convert('RGB') for p in batch_image_paths]

            # Convert list of images to a single batch tensor by stacking them
            images = torch.stack([transforms.ToTensor()(image) for image in images])  # Apply ToTensor to each image

            # Apply batch-level resizing and normalization using torchvision's functional transforms
            images = resize(images, [224, 224])  # Resizing all images in the batch at once
            images = normalize(images, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # Normalize all images in the batch

            # Store the preprocessed images on CPU
            self.images.extend(images.cpu())  # Ensure the images stay on the CPU

    def _batch_process_tasks(self, task_names):
        """
        Batch process task embeddings for all task names.
        """
        print("Batch processing task embeddings...")
        for i in tqdm(range(0, len(task_names), self.batch_size)):
            batch_task_names = task_names[i:i + self.batch_size]
            task_embeddings = self.lang_encoder.get_lang_emb(batch_task_names)
            self.task_embeddings.extend(task_embeddings)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Move data to GPU only when accessing it during training
        image = self.images[idx].to(self.device)  # Move the image to GPU only when needed
        task_embedding = self.task_embeddings[idx].to(self.device)  # Move task embedding to GPU
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)  # Move label to GPU

        return image, task_embedding, label


def combined_lr_lambda(step: int, warmup_steps: int, decay_fn):
    if step < warmup_steps:
        return step / warmup_steps
    return decay_fn(step - warmup_steps)


def main(args):
    # Initialize the feature extractor

    # Argument Parsing
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")

    resnet_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Create the dataset
    root_dir = '/home/ubuntu/robocasa-statics/export-images-from-demo'
    # dataset = CustomImageDataset(root_dir, feature_extractor if args.model != 'resnet' else resnet_transformer)
    dataset = CustomImageDataset(root_dir, resnet_transformer, device, target_task=args.task_dir)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)

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

    criterion = nn.MSELoss()

    warmup_steps = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # T_max is the number of epochs (or iterations) until the learning rate reaches its minimum

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    task_name = wandb.run.name
    wandb.watch(model)

    save_dir = f'value_models/{task_name}'
    os.makedirs(save_dir, exist_ok=True)

    early_stopping_patience = 3
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)
        batch_loss = 0

        for i, (images, task_embs, labels) in enumerate(progress_bar):
            images, task_embs, labels = images.to(device), task_embs.to(device), labels.to(device)
            if args.task_dir:
                task_embs = None
            optimizer.zero_grad()
            # outputs = model(image1, image2)
            with autocast():
                outputs = model(images, task_embs)
                loss = criterion(outputs, labels.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()

            running_loss += loss.item()
            batch_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
                running_loss = 0.0

            progress_bar.set_postfix(loss=loss.item())

            scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {batch_loss / len(train_dataloader)}, LR: {scheduler.get_last_lr()}')

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
                wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

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

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Set multiprocessing to use 'spawn'

    parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    parser.add_argument('--tag', type=str, required=True, help='Tag for the training task.')
    parser.add_argument('--model', type=str, required=False, choices=['attn', 'resnet'], help='Name for the selection model')
    # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    # parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
    # parser.add_argument('--cuda', type=str, required=True, help='the No of cuda')
    parser.add_argument('--seed', type=int, required=True, help='training random seed')
    # parser.add_argument('--task_dir', type=str, default=None, required=False, help='specify a task')
    args = parser.parse_args()

    tag = args.tag
    name = 'all-tasks-in-one-predicate-progress'
    model = args.model
    lr = 1e-4
    num_epochs = 1000
    cuda = 0
    # seed = 999
    batch_size = 500

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
        ['name', 'model', 'lr',  'batch_size', 'num_epochs', 'cuda', 'seed', 'task_dir']
    )
    #
    # for i, task_dir in enumerate(sub_tasks):
    args = Args(name, model, lr, batch_size, num_epochs, cuda, args.seed, None)
    run_name = f"{tag}_all_task_model_{model}_lr_{lr}_bs_{batch_size}_epochs_{num_epochs}_seed_{args.seed}"
    wandb.init(project="value-model-for-all-single-tasks", entity="minchiuan", name=run_name, config={
        "system_metrics": True  # Enable system metrics logging
    })
    main(args)
