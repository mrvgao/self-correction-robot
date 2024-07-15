import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import models, transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor
import wandb
from tqdm import tqdm
from transformers import DetrModel
import argparse

# Argument Parsing
parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
parser.add_argument('--name', type=str, required=True, help='Name for the training task.')
parser.add_argument('--model', type=str, required=True, choices=['detr', 'vit', 'resnet'], help='Name for the selection model')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
args = parser.parse_args()

# set wandb monitor
run_name = f"{args.name}_model_{args.model}_lr{args.lr}_bs{args.batch_size}_epochs{args.num_epochs}"
wandb.init(project="value-model-via-vision-transformer-finetuning", entity="minchiuan", name=run_name, config={
    "system_metrics": True  # Enable system metrics logging
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.image_pairs = []

        # Prepare a list of all image pairs and corresponding labels
        for task_dir in os.listdir(root_dir):
            task_path = os.path.join(root_dir, task_dir)
            if os.path.isdir(task_path):
                images = sorted([f for f in os.listdir(task_path) if f.endswith('.png')])
                # Get the numeric parts of the filenames
                image_numbers = sorted([int(os.path.splitext(f)[0]) for f in images])
                M = image_numbers[0]
                N = image_numbers[-1]

                for i in range(len(image_numbers)):
                    image1_path = os.path.join(task_path, f'{M}.png')
                    image2_path = os.path.join(task_path, f'{image_numbers[i]}.png')

                    # Calculate the label
                    label = -1 * (N - image_numbers[i]) + 1
                    self.image_pairs.append((image1_path, image2_path, label))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path, label = self.image_pairs[idx]

        # Load the images
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        # Preprocess the images
        if args.model != 'resnet':
            image1 = self.feature_extractor(images=image1, return_tensors="pt")['pixel_values'].squeeze()
            image2 = self.feature_extractor(images=image2, return_tensors="pt")['pixel_values'].squeeze()
        else:
            image1 = self.feature_extractor(image1)
            image2 = self.feature_extractor(image2)


        return image1, image2, torch.tensor(label, dtype=torch.float32)


class ValueDetrModel(nn.Module):
    def __init__(self, pretrained_model_name='facebook/detr-resnet-50'):
        super(ValueDetrModel, self).__init__()
        self.detr = DetrModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.detr.config.hidden_size * 2, 1)  # concatenate features from two images and map to a single value

    def forward(self, image1, image2):
        outputs1 = self.detr(image1).last_hidden_state[:, 0, :]  # Using [CLS] token representation
        outputs2 = self.detr(image2).last_hidden_state[:, 0, :]
        concatenated = torch.cat((outputs1, outputs2), dim=1)
        x = self.fc(concatenated)
        return x


class ValueViTModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ValueViTModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.vit.config.hidden_size * 2,
                            1)  # concatenate features from two images and map to a single value

    def forward(self, image1, image2):
        outputs1 = self.vit(image1).pooler_output
        outputs2 = self.vit(image2).pooler_output
        concatenated = torch.cat((outputs1, outputs2), dim=1)
        x = self.fc(concatenated)
        return x

class ValueResNetModel(nn.Module):
    def __init__(self, pretrained_model_name='resnet50'):
        super(ValueResNetModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # Define additional fully connected layers with dropout
        self.fc1 = nn.Linear(self.resnet_fc_in_features * 2, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, image1, image2):
        outputs1 = self.resnet(image1)
        outputs2 = self.resnet(image2)
        concatenated = torch.cat((outputs1, outputs2), dim=1)

        # Pass through fully connected layers with dropout
        x = self.fc1(concatenated)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def main():
    # Initialize the feature extractor

    resnet_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Create the dataset
    root_dir = '/home/minquangao/robocasa/playground/counterToCab-robot-merged-image/'
    dataset = CustomImageDataset(root_dir, feature_extractor if args.model != 'resnet' else resnet_transformer)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the dataloader
    # Example training loop
    num_epochs = args.num_epochs
    # model = ProgressViTModel().to(device)
    if args.model == 'detr':
        model = ValueDetrModel().to(device)
    elif args.model == 'vit':
        model = ValueViTModel().to(device)
    elif args.model == 'resnet':
        model = ValueResNetModel().to(device)
    else:
        raise ValueError("unsupported model name, ", args.model)

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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)
        batch_loss = 0

        for i, (image1, image2, labels) in enumerate(progress_bar):
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(image1, image2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": avg_loss})
                running_loss = 0.0

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {batch_loss / len(train_dataloader):.4f}")

        # Save the model
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=True)
                for image1, image2, labels in progress_bar:
                    # Move data to GPU
                    image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)

                    outputs = model(image1, image2)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()


                avg_val_loss = val_loss / len(val_dataloader)
                scheduler.step(avg_val_loss)
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

    wandb.finish()


if __name__ == "__main__":
    main()
