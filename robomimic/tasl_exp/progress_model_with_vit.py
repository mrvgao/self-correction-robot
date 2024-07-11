import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor
import wandb
from tqdm import tqdm

wandb.init(project="progress-model-via-vision-transformer-finetuning", entity="minchiuan", config={
    "system_metrics": True  # Enable system metrics logging
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ProgressViTModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ProgressViTModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.vit.config.hidden_size * 2,
                            1)  # concatenate features from two images and map to a single value
        self.sigmoid = nn.Sigmoid()

    def forward(self, image1, image2):
        outputs1 = self.vit(image1).pooler_output
        outputs2 = self.vit(image2).pooler_output
        concatenated = torch.cat((outputs1, outputs2), dim=1)
        x = self.fc(concatenated)
        x = self.sigmoid(x)
        return x


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
                image_numbers = sorted([int(os.path.splitext(f)[0]) for f in images])
                M = image_numbers[0]
                N = image_numbers[-1]

                for i in range(0, len(images)):
                    image1_path = os.path.join(task_path, f'{M}.png')
                    image2_path = os.path.join(task_path, f'{image_numbers[i]}.png')
                    label = (image_numbers[i] - M) / (len(images) - 1)
                    self.image_pairs.append((image1_path, image2_path, label))

        print('finish initialized')

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path, label = self.image_pairs[idx]

        # Load the images
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        # Preprocess the images
        image1 = self.feature_extractor(images=image1, return_tensors="pt")['pixel_values'].squeeze()
        image2 = self.feature_extractor(images=image2, return_tensors="pt")['pixel_values'].squeeze()

        return image1, image2, torch.tensor(label, dtype=torch.float32)


def main():
    # Initialize the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Create the dataset
    root_dir = '/home/minquangao/robocasa/playground/counterToCab-robot-merged-image/'
    dataset = CustomImageDataset(root_dir, feature_extractor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create the dataloader
    # Example training loop
    num_epochs = 100
    model = ProgressViTModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)
        for i, (image1, image2, labels) in enumerate(progress_bar):
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(image1, image2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": avg_loss})
                running_loss = 0.0

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")

        # Save the model
        torch.save(model.state_dict(), f'progress_vit_model_epoch_{epoch}.pth')

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
                wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
