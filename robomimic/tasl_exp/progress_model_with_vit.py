import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor


class ProgressViTModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(CustomViTModel, self).__init__()
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


# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Create the dataset
root_dir = '/home/minquangao/robocasa/playground/counterToCab-robot-merged-image/'
dataset = CustomImageDataset(root_dir, feature_extractor)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Example of iterating over the dataloader
for i, (image1, image2, labels) in enumerate(dataloader):
    if i > 10: break
    print(image1.shape, image2.shape, labels)
    # Now you can use these batches for training