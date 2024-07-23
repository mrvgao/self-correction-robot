import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTFeatureExtractor, ViTModel
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
from transformers import CLIPModel, CLIPProcessor

# Argument Parsing
parser = argparse.ArgumentParser(description='Train a Progress Predication Model Via CLIP and MSE.')
parser.add_argument('--name', type=str, required=True, help='Name for the training task.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model.")
args = parser.parse_args()

# set wandb monitor
run_name = f"{args.name}_{args.lr}_bs{args.batch_size}_epochs{args.num_epochs}"
wandb.init(project="progress-model-via-CLIP-MSE", entity="minchiuan", name=run_name, config={
    "system_metrics": True  # Enable system metrics logging
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, processor, sentence):
        self.root_dir = root_dir
        self.processor = processor
        self.image_pairs = []
        self.sentence = sentence

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
                    self.image_pairs.append((image2_path, label))

        print('finish initialized')

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image2_path, label = self.image_pairs[idx]

        # Load the images
        # image1 = Image.open(image1_path).convert('RGB')
        image = Image.open(image2_path).convert('RGB')

        # Preprocess the images and text
        inputs = self.processor(text=self.sentence, images=image, return_tensors="pt", padding=True)
        image = inputs['pixel_values'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)


class CLIPCombinedModel(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14'):
        super(CLIPCombinedModel, self).__init__()

        # Load CLIP model
        print('loading clip model')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        # Load pre-trained ResNet model
        print('loading resnet model')
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.fc = nn.Identity()  # Remove the final classification layer

        # Define projection layers to match dimensions
        self.image_proj = nn.Linear(self.clip_model.config.projection_dim + 2048, 768)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, input_ids, attention_mask):
        # Process the image through ResNet
        resnet_features = self.resnet_model(image)

        # Process the image and sentence through CLIP
        vision_outputs = self.clip_model.get_image_features(image)
        text_outputs = self.clip_model.get_text_features(input_ids, attention_mask)

        # Concatenate ResNet features and CLIP vision features
        combined_features = torch.cat((vision_outputs, resnet_features), dim=1)

        # Project combined features to match text feature dimensions
        projected_features = self.image_proj(combined_features)

        # Get similarity scores with text features
        similarity_score = (projected_features * text_outputs).sum(dim=1, keepdim=True)

        # Pass through the fully connected layer
        # output = self.fc(similarity_score)

        # Apply sigmoid activation to constrain output between 0 and 1
        output = self.sigmoid(similarity_score)

        return output.squeeze()


def main():
    # Initialize the feature extractor
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    task_descprtion = """the input two images are observed from a robot in its initial state and its current state. 
    For each image, the left one is the robot's left eye's observatation, the middle one is the robot's hand eye's observations, 
    the right one is the robot's right eye's observations. This robot needs to complete its task as the following requiremnet: 1. 
    One obejct from initial observation image should be put in the cabinet; 2. The object's background is different with the initial image's counter's background;
    3. The robot's hands are higher than counter and higher than cabinet; 
    """
    task_descrpition = "this robot has completed the task: pick up an object from counter and put it on the cabinet"

    # Create the dataset
    root_dir = '/home/minquangao/robocasa/playground/counterToCab-robot-merged-image/'

    dataset = CustomImageDataset(root_dir, processor, task_descrpition)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the dataloader
    # Example training loop
    num_epochs = args.num_epochs

    model = CLIPCombinedModel().to(device)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Loaded pretrained model from {args.model_path}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    task_name = wandb.run.name
    wandb.watch(model)

    save_dir = f'models/{task_name}'
    os.makedirs(save_dir, exist_ok=True)

    early_stopping_patience = 3
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)
        for i, (image, input_ids, attention_mask, label) in enumerate(progress_bar):
            image,label = image.to(device), label.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(image, input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            running_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": avg_loss})
                running_loss = 0.0

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(train_dataloader):.4f}")

        # Save the model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=True)
                for image, input_ids, attention_mask, label in progress_bar:
                    # Move data to GPU
                    image, input_ids, label = image.to(device),input_ids.to(device), label.to(device)
                    attention_mask = attention_mask.to(device)

                    outputs = model(image, input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), label)
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

    wandb.finish()


if __name__ == '__main__':
    main()
