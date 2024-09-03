import torch
import torch.nn as nn
from transformers import DetrModel
from transformers import ViTModel
from torchvision import models


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
    def __init__(self, text_embedding_dim=768):
        super(ValueResNetModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # Define additional fully connected layers with dropout

        self.fc1_double = nn.Linear(self.resnet_fc_in_features * 2, 512)
        # self.fc1_double = nn.Linear(2080, 512)
        self.fc1_single = nn.Linear(self.resnet_fc_in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, image1, image2):
        # Pass through fully connected layers with dropout
        if image1 is not None:
            outputs1 = self.resnet(image1)
            outputs2 = self.resnet(image2)
            concatenated = torch.cat((outputs1, outputs2), dim=1)
            # Pass through fully connected layers with dropout
            fc = self.fc1_double
        else:
            outputs2 = self.resnet(image2)
            concatenated = outputs2
            fc = self.fc1_single

        x = fc(concatenated)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ValueResNetModelWithText(nn.Module):
    def __init__(self, text_embedding_dim=768):
        super(ValueResNetModelWithText, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # Define a more complex text processing network
        text_out_dim = 128  # Increased output dimension for more capacity
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, text_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc1_double = nn.Linear(self.resnet_fc_in_features + text_out_dim, 512)
        # self.fc1_double = nn.Linear(2080, 512)
        self.fc1_single = nn.Linear(self.resnet_fc_in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 1)

        self.get_value = nn.Sequential(
            self.fc1_double,
            self.relu1,
            self.dropout1,
            self.fc2,
            self.relu2,
            self.dropout2,
            self.fc3
        )

    def forward(self, image, text_embedding):
        image_features = self.resnet(image)
        if text_embedding is not None:
            text_features = self.text_fc(text_embedding)
            concatenated = torch.cat((image_features, text_features), dim=1)
            fc = self.fc1_double
        else:
            concatenated = image_features
            fc = self.fc1_single
        x = fc(concatenated)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
