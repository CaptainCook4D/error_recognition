import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class NN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size * 8)
        self.layer2 = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, in_channels, final_width, final_height, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_width * final_height, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
