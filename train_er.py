"""
Script for training the ERROR RECOGNITION model
"""
from torch import nn

from core.config import Config
import torch
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset
from torch.utils.data import DataLoader
from test_er import test_er
from core.models.blocks import MLP
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

"""
Data type: ["Segments"]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore, Imagebind]
Modality: [Video]
Model: [ErMLP, ErCNN]

Output: 4 models

Data type: [Steps]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore]
Modality: [Video]
Model: [ErFormer]

Output: 5 models

Data type: [Steps]
Backbones: [Imagebind]
Modality: [Video, Audio, Text, Depth]
Model: [ErMMFormer]

Output: 4 models
"""


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    # Wrap the train_loader with tqdm for a progress bar
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Set the description of the tqdm instance to show the loss
        train_loader.set_description(f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')


def train_er(config):
    torch.manual_seed(config.seed)
    device = config.device

    train_dataset = CaptainCookStepDataset(config, 'train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dataset = CaptainCookStepDataset(config, 'val')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)

    d_model = None
    for data, _ in train_loader:
        d_model = data.shape[2]
        break

    assert d_model is not None, "Data not found in the dataset"

    model = MLP(d_model, d_model//2, 1).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.num_epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch, criterion)
        test_er(model, device, test_loader, 'val', criterion)
        scheduler.step()
        if config.save_model:
            torch.save(model.state_dict(), f"{config.backbone}_{epoch}_MLP.pt")


if __name__ == "__main__":
    conf = Config()
    train_er(conf)