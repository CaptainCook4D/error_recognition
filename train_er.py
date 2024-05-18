import numpy as np
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.models.blocks import MLP
from core.config import Config
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset
from test_er import test_er


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # chunk_losses = []
        # for i in range(0, data.shape[1], 10):
        #     data_chunk = data[:, i:i + 10, :]
        #     target_chunk = target[:, i:i + 10, :]

        # Send to the model and calculate the loss
        optimizer.zero_grad()
        output_chunk = model(data)
        loss = criterion(output_chunk, target)
        # chunk_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        train_loader.set_description(
            f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
        )

    return train_losses


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

    model = MLP(d_model, d_model // 2, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize variables to track the best model based on the desired metric (e.g., AUC)
    best_model = {'model_state': None, 'metric': 0}

    # Open a file to store the losses and metrics
    with open('training_performance.txt', 'w') as f:
        f.write('Epoch, Train Loss, Test Loss, Precision, Recall, F1, AUC\n')
        for epoch in range(1, config.num_epochs + 1):
            train_losses = train_epoch(model, device, train_loader, optimizer, epoch, criterion)
            test_losses, precision, recall, f1, auc = test_er(model, device, test_loader, 'val', criterion)
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            # Write losses and metrics to file
            f.write(
                f'{epoch}, {avg_train_loss:.6f}, {avg_test_loss:.6f}, {precision:.6f}, {recall:.6f}, {f1:.6f}, {auc:.6f}\n')

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, '
                  f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}')

            # Update best model based on the chosen metric, here using AUC as an example
            if auc > best_model['metric']:
                best_model['metric'] = auc
                best_model['model_state'] = model.state_dict()

            if epoch % 10 == 0 and config.save_model:
                torch.save(model.state_dict(), f"{config.backbone}_{epoch}_MLP.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            torch.save(best_model['model_state'], f"{config.backbone}_best_MLP.pt")


if __name__ == "__main__":
    conf = Config()
    train_er(conf)
