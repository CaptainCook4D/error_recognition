import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from base import test_er_model, fetch_model, store_model
from constants import Constants as const
from core.config import Config
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset
from dataloader.CaptainCookStepDataset import collate_fn


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # positive_weight = target.detach().clone()
        # negative_weight = 1 - target.detach().clone()
        #
        # weight = 0.7 * positive_weight + 0.3 * negative_weight

        loss = criterion(output, target)
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

    train_dataset = CaptainCookStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, batch_size=10 * config.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CaptainCookStepDataset(config, const.VAL, config.split)
    val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size)

    model = fetch_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.8))

    # Initialize variables to track the best model based on the desired metric (e.g., AUC)
    best_model = {'model_state': None, 'metric': 0}

    model_name = config.model_name
    if config.model_name is None:
        model_name = f"{config.task_name}_{config.variant}_{config.backbone}"

    train_stats_directory = f"stats/{config.task_name}/{config.variant}/{config.backbone}"
    os.makedirs(train_stats_directory, exist_ok=True)
    train_stats_file = f"{model_name}_training_performance.txt"
    train_stats_file_path = os.path.join(train_stats_directory, train_stats_file)

    # Open a file to store the losses and metrics
    with open(train_stats_file_path, 'w') as f:
        f.write('Epoch, Train Loss, Test Loss, Precision, Recall, F1, AUC\n')
        for epoch in range(1, config.num_epochs + 1):
            train_losses = train_epoch(model, device, train_loader, optimizer, epoch, criterion)
            val_losses, sub_step_metrics, step_metrics = test_er_model(model, val_loader, criterion, device,
                                                                       phase='val')
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_test_loss = sum(val_losses) / len(val_losses)

            precision = step_metrics['precision']
            recall = step_metrics['recall']
            f1 = step_metrics['f1']
            auc = step_metrics['auc']

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
                store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")


if __name__ == "__main__":
    conf = Config()
    train_er(conf)
