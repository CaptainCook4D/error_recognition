import os

import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader
from base import test_er_model, store_model, fetch_model, fetch_model_name, train_epoch
from constants import Constants as const
from core.config import Config
from core.utils import init_logger_and_wandb
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset
from dataloader.CaptainCookStepDataset import collate_fn
from dataloader.CaptainCookSubStepDataset import CaptainCookSubStepDataset


def train_er_model(train_loader, val_loader, device, config, test_loader=None):
    model = fetch_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], dtype=torch.float32).to(device))
    # criterion = nn.BCEWithLogitsLoss()
    # Initialize variables to track the best model based on the desired metric (e.g., AUC)
    best_model = {'model_state': None, 'metric': 0}

    model_name = config.model_name
    if config.model_name is None:
        model_name = fetch_model_name(config)
        config.model_name = model_name

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

            if test_loader is not None:
                test_losses, test_sub_step_metrics, test_step_metrics = test_er_model(model, test_loader, criterion,
                                                                                      device,
                                                                                      phase='test')

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            precision = step_metrics['precision']
            recall = step_metrics['recall']
            f1 = step_metrics['f1']
            auc = step_metrics['auc']

            # Write losses and metrics to file
            f.write(
                f'{epoch}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {avg_test_loss:.6f}, {precision:.6f}, {recall:.6f}, {f1:.6f}, {auc:.6f}\n')

            running_metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "val_loss": avg_val_loss,
                "val_metrics": {
                    "step_metrics": step_metrics,
                    "sub_step_metrics": sub_step_metrics
                },
                "test_metrics": {
                    "step_metrics": test_step_metrics,
                    "sub_step_metrics": test_sub_step_metrics
                }
            }

            wandb.log(running_metrics)

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, '
                  f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}')

            # Update best model based on the chosen metric, here using AUC as an example
            if auc > best_model['metric']:
                best_model['metric'] = auc
                best_model['model_state'] = model.state_dict()

            store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")


def train_sub_step_test_step_er(config):
    torch.manual_seed(config.seed)
    device = config.device

    cuda_kwargs = {
        "num_workers": 1,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": 1024}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    train_dataset = CaptainCookSubStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)

    print("-------------------------------------------------------------")
    print("Training sub-step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(f"Split: {config.split}")
    print("-------------------------------------------------------------")

    train_er_model(train_loader, val_loader, device, config)


def train_step_test_step_er(config):
    torch.manual_seed(config.seed)
    device = config.device

    cuda_kwargs = {
        "num_workers": 8,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": 1}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    print("-------------------------------------------------------------")
    print("Training step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(config.args)
    print("-------------------------------------------------------------")

    train_dataset = CaptainCookStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.VAL, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    train_er_model(train_loader, val_loader, device, config, test_loader=test_loader)


if __name__ == "__main__":
    conf = Config()
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name
    init_logger_and_wandb(conf)
    train_step_test_step_er(conf)
    wandb.finish()
