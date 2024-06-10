import csv
import os

import wandb
from torch import optim, nn
from torch.utils.data import DataLoader

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import Constants as const
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

from core.models.blocks import fetch_input_dim, MLP
from core.models.er_former import ErFormer
from dataloader.CaptainCookStepDataset import collate_fn, CaptainCookStepDataset
from dataloader.CaptainCookSubStepDataset import CaptainCookSubStepDataset

db_service = FirebaseService()


def fetch_model_name(config):
    if config.model_name is None:
        if config.backbone in [const.RESNET3D, const.X3D, const.SLOWFAST, const.OMNIVORE]:
            config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{config.modality}"
        elif config.backbone == const.IMAGEBIND:
            combined_modality_name = '_'.join(config.modality)
            config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{combined_modality_name}"

    return config.model_name


def fetch_model(config):
    model = None
    if config.variant == const.MLP_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            input_dim = fetch_input_dim(config)
            model = MLP(input_dim, 512, 1)
    elif config.variant == const.TRANSFORMER_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            model = ErFormer(config)

    assert model is not None, f"Model not found for variant: {config.variant} and backbone: {config.backbone}"
    model.to(config.device)
    return model


def convert_and_round(value):
    value = value * 100.0
    if isinstance(value, torch.Tensor):
        return np.round(value.numpy(), 2)
    return np.round(value, 2)


def collate_stats(config, sub_step_metrics, step_metrics):
    collated_stats = [config.split, config.backbone, config.variant, config.modality]
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        collated_stats.append(convert_and_round(sub_step_metrics[metric]))
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        # Round to two digits before appending
        collated_stats.append(convert_and_round(step_metrics[metric]))
    return collated_stats


def save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                        threshold=0.5):
    results_dir = os.path.join(os.getcwd(), const.RESULTS)
    task_results_dir = os.path.join(results_dir, config.task_name, "combined_results")
    os.makedirs(task_results_dir, exist_ok=True)
    config.model_name = fetch_model_name(config)

    results_file_path = os.path.join(task_results_dir,
                                     f'step-{step_normalization}_substep-{sub_step_normalization}_threshold-{threshold}.csv')
    collated_stats = collate_stats(config, sub_step_metrics, step_metrics)

    file_exist = os.path.isfile(results_file_path)

    with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
        writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
        if not file_exist:
            writer.writerow([
                "Split", "Backbone", "Variant", "Modality",
                "Sub-Step Precision", "Sub-Step Recall", "Sub-Step F1", "Sub-Step Accuracy", "Sub-Step AUC",
                "Sub-Step PR AUC",
                "Step Precision", "Step Recall", "Step F1", "Step Accuracy", "Step AUC", "Step PR AUC"
            ])
        writer.writerow(collated_stats)


def save_results_to_firebase(config, sub_step_metrics, step_metrics):
    sub_step_metrics = Metrics.from_dict(sub_step_metrics)
    step_metrics = Metrics.from_dict(step_metrics)
    result_details = ResultDetails(sub_step_metrics, step_metrics)
    result = Result(
        task_name=config.task_name,
        variant=config.variant,
        backbone=config.backbone,
        split=config.split,
        model_name=config.model_name,
        modality=config.modality
    )

    result.add_result_details(result_details)
    db_service.update_result(result.result_id, result.to_dict())


def save_results(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                 threshold=0.5):
    # 1. Save evaluation results to csv
    save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization, sub_step_normalization, threshold)
    # 2. Save evaluation results to firebase
    save_results_to_firebase(config, sub_step_metrics, step_metrics)


def store_model(model, config, ckpt_name: str):
    task_directory = os.path.join(config.ckpt_directory, config.task_name)
    os.makedirs(task_directory, exist_ok=True)

    variant_directory = os.path.join(task_directory, config.variant)
    os.makedirs(variant_directory, exist_ok=True)

    backbone_directory = os.path.join(variant_directory, config.backbone)
    os.makedirs(backbone_directory, exist_ok=True)

    ckpt_file_path = os.path.join(backbone_directory, ckpt_name)
    torch.save(model.state_dict(), ckpt_file_path)


# ----------------------- TRAIN BASE FILES -----------------------


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        assert not torch.isnan(loss).any(), "Loss contains NaN values"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

        train_losses.append(loss.item())
        train_loader.set_description(
            f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
        )
    optimizer.step()
    return train_losses


def train_model_base(train_loader, val_loader, config, test_loader=None):
    model = fetch_model(config)
    device = config.device
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


def train_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

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

    return train_loader, val_loader, test_loader


def train_sub_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

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
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    print("-------------------------------------------------------------")
    print("Training sub-step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(f"Split: {config.split}")
    print("-------------------------------------------------------------")

    return train_loader, val_loader, test_loader


# ----------------------- TEST BASE FILES -----------------------


def test_er_model(model, test_loader, criterion, device, phase, step_normalization=False, sub_step_normalization=False,
                  threshold=0.5):
    total_samples = 0
    all_targets = []
    all_outputs = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)
    test_losses = []

    test_step_start_end_list = []
    counter = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_samples += data.shape[0]
            loss = criterion(output, target)
            test_losses.append(loss.item())

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]

            # Set the description of the tqdm instance to show the loss
            test_loader.set_description(f'{phase} Progress: {total_samples}/{num_batches}')

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Assert that none of the outputs are NaN
    assert not np.isnan(all_outputs).any(), "Outputs contain NaN values"

    # ------------------------- Sub-Step Level Metrics -------------------------
    all_sub_step_targets = all_targets.copy()
    all_sub_step_outputs = all_outputs.copy()

    # Calculate metrics at the sub-step level
    pred_sub_step_labels = (all_sub_step_outputs > 0.5).astype(int)
    sub_step_precision = precision_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_recall = recall_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_f1 = f1_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_accuracy = accuracy_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_auc = roc_auc_score(all_sub_step_targets, all_sub_step_outputs)
    sub_step_pr_auc = binary_auprc(torch.tensor(pred_sub_step_labels), torch.tensor(all_sub_step_targets))

    sub_step_metrics = {
        const.PRECISION: sub_step_precision,
        const.RECALL: sub_step_recall,
        const.F1: sub_step_f1,
        const.ACCURACY: sub_step_accuracy,
        const.AUC: sub_step_auc,
        const.PR_AUC: sub_step_pr_auc
    }

    # -------------------------- Step Level Metrics --------------------------
    all_step_targets = []
    all_step_outputs = []

    # threshold_outputs = all_outputs / max_probability

    for start, end in test_step_start_end_list:
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]

        # sorted_step_output = np.sort(step_output)
        # # Top 50% of the predictions
        # threshold = np.percentile(sorted_step_output, 50)
        # step_output = step_output[step_output > threshold]

        # pos_output = step_output[step_output > 0.5]
        # neg_output = step_output[step_output <= 0.5]
        #
        # if len(pos_output) > len(neg_output):
        #     step_output = pos_output
        # else:
        #     step_output = neg_output
        step_output = np.array(step_output)
        # # Scale the output to [0, 1]
        if start - end > 1:
            if sub_step_normalization:
                prob_range = np.max(step_output) - np.min(step_output)
                step_output = (step_output - np.min(step_output)) / prob_range

        mean_step_output = np.mean(step_output)
        step_target = 1 if np.mean(step_target) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)

    all_step_outputs = np.array(all_step_outputs)

    # # Scale the output to [0, 1]
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range

    all_step_targets = np.array(all_step_targets)

    # Calculate metrics at the step level
    pred_step_labels = (all_step_outputs > threshold).astype(int)
    precision = precision_score(all_step_targets, pred_step_labels, zero_division=0)
    recall = recall_score(all_step_targets, pred_step_labels)
    f1 = f1_score(all_step_targets, pred_step_labels)
    accuracy = accuracy_score(all_step_targets, pred_step_labels)

    auc = roc_auc_score(all_step_targets, all_step_outputs)
    pr_auc = binary_auprc(torch.tensor(pred_step_labels), torch.tensor(all_step_targets))

    step_metrics = {
        const.PRECISION: precision,
        const.RECALL: recall,
        const.F1: f1,
        const.ACCURACY: accuracy,
        const.AUC: auc,
        const.PR_AUC: pr_auc
    }

    # Print step level metrics
    print("----------------------------------------------------------------")
    print(f'{phase} Sub Step Level Metrics: {sub_step_metrics}')
    print(f"{phase} Step Level Metrics: {step_metrics}")
    print("----------------------------------------------------------------")

    return test_losses, sub_step_metrics, step_metrics
