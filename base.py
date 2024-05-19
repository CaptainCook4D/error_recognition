import csv
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import Constants as const
from core.models.blocks import MLP
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

db_service = FirebaseService()


def fetch_input_dim(config):
    if config.backbone == const.OMNIVORE:
        return 1024


def fetch_model(config):
    model = None
    if config.variant == const.MLP_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST]:
            input_dim = fetch_input_dim(config)
            model = MLP(input_dim, 512, 1)

    assert model is not None, f"Model not found for variant: {config.variant} and backbone: {config.backbone}"

    model.to(config.device)

    return model


def collate_stats(sub_step_metrics, step_metrics):
    collated_stats = []
    for metric in [const.PRECISION, const.RECALL, const.F1, const.AUC]:
        collated_stats.append(sub_step_metrics[metric])
    for metric in [const.PRECISION, const.RECALL, const.F1, const.AUC]:
        collated_stats.append(step_metrics[metric])

    return collated_stats


def save_results_to_csv(config, sub_step_metrics, step_metrics):
    results_dir = os.path.join(os.getcwd(), const.RESULTS)
    task_results_dir = os.path.join(results_dir, config.task_name)
    os.makedirs(task_results_dir, exist_ok=True)

    results_file_path = os.path.join(task_results_dir, f'{config.model_name}.csv')
    collated_stats = collate_stats(sub_step_metrics, step_metrics)

    file_exist = os.path.isfile(results_file_path)

    with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
        writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
        if not file_exist:
            writer.writerow([
                "Task Name", "Variant", "Backbone", "Model Name",
                "Sub-Step Precision", "Sub-Step Recall", "Sub-Step F1", "Sub-Step AUC",
                "Step Precision", "Step Recall", "Step F1", "Step AUC"
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
        model_name=config.model_name,
    )

    result.add_result_details(result_details)
    db_service.update_result(result.result_id, result.to_dict())


def save_results(config, sub_step_metrics, step_metrics):
    # 1. Save evaluation results to csv
    save_results_to_csv(config, sub_step_metrics, step_metrics)
    # 2. Save evaluation results to firebase
    save_results_to_firebase(config, sub_step_metrics, step_metrics)


def store_model(model, config, ckpt_name: str):
    task_directory = os.path.join(config.ckpt_directory, config.task_name)
    os.makedirs(task_directory, exist_ok=True)

    ckpt_file_path = os.path.join(task_directory, ckpt_name)
    torch.save(model.state_dict(), ckpt_file_path)


# ----------------------- TRAIN BASE FILES -----------------------


# ----------------------- TEST BASE FILES -----------------------


def test_er_model(model, test_loader, criterion, device, phase):
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

    # prob_range = np.max(all_outputs) - np.min(all_outputs)
    # all_outputs = (all_outputs - np.min(all_outputs)) / prob_range

    # ------------------------- Sub-Step Level Metrics -------------------------
    all_sub_step_targets = all_targets.copy()
    all_sub_step_outputs = all_outputs.copy()

    # Calculate metrics at the sub-step level
    pred_sub_step_labels = (all_sub_step_outputs > 0.5).astype(int)
    sub_step_precision = precision_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_recall = recall_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_f1 = f1_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_auc = roc_auc_score(all_sub_step_targets, all_sub_step_outputs)
    sub_step_pr_auc = binary_auprc(torch.tensor(pred_sub_step_labels), torch.tensor(all_sub_step_targets))

    sub_step_metrics = {
        const.PRECISION: sub_step_precision,
        const.RECALL: sub_step_recall,
        const.F1: sub_step_f1,
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
        # # Top 10% of the predictions - 90th percentile
        # threshold = np.percentile(sorted_step_output, 90)
        # step_output = step_output[step_output > threshold]
        mean_step_output = np.mean(step_output)
        step_target = 1 if np.mean(step_target) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)

    all_step_outputs = np.array(all_step_outputs)
    all_step_targets = np.array(all_step_targets)

    # Calculate metrics at the step level
    pred_step_labels = (all_step_outputs > 0.5).astype(int)
    precision = precision_score(all_step_targets, pred_step_labels)
    recall = recall_score(all_step_targets, pred_step_labels)
    f1 = f1_score(all_step_targets, pred_step_labels)

    auc = roc_auc_score(all_step_targets, all_step_outputs)
    pr_auc = binary_auprc(torch.tensor(pred_step_labels), torch.tensor(all_step_targets))

    step_metrics = {
        const.PRECISION: precision,
        const.RECALL: recall,
        const.F1: f1,
        const.AUC: auc,
        const.PR_AUC: pr_auc
    }

    # Print step level metrics
    print("----------------------------------------------------------------")
    print(f'Sub Step Level Metrics: {sub_step_metrics}')
    print(f"Step Level Metrics: {step_metrics}")

    return test_losses, sub_step_metrics, step_metrics
