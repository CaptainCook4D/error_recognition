"""
Evaluation script for the trained models.
Metrics used:  [Accuracy, Precision, Recall, F1-score]
Plots: [AUROC, Confusion Matrix, Calibration Curve]
"""
import torch
from torch.utils.data import DataLoader
from constants import Constants as const

from base import fetch_model, test_er_model, save_results
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset


def evaluate_er_models(config):
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    model = fetch_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the model from the ckpt file
    model.load_state_dict(torch.load(config.ckpt_path))
    model.eval()

    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)

    # Calculate the evaluation metrics
    test_losses, sub_step_metrics, step_metrics = test_er_model(model, test_loader, criterion, config.device,
                                                                phase='test')

    # Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    save_results(config, sub_step_metrics, step_metrics)


def evaluate_ecr_models():
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    pass
