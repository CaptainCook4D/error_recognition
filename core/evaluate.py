"""
Evaluation script for the trained models.
Metrics used:  [Accuracy, Precision, Recall, F1-score]
Plots: [AUROC, Confusion Matrix, Calibration Curve]
"""
import torch
from torch.utils.data import DataLoader
from constants import Constants as const
from core.config import Config
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

    # ----------------------PRINT METRICS----------------------
    print("Sub-Step Metrics: ", sub_step_metrics)
    print("Step Metrics: ", step_metrics)
    # ---------------------------------------------------------

    # Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    save_results(config, sub_step_metrics, step_metrics)


def evaluate_ecr_models():
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    pass


if __name__ == '__main__':
    val_best_epochs = [43, 9, 15, 25, 15, 40, 41, 3, 37, 28, 38, 20, 44, 9, 18, 5, 17, 40, 27, 22, 38, 31, 45, 30, 39, 34, 27, 25, 28, 4, 25, 7, 39, 7, 10, 13, 26, 22, 14, None, 26, 24, 7, 41, 40, 7, 12, 2, 37, 14, 50, 10, 46, None, 4, 19]
    test_best_epochs = [33, 31, 49, 25, 19, 8, 44, 34, 45, 39, 48, 8, 40, 48, 33, 31, 31, 49, 5, 43, 45, 25, 42, 2, 41, 27, 31, 5, 28, 4, 3, 20, 13, 6, 39, 51, 48, 8, 21, None, 19, 17, 5, 6, 7, 23, 2, 49, 11, 39, 11, 50, 31, None, 4, 8]
    i = 0
    for split in [const.STEP_SPLIT, const.RECORDINGS_SPLIT, const.ENVIRONMENT_SPLIT, const.PERSON_SPLIT]:
        for backbone in [const.OMNIVORE, const.SLOWFAST, const.X3D, const.RESNET3D, const.IMAGEBIND]:
            for variant in [const.MLP_VARIANT, const.TRANSFORMER_VARIANT]:
                config = Config()
                config.split = split
                config.backbone = backbone
                config.variant = variant
                for phase in [const.VAL, const.TEST]:
                    if phase == const.VAL:
                        best_epochs = val_best_epochs
                    else:
                        best_epochs = test_best_epochs
                    if backbone == const.IMAGEBIND:
                        for modality in [const.VIDEO, const.AUDIO, 'video_audio']:
                            config.modality = modality
                            config.ckpt_path = f"/data/rohith/captain_cook/checkpoints/error_recognition/{variant}/{backbone}/error_recognition_{variant}_{backbone}_{split}_best_epoch_{best_epochs[i]}.pt"
                            evaluate_er_models(config)
                    else:
                        config.modality = const.VIDEO
                        config.ckpt_path = f"/data/rohith/captain_cook/checkpoints/error_recognition/{variant}/{backbone}/error_recognition_{variant}_{backbone}_{split}_best_epoch_{best_epochs[i]}.pt
                        evaluate_er_models(config)
