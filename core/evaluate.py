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
from dataloader.CaptainCookStepDataset import collate_fn


def evaluate_er_models(config, step_normalization, sub_step_normalization, threshold):
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    model = fetch_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the model from the ckpt file
    model.load_state_dict(torch.load(config.ckpt_directory))
    model.eval()

    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn)

    # Calculate the evaluation metrics
    test_losses, sub_step_metrics, step_metrics = test_er_model(model, test_loader, criterion, config.device,
                                                                phase='test', step_normalization=step_normalization,
                                                                sub_step_normalization=sub_step_normalization,
                                                                threshold=threshold)

    # ----------------------PRINT METRICS----------------------
    print("Sub-Step Metrics: ", sub_step_metrics)
    print("Step Metrics: ", step_metrics)
    # ---------------------------------------------------------

    # Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    save_results(config, sub_step_metrics, step_metrics, step_normalization, sub_step_normalization, threshold)


def evaluate_ecr_models(config, step_normalization, sub_step_normalization, threshold):
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    pass


def evaluate_eer_models(config, step_normalization, sub_step_normalization, threshold):
    # 1. Load the trained model
    # 2. Load the test data
    # 3. Pass the test data through the model
    # 4. Calculate the evaluation metrics
    # 5. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    model = fetch_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the model from the ckpt file
    model.load_state_dict(torch.load(config.ckpt_directory))
    model.eval()

    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn)

    # Calculate the evaluation metrics
    test_losses, sub_step_metrics, step_metrics = test_er_model(model, test_loader, criterion, config.device,
                                                                phase='test', step_normalization=step_normalization,
                                                                sub_step_normalization=sub_step_normalization,
                                                                threshold=threshold)

    # ----------------------PRINT METRICS----------------------
    print("Sub-Step Metrics: ", sub_step_metrics)
    print("Step Metrics: ", step_metrics)
    # ---------------------------------------------------------

    # Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    save_results(config, sub_step_metrics, step_metrics, step_normalization, sub_step_normalization, threshold)


def main_er():
    val_best_epochs = [43, 9, 15, 25, 15, 40, 41, 3, 37, 28, 38, 20, 44, 9, 18, 5, 17, 40, 27, 22, 38, 31, 45, 30, 39,
                       34, 27, 25, 28, 4, 25, 7, 39, 7, 10, 13, 26, 22, 14, None, 26, 24, 7, 41, 40, 7, 12, 2, 37, 14,
                       50, 10, 46, None, 4, 19]
    test_best_epochs = [33, 31, 49, 25, 19, 8, 44, 34, 45, 39, 48, 8, 40, 48, 33, 31, 31, 49, 5, 43, 45, 25, 42, 2, 41,
                        27, 31, 5, 28, 4, 3, 20, 13, 6, 39, 51, 48, 8, 21, None, 19, 17, 5, 6, 7, 23, 2, 49, 11, 39, 11,
                        50, 31, None, 4, 8]
    best_epochs = [43, 9, 15, 25, 15, 40, 41, 3, 37, 28, 38, 20, 44, 9, 33, 31, 31, 49, 5, 43, 45, 25, 42, 2, 41, 27,
                   31, 5, 28, 4, 3, 20, 13, 6, 39, 50, 48, 8, 21, None, 19, 17, 5, 6, 7, 23, 2, 49, 11, 39, 11, 50, 31,
                   None, 4, 8]

    for step_normalization in [True]:
        for sub_step_normalization in [True]:
            for threshold in [0.5, 0.6]:
                epoch_index = 0
                for split in [const.STEP_SPLIT, const.RECORDINGS_SPLIT, const.PERSON_SPLIT, const.ENVIRONMENT_SPLIT]:
                    for backbone in [const.OMNIVORE, const.SLOWFAST, const.X3D, const.RESNET3D, const.IMAGEBIND]:
                        for variant in [const.MLP_VARIANT, const.TRANSFORMER_VARIANT]:
                            conf = Config()
                            conf.split = split
                            conf.backbone = backbone
                            conf.variant = variant
                            conf.phase = const.TEST
                            if backbone == const.IMAGEBIND:
                                for modality in [[const.VIDEO], [const.AUDIO], [const.VIDEO, const.AUDIO]]:
                                    if best_epochs[epoch_index] is None:
                                        epoch_index += 1
                                        continue
                                    conf.modality = modality
                                    modality = "_".join(modality)
                                    conf.split = split
                                    conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/error_recognition/{variant}/{backbone}/error_recognition_{variant}_{backbone}_{modality}_{split}_epoch_{best_epochs[epoch_index]}.pt"
                                    print(f"{variant}_{backbone}_{modality}_{split}_{best_epochs[epoch_index]}.pt")
                                    # conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/error_recognition/MLP/imagebind/error_recognition_MLP_imagebind_video_recordings_epoch_42.pt"
                                    evaluate_er_models(conf, step_normalization, sub_step_normalization, threshold)
                                    epoch_index += 1
                            else:
                                conf.modality = const.VIDEO
                                conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/error_recognition/{variant}/{backbone}/error_recognition_{variant}_{backbone}_{split}_epoch_{best_epochs[epoch_index]}.pt"
                                print(f"{variant}_{backbone}_{split}_{best_epochs[epoch_index]}.pt")
                                evaluate_er_models(conf, step_normalization, sub_step_normalization, threshold)
                                epoch_index += 1


def main_cr_er():
    best_epochs = [17, 24]
    for step_normalization in [True]:
        for sub_step_normalization in [True]:
            for threshold in [0.5, 0.6]:
                epoch_index = 0
                for split in [const.STEP_SPLIT, const.RECORDINGS_SPLIT]:
                    for backbone in [const.IMAGEBIND]:
                        for variant in [const.TRANSFORMER_VARIANT]:
                            conf = Config()
                            conf.split = split
                            conf.backbone = backbone
                            conf.variant = variant
                            conf.phase = const.TEST
                            if backbone == const.IMAGEBIND:
                                modality = [const.VIDEO, const.AUDIO, const.TEXT]
                                conf.modality = modality
                                modality = "_".join(modality)
                                conf.split = split
                                conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/error_recognition/{variant}/{backbone}/error_recognition_{split}_{backbone}_{variant}_{modality}_epoch_{best_epochs[epoch_index]}.pt"
                                print(f"{variant}_{backbone}_{modality}_{split}_{best_epochs[epoch_index]}.pt")
                                # conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/error_recognition/MLP/imagebind/error_recognition_MLP_imagebind_video_recordings_epoch_42.pt"
                                evaluate_er_models(conf, step_normalization, sub_step_normalization, threshold)
                                epoch_index += 1


def main_eer():
    best_epochs = [24, 12, 44, 33, 29, 5, 33, 3, 46, 12, 32, 20, 43, 7, 45, 37]
    for step_normalization in [True]:
        for sub_step_normalization in [True]:
            for threshold in [0.5, 0.6]:
                epoch_index = 0
                for split in [const.RECORDINGS_SPLIT, const.STEP_SPLIT]:
                    for backbone in [const.OMNIVORE, const.SLOWFAST, const.X3D, const.RESNET3D]:
                        for variant in [const.MLP_VARIANT, const.TRANSFORMER_VARIANT]:
                            conf = Config()
                            conf.task_name = const.EARLY_ERROR_RECOGNITION
                            conf.split = split
                            conf.backbone = backbone
                            conf.variant = variant
                            conf.phase = const.TEST

                            modality = conf.modality
                            conf.ckpt_directory = f"/data/rohith/captain_cook/checkpoints/{conf.task_name}/{variant}/{backbone}/early_error_recognition_{split}_{backbone}_{variant}_{modality}_epoch_{best_epochs[epoch_index]}.pt"
                            print(f"{variant}_{backbone}_{split}_{best_epochs[epoch_index]}.pt")
                            evaluate_er_models(conf, step_normalization, sub_step_normalization, threshold)
                            epoch_index += 1


if __name__ == '__main__':
    main_cr_er()
