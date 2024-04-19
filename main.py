from __future__ import print_function

import argparse
import os
import os.path
import time
from pprint import pprint
import time
import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import pickle
from tqdm import tqdm
import sys
import wandb
from loguru import logger
from scripts import train, test
from dataloader import ErrorDataset, collate_files_without_frames
from utils import init_logger_and_wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import NN, CNN, NN1


def is_identity_layer(layer):
    # Get the weight and bias from the layer
    weight = layer.weight.data
    bias = layer.bias.data

    # Check if the weight is an identity matrix
    identity_weight = torch.eye(weight.size(0), weight.size(1))
    if weight.device != torch.device("cpu"):
        identity_weight = identity_weight.to(weight.device)

    # Check if weights form an identity matrix and biases are zero
    is_identity = torch.allclose(weight, identity_weight, atol=1e-6) and torch.all(
        bias == 0
    )

    return is_identity


def train_runner(args):
    debug = args.debug
    if debug:
        args.epochs = 2
    project_name = f"SL_Error_Detection_{args.backbone}"
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Modifying model path based on args.no_train

    if not args.no_train:
        if args.only_main_errors:
            args.model_path = os.path.join(
                "trained_models/checkpoints_only_main_errors/",
                date_time,
                str(args.backbone),
            )
        else:
            args.model_path = os.path.join(
                "trained_models/checkpoints/",
                date_time,
                str(args.backbone),
            )
            if args.one_vs_all:
                logger.info("Running one vs all")
                logger.info(f"Error type: {args.error_type}")
                args.model_path = os.path.join(
                    "trained_models/checkpoints_one_vs_all/",
                    date_time,
                    str(args.error_type),
                    str(args.backbone),
                )

    else:
        args.model_path = os.path.join(
            "trained_models/checkpoints/",
            date_time,
        )

    if args.debug:
        output_dir = os.path.join("debug", "trained_models", project_name)
    else:
        output_dir = args.model_path
    model_outputs_dir = output_dir.replace("models", "model_outputs")
    args.model_outputs_dir = model_outputs_dir
    if os.path.exists(model_outputs_dir):
        logger.info(f"Model outputs directory already exists")
    else:
        os.makedirs(model_outputs_dir)
    device, use_cuda, use_mps = init_logger_and_wandb(project_name, args)
    args.device = device
    # base_folder_path = "/home/akshay/work"
    base_folder_path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks"
    if args.only_main_errors:
        labels_dir = (
            base_folder_path
            + "/Supervised-Benchmarks-for-Error-Dataset/sl_labels_updated/sl_labels_for_one_sec_chunks_only_main_errors"
        )
    else:
        labels_dir = (
            base_folder_path
            + "/Supervised-Benchmarks-for-Error-Dataset/sl_labels_updated/labels_for_one_sec_chunks"
        )
    features_dir = f"/data/error_detection/dataset/features/{args.backbone}"
    # features_dir = f"/bigdata/akshay/datacollection/error_detection/dataset/features/{args.backbone}"
    train_dataset = ErrorDataset(
        root_dir=features_dir,
        labels_dir=labels_dir,
        split="train",
        args=args,
    )
    val_dataset = ErrorDataset(
        root_dir=features_dir,
        labels_dir=labels_dir,
        split="val",
        args=args,
    )
    test_dataset = ErrorDataset(
        root_dir=features_dir,
        labels_dir=labels_dir,
        split="test",
        args=args,
    )
    if args.backbone == "video_mae":
        input_size = train_dataset[0][0].shape[0]
    else:
        input_size = train_dataset[0][0].shape[1]
    torch.manual_seed(args.seed)

    # Define the output directory
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            "num_workers": 8,
            "pin_memory": True,
        }
        train_kwargs = {**train_kwargs, **cuda_kwargs, "shuffle": True}
        test_kwargs = {**test_kwargs, **cuda_kwargs, "shuffle": False}
    logger.info(f"You have selected {args.backbone}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_files_without_frames, **train_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collate_files_without_frames, **test_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=collate_files_without_frames, **test_kwargs
    )
    # Initialize the model model for training
    if args.no_train:
        model = torch.load(args.model_path)
    elif args.backbone in ["omnivore", "video_mae", "3dresnet"]:
        model = NN(input_size, 512, 1)
    elif args.backbone in ["slowfast_1", "x3d"]:
        if args.backbone == "3dresnet":
            in_channels, final_width, final_height = 2048, 64, 8
        model = CNN(in_channels, final_width, final_height, 1)
    elif args.backbone == "slowfast":
        model = NN(input_size, 512, 1)
    elif args.backbone == "x3d_pca_nc64":
        input_size = 12288
        model = NN1(input_size, 512, 1)
    # is_identity = is_identity_layer(model.layer2)
    # print("Is layer2 an identity layer?", is_identity)
    model.cuda()
    logger.info(f"Model: {model}")
    try:
        checkpoint_epoch = args.epochs // 10
    except ZeroDivisionError:
        checkpoint_epoch = 1
    # model = torch.nn.DataParallel(model)
    if not args.no_train:
        # Define the optimizer
        epoch = args.epochs
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        # Define the learning rate scheduler
        # Train the model
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, epoch, args.epochs)
            print("Learning rate:", optimizer.param_groups[0]["lr"])
            val_loss, _, _, _, _ = test(model, val_loader, device)
            # Log the train and val loss
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            logger.info("+----------------------------------+")
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Train loss: {train_loss}")
            logger.info(f"Val loss: {val_loss}")
            # Save checkpoint every 5 epochs
            try:
                checkpoint_model = epoch % checkpoint_epoch == 0
            except ZeroDivisionError:
                continue
            if checkpoint_model:
                checkpoint_dir = os.path.join(
                    f"{output_dir}/checkpoints/",
                    date_time,
                )
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

        if not args.not_save_model and not args.no_train:
            if not os.path.exists(output_dir):
                # If the folder does not exist, create it
                os.makedirs(output_dir)
                print("Folder created successfully!")
            else:
                print("Folder already exists!")
            torch.save(model.state_dict(), f"{output_dir}/{args.backbone}.pt")
        (
            total_loss,
            all_outputs,
            true_labels,
            all_indices_for_split,
            all_recording_ids,
        ) = test(model, test_loader, device)
        logger.info(f"Total test loss: {total_loss}")
        logger.info("Metrics for one second segments")

        probabilities = torch.sigmoid(torch.tensor(all_outputs)).numpy()
        # Apply threshold for classification
        threshold = 0.5
        predictions = (probabilities > threshold).astype(int)
        true_labels = true_labels.astype(int)
        predictions = predictions.reshape(-1)
        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1: {f1}")
        wandb.log(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        np.savez(
            model_outputs_dir + "/test_outputs.npz",
            predictions=predictions,
            probabilities=probabilities,
            true_labels=true_labels,
            all_indices_for_split=all_indices_for_split,
            all_recording_ids=all_recording_ids,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="SS-CMPE Experiments")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Are we in debug mode?"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16384,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        default=False,
        help="disables training",
    )
    parser.add_argument(
        "--one-vs-all",
        action="store_true",
        default=False,
        help="Do you want to run one error type vs all?",
    )
    parser.add_argument(
        "--error-type",
        type=str,
        choices=[
            "Technique Error",
            "Preparation Error",
            "Measurement Error",
            "Temperature Error",
            "Timing Error",
        ],
        help="Which error type to use for one vs all?",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        help="Provide the model location for evaluation",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--not-save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--only-main-errors",
        action="store_true",
        default=False,
        help="Only use main errors as error labels",
    )
    # Add the argument for the number of layers
    parser.add_argument(
        "--layers", type=int, nargs="+", default=3, help="Number of layers"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="video_mae",
        choices=[
            "omnivore",
            "slowfast",
            "3dresnet",
            "x3d",
            "slowfast_1",
            "x3d_1",
            "x3d_pca",
            "x3d_pca_nc64",
            "video_mae",
        ],
        help="Which model to use",
    )

    args = parser.parse_args()
    pprint(args)
    if args.one_vs_all:
        assert (
            args.error_type is not None
        ), "Please provide the error type if you want to run one vs all"
    train_runner(args)
