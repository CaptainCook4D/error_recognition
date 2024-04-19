from argparse import ArgumentParser
import os
import random
import time

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# from natsort import natsorted
# from accimage import Image as AccImage

import sys


def divide_intervals(input_df, frames_per_segment=30):
    """
    The function `divide_intervals` takes an input DataFrame and divides each row into smaller intervals
    based on a specified time duration.

    Args:
      input_df: The input_df is a pandas DataFrame that contains information about intervals. Each row
    represents an interval and has the following columns:
      frames_per_segment: The `frames_per_segment` parameter specifies the duration of each sub-interval in
    # of frames. It determines how long each segment should be when dividing the intervals in the input
    dataframe. Defaults to 30

    Returns:
      a new DataFrame called `sub_interval_df` which contains the sub-intervals of the input DataFrame
    `input_df`.
    """
    new_rows = []

    for index, row in input_df.iterrows():
        duration = row["end_frame"] - row["start_frame"]
        num_sub_intervals = duration // frames_per_segment

        for i in range(num_sub_intervals):
            start_sub = row["start_frame"] + i
            end_sub = start_sub + frames_per_segment

            new_rows.append(
                {
                    "index": i,
                    "start_frame": start_sub,
                    "end_frame": end_sub,
                    "recording_id": row["recording_id"],
                    "recipe_id": row["recipe_id"],
                    "start_timestamp": row["start_timestamp"],
                    "end_time": row["end_time"],
                    "step_description": row["step_description"],
                    "error_label": row["error_label"],
                    "modified_step_descriptions": row["modified_step_descriptions"],
                    "error_tag": row["error_tag"],
                    "error_description": row["error_description"],
                }
            )

    sub_interval_df = pd.DataFrame(new_rows)
    return sub_interval_df


class ErrorDataset(Dataset):
    """Error dataset for Supervised Error Detection.
    Args:
        root_dir (string): Directory with videos.
        labels_dir (string): Directory with splits.
        split (str): train, valid or test split.
        clip_len (int): number of frames in clip, 30/60/120.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """

    def __init__(
        self,
        root_dir,
        labels_dir,
        args,
        clip_len=30,
        split="train",
        transforms_=None,
        test_sample_num=1,
        fps=30,
        method="omnivore",
    ):
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.clip_len = clip_len
        self.split = split
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        assert os.path.exists(self.root_dir), "root_dir path does not exist!"
        assert split in ["train", "val", "test"], "split must be train, val or test!"
        split_path = os.path.join(labels_dir, self.split + ".csv")
        split_file = pd.read_csv(split_path, sep=",")
        logger.info("Using split " + self.split)
        self.fps = fps
        self.recording_ids = split_file["recording_id"]
        self.recipe_ids = split_file["recipe_id"]
        self.start_frames = split_file["start_frame"]
        self.end_frames = split_file["end_frame"]
        self.indices_for_splits = split_file["index"]
        self.labels = torch.tensor(split_file["error_label"].values).float()
        if args.one_vs_all:
            self.labels = torch.tensor(split_file[args.error_type].values).float()
            logger.success(f"Loading labels for {args.error_type}")
        self.method = method

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        videoname = self.recipe_ids[idx]
        label = self.labels[idx]
        start_time = self.start_frames[idx] // self.fps
        end_time = start_time + self.fps // self.clip_len
        index_for_split = self.indices_for_splits[idx]
        recording_id = self.recording_ids[idx]
        path_to_features = os.path.join(
            self.root_dir,
            videoname + "_360p",
            videoname + "_360p_" + f"{float(start_time)}_{float(end_time)}.npy",
        )
        # Add this if some features are not available
        try:
            features = torch.from_numpy(np.load(path_to_features))
        except FileNotFoundError:
            return None
        if self.method == "3dresnet":
            features = features.reshape(1, 2048, 64, 8)
        return (
            features,
            label,
            start_time,
            end_time,
            index_for_split,
            recording_id,
        )


from torch.utils.data.dataloader import default_collate


def collate_files_without_frames(batch):
    new_batch = []
    # ids = []
    for _batch in batch:
        if _batch is None:
            continue
        new_batch.append(_batch)
        # ids.append(_batch[-1])
    return default_collate(new_batch)


if __name__ == "__main__":
    args = ArgumentParser()
    this_root_dir = "/data/error_detection/dataset/features/omnivore"
    this_labels_dir = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_labels"
    this_dataset = ErrorDataset(
        this_root_dir,
        this_labels_dir,
        split="train",
    )
    for i in range(len(this_dataset)):
        print(this_dataset[i][0].shape)
