import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CaptainCookSubStepDataset(Dataset):

    def __init__(self, config, phase, split):
        self._config = config
        self._backbone = self._config.backbone
        self._phase = phase
        self._split = split

        if self._split is None:
            self._split = "recordings"

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"
        self._features_directory = self._config.video_features_directory

        self._recording_ids_file = f"{self._split}_data_split_combined.json"

        with open(f'../annotations/data_splits/{self._recording_ids_file}', 'r') as file:
            self._recording_ids_json = json.load(file)

        if self._phase == 'train':
            self._recording_ids = self._recording_ids_json['train'] + self._recording_ids_json['val']
        else:
            self._recording_ids = self._recording_ids_json[self._phase]

        with open('../annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        sub_step_id = 0
        self._sub_step_dict = {}
        for recording_id in self._recording_ids:
            for step in self._annotations[recording_id]['steps']:
                if step['start_time'] < 0 or step['end_time'] < 0:
                    # Ignore missing steps
                    continue

                start_time = math.floor(step['start_time'])
                end_time = math.floor(step['end_time'])

                for sub_step_time in range(start_time, end_time):
                    self._sub_step_dict[sub_step_id] = (
                    recording_id, (sub_step_time, sub_step_time + 1), step['has_errors'])
                    sub_step_id += 1

    def __len__(self):
        assert len(self._sub_step_dict) > 0, "No data found in the dataset"
        return len(self._sub_step_dict)

    def __getitem__(self, idx):
        recording_id = self._sub_step_dict[idx][0]
        start_time, end_time = self._sub_step_dict[idx][1]
        has_errors = self._sub_step_dict[idx][2]

        features_path = os.path.join(self._features_directory, self._backbone, f'{recording_id}_360p.mp4_1s_1s.npz')
        features_data = np.load(features_path)
        recording_features = features_data['arr_0']

        sub_step_features = recording_features[start_time:end_time]
        sub_step_features = torch.from_numpy(sub_step_features).float()

        if has_errors:
            sub_step_labels = torch.ones(1, 1)
        else:
            sub_step_labels = torch.zeros(1, 1)

        features_data.close()

        return sub_step_features, sub_step_labels


def collate_fn(batch):
    # batch is a list of tuples, and each tuple is (step_features, step_labels)
    step_features, step_labels = zip(*batch)

    # Stack the step_features and step_labels
    step_features = torch.cat(step_features, dim=0)
    step_labels = torch.cat(step_labels, dim=0)

    return step_features, step_labels
