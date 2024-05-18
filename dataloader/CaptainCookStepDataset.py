import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
import math
from torch.utils.data import DataLoader


class CaptainCookStepDataset(Dataset):

    def __init__(self, config, phase):
        self._config = config
        self._backbone = self._config.backbone
        self._phase = phase

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"
        self._features_directory = self._config.features_directory

        with open('../annotations/data_splits/environment_data_split_combined.json', 'r') as file:
            self._recording_ids = json.load(file)[self._phase]

        with open('../annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        self._step_dict = {}
        index_id = 0
        for recording in self._recording_ids:
            # 1. Prepare step_id, list(<start, end>) for the recording_id
            recording_step_dictionary = {}
            for step in self._annotations[recording]['steps']:
                if step['start_time'] < 0 or step['end_time'] < 0:
                    # Ignore missing steps
                    continue
                if recording_step_dictionary.get(step['step_id']) is None:
                    recording_step_dictionary[step['step_id']] = []

                recording_step_dictionary[step['step_id']].append(
                    (math.floor(step['start_time']), math.ceil(step['end_time']), step['has_errors']))

            # 2. Add step start and end time list to the step_dict
            for step_id in recording_step_dictionary.keys():
                self._step_dict[index_id] = (recording, recording_step_dictionary[step_id])
                index_id += 1

    def __len__(self):
        assert len(self._step_dict) > 0, "No data found in the dataset"
        return len(self._step_dict)

    def __getitem__(self, idx):
        recording_id = self._step_dict[idx][0]
        step_start_end_list = self._step_dict[idx][1]

        features_path = os.path.join(self._features_directory, self._backbone, f'{recording_id}_360p.mp4_1s_1s.npz')
        features_data = np.load(features_path)
        recording_features = features_data['arr_0']

        # Build step features by concatenating the features of the step from the list
        step_features = []
        step_has_errors = None
        for step_start_time, step_end_time, has_errors in step_start_end_list:
            sub_step_features = recording_features[step_start_time:step_end_time, :]
            step_features.append(sub_step_features)
            step_has_errors = has_errors
        step_features = np.concatenate(step_features, axis=0)
        step_features = torch.from_numpy(step_features).float()
        N, d = step_features.shape

        step_labels = torch.zeros(N, 1) if not step_has_errors else torch.ones(N, 1)
        features_data.close()

        return step_features, step_labels
