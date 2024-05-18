import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
import math
from torch.utils.data import DataLoader


class CaptainCookSegmentDataset(Dataset):
    def __init__(self, phase, config=None):
        # self._config = config
        # self._backbone = self._config.backbone
        # self._modality = self._config.modality
        self._phase = phase
        # self._segment_length = self._config.segment_length

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"
        # self.features_directory = os.path.join('../Features', self._backbone)
        self.features_directory = os.path.join('../Features', 'omnivore')

        with open('../annotations/data_splits/environment_data_split_combined.json', 'r') as file:
            self.recording_ids = json.load(file)[self._phase]

        with open('../annotations/annotation_json/step_annotations.json', 'r') as f:
            self.annotations = json.load(f)

        self.step_dict = {}
        index_id = 0
        for recording in self.recording_ids:
            for step in self.annotations[recording]['steps']:
                if step['start_time'] < 0 or step['end_time'] < 0:
                    continue
                self.step_dict[index_id] = (recording, (math.floor(step['start_time']), math.floor(step['end_time'])), step['has_errors'])
                index_id += 1
        #self.data = data
        #self.labels = labels
        # Only imagebind has support for multiple modalities

        # TODO: Implement the following
        # Aim: To load segment length features for all videos from the dataset
        # Steps:
        # 1. Load all features file paths for the dataset
        # 2. Load error annotations json from annotations directory
        # 3. Prepare segment unique_id for each segment feature :  "{recording_id}_{step_id}_{segment_start}_{segment_end}"
        # 4. Write a method to fetch specific features given the segment unique id
        # 5. Prepare gt_annotations for each segment : Include all categories of errors for each segment
        # Format : [num_error_categories] - 0/1 matrix
        # 6. Output : Two dictionaries : features, gt_annotations with segment unique id as key


    def __len__(self):
        return len(self.step_dict)
        # # Return the total number of segments from the created dictionaries
        # pass

    def __getitem__(self, idx):
        features_path = os.path.join(self.features_directory, self.step_dict[idx][0] + '_360p.mp4_1s_1s.npz')
        recording_features = np.load(features_path)
        step_features = recording_features[self.step_dict[idx][1][0]:self.step_dict[idx][1][1], ]
        step_features = torch.from_numpy(step_features).float()
        step_labels = torch.zeros(1, step_features.shape[0]) if not self.step_dict[idx][2] else torch.ones(1, step_features.shape[0])
        recording_features.close()
        return step_features, step_labels
        # return self.data[idx], self.labels[idx]
        # Return the segment features and gt_annotations for the given idx
        # Use the segment unique id to fetch the features path and gt_annotations from the dictionaries
        # Use the method created in __init__ to fetch the features from the unique step id
        # Output: Features - (1024), Gt Annotations - (num_error_categories)



if __name__ == '__main__':

    train_dataset = CaptainCookSegmentDataset('val')
    train_loader = DataLoader(train_dataset)
    x = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape, target.shape)
        if batch_idx >5:
            break
    print(x)
    print('length: ', len(train_loader.dataset))

