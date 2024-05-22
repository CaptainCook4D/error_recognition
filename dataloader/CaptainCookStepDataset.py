import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from constants import Constants as const


class CaptainCookStepDataset(Dataset):

    def __init__(self, config, phase, split):
        self._config = config
        self._backbone = self._config.backbone
        self._phase = phase
        self._split = split

        # Shall be activated only for IMAGEBIND
        # Modality is a list [Depth, Audio, Text, Video]
        self._modality = config.modality
        if len(self._modality) > 1:
            assert self._backbone == const.IMAGEBIND, f"Invalid backbone for modality: {self._modality}"

        with open('../annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        print("Loaded annotations...... ")

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"

        self._features_directory = self._config.video_features_directory

        if self._split == const.STEP_SPLIT:
            self._init_step_split(config, phase)
        else:
            self._init_other_split_from_file(config, phase)

    def _init_step_split(self, config, phase):
        self._recording_ids_file = "recordings_combined_splits.json"
        print(f"Loading recording ids from {self._recording_ids_file}")
        # annotations_file_path = os.path.join(os.path.dirname(__file__), f'../er_annotations/{
        # self._recording_ids_file}')
        annotations_file_path = f"/home/rxp190007/CODE/error_recognition/er_annotations/{self._recording_ids_file}"
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = self._recording_ids_json['train'] + self._recording_ids_json['val'] + \
                              self._recording_ids_json['test']

        self._step_dict = {}
        step_index_id = 0
        for recording_id in self._recording_ids:
            if recording_id == '12_6' and self._backbone == const.IMAGEBIND:
                # Skip this recording as it has no features
                continue
            self._normal_step_dict = {}
            self._error_step_dict = {}
            normal_index_id = 0
            error_index_id = 0
            # 1. Prepare step_id, list(<start, end>) for the recording_id
            recording_step_dictionary = {}
            for step in self._annotations[recording_id]['steps']:
                if step['start_time'] < 0 or step['end_time'] < 0:
                    # Ignore missing steps
                    continue
                if recording_step_dictionary.get(step['step_id']) is None:
                    recording_step_dictionary[step['step_id']] = []
                if self._backbone == const.IMAGEBIND:
                    recording_step_dictionary[step['step_id']].append(
                        (math.floor(step['start_time']/2), math.ceil(step['end_time']/2), step['has_errors']))
                else:
                    recording_step_dictionary[step['step_id']].append(
                    (math.floor(step['start_time']), math.ceil(step['end_time']), step['has_errors']))

            # 2. Add step start and end time list to the step_dict
            for step_id in recording_step_dictionary.keys():
                # If the step has errors, add it to the error_step_dict, else add it to the normal_step_dict
                if recording_step_dictionary[step_id][0][2]:
                    self._error_step_dict[f'E{error_index_id}'] = (recording_id, recording_step_dictionary[step_id])
                    error_index_id += 1
                else:
                    self._normal_step_dict[f'N{normal_index_id}'] = (
                        recording_id, recording_step_dictionary[step_id])
                    normal_index_id += 1

            np.random.seed(config.seed)
            np.random.shuffle(list(self._normal_step_dict.keys()))
            np.random.shuffle(list(self._error_step_dict.keys()))

            normal_step_indices = list(self._normal_step_dict.keys())
            error_step_indices = list(self._error_step_dict.keys())

            self._split_proportion = [0.75, 0.16, 0.9]

            num_normal_steps = len(normal_step_indices)
            num_error_steps = len(error_step_indices)

            self._split_proportion_normal = [int(num_normal_steps * self._split_proportion[0]),
                                             int(num_normal_steps * (
                                                     self._split_proportion[0] + self._split_proportion[1]))]
            self._split_proportion_error = [int(num_error_steps * self._split_proportion[0]),
                                            int(num_error_steps * (
                                                    self._split_proportion[0] + self._split_proportion[1]))]

            if phase == 'train':
                self._train_normal = normal_step_indices[:self._split_proportion_normal[0]]
                self._train_error = error_step_indices[:self._split_proportion_error[0]]
                train_indices = self._train_normal + self._train_error
                for index_id in train_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'test':
                self._val_normal = normal_step_indices[
                                   self._split_proportion_normal[0]:self._split_proportion_normal[1]]
                self._val_error = error_step_indices[
                                  self._split_proportion_error[0]:self._split_proportion_error[1]]
                val_indices = self._val_normal + self._val_error
                for index_id in val_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'val':
                self._test_normal = normal_step_indices[self._split_proportion_normal[1]:]
                self._test_error = error_step_indices[self._split_proportion_error[1]:]
                test_indices = self._test_normal + self._test_error
                for index_id in test_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1

    def _init_other_split_from_file(self, config, phase):
        self._recording_ids_file = f"{self._split}_combined_splits.json"
        # annotations_file_path = os.path.join(os.path.dirname(__file__), f'../er_annotations/{self._recording_ids_file}')
        annotations_file_path = f"/home/rxp190007/CODE/error_recognition/er_annotations/{self._recording_ids_file}"
        print(f"Loading recording ids from {self._recording_ids_file}")
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = self._recording_ids_json[phase]
        self._step_dict = {}
        index_id = 0
        for recording in self._recording_ids:
            if recording == '12_6' and self._backbone == const.IMAGEBIND:
                # Skip this recording as it has no features
                continue
            # 1. Prepare step_id, list(<start, end>) for the recording_id
            recording_step_dictionary = {}
            for step in self._annotations[recording]['steps']:
                if step['start_time'] < 0 or step['end_time'] < 0:
                    # Ignore missing steps
                    continue
                if recording_step_dictionary.get(step['step_id']) is None:
                    recording_step_dictionary[step['step_id']] = []

                if self._backbone == const.IMAGEBIND:
                    recording_step_dictionary[step['step_id']].append(
                        (math.floor(step['start_time']/2), math.ceil(step['end_time']/2), step['has_errors']))
                else:
                    recording_step_dictionary[step['step_id']].append(
                    (math.floor(step['start_time']), math.ceil(step['end_time']), step['has_errors']))

            # 2. Add step start and end time list to the step_dict
            for step_id in recording_step_dictionary.keys():
                self._step_dict[index_id] = (recording, recording_step_dictionary[step_id])
                index_id += 1

    def __len__(self):
        assert len(self._step_dict) > 0, "No data found in the dataset"
        return len(self._step_dict)

    def _build_modality_step_features_labels(self, recording_features, step_start_end_list):
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

        if step_has_errors:
            step_labels = torch.ones(N, 1)
        else:
            step_labels = torch.zeros(N, 1)

        return step_features, step_labels

    @staticmethod
    # Filename should be AUDIO: {recording_id}.wav.npz, VIDEO: {recording_id}_360p.mp4.npz
    def fetch_imagebind_data(data, filename):
        numpy_data = np.frombuffer(data[f"{filename}"], dtype=np.float32).reshape((-1, 1024))
        return numpy_data

    def _get_video_features(self, recording_id, step_start_end_list):
        if self._backbone == const.IMAGEBIND:
            recording_name = f"{recording_id}_360p.mp4"
            features_path = os.path.join(self._config.video_features_directory, f"{self._backbone}_2",
                                         f'{recording_name}.npz')
            features_data = np.load(features_path)
            recording_features = self.fetch_imagebind_data(features_data, "video_embeddings")
        else:
            features_path = os.path.join(self._config.video_features_directory, self._backbone,
                                         f'{recording_id}_360p.mp4_1s_1s.npz')
            features_data = np.load(features_path)
            recording_features = features_data['arr_0']

        step_features, step_labels = self._build_modality_step_features_labels(recording_features, step_start_end_list)
        features_data.close()
        return step_features, step_labels

    def _get_audio_features(self, recording_id, step_start_end_list):
        recording_name = f"{recording_id}.wav"
        features_path = os.path.join(self._config.audio_features_directory, self._backbone, f'{recording_name}.npz')
        features_data = np.load(features_path)
        recording_features = self.fetch_imagebind_data(features_data, "video_embeddings")
        step_features, step_labels = self._build_modality_step_features_labels(recording_features, step_start_end_list)
        features_data.close()
        return step_features, step_labels

    def _get_depth_features(self, recording_id, step_start_end_list):
        features_path = os.path.join(self._config.depth_features_directory, self._backbone,
                                     f'{recording_id}_360p.mp4_1s_1s.npz')
        features_data = np.load(features_path)
        # TODO: Correct this
        recording_features = features_data['arr_0']
        step_features, step_labels = self._build_modality_step_features_labels(recording_features, step_start_end_list)
        features_data.close()
        return step_features, step_labels

    def _get_text_features(self, recording_id, step_start_end_list):
        features_path = os.path.join(self._config.text_features_directory, self._backbone,
                                     f'{recording_id}_360p.mp4_1s_1s.npz')
        features_data = np.load(features_path)
        # TODO: Correct this
        recording_features = features_data['arr_0']
        step_features, step_labels = self._build_modality_step_features_labels(recording_features, step_start_end_list)
        features_data.close()
        return step_features, step_labels

    def _get_imagebind_features(self, recording_id, step_start_end_list):
        step_features = []
        step_labels = []

        # Load video features
        if const.VIDEO in self._modality:
            video_step_features, video_step_labels = self._get_video_features(recording_id, step_start_end_list)
            step_features.append(video_step_features)
            if len(step_labels) == 0:
                step_labels = video_step_labels

        # Load audio features
        if const.AUDIO in self._modality:
            audio_step_features, audio_step_labels = self._get_audio_features(recording_id, step_start_end_list)
            step_features.append(audio_step_features)
            if len(step_labels) == 0:
                step_labels = audio_step_labels

        # Load text features
        if const.TEXT in self._modality:
            text_step_features, text_step_labels = self._get_text_features(recording_id, step_start_end_list)
            step_features.append(text_step_features)
            if len(step_labels) == 0:
                step_labels = text_step_labels

        # Load depth features
        if const.DEPTH in self._modality:
            depth_step_features, depth_step_labels = self._get_depth_features(recording_id, step_start_end_list)
            step_features.append(depth_step_features)
            if len(step_labels) == 0:
                step_labels = depth_step_labels

        if len(step_features) > 1:
            step_features = torch.cat(step_features, dim=-1)
        else:
            step_features = step_features[0]

        return step_features, step_labels

    def __getitem__(self, idx):
        recording_id = self._step_dict[idx][0]
        # print(f"Recording ID: {recording_id}")
        step_start_end_list = self._step_dict[idx][1]

        step_features = None
        step_labels = None
        if self._backbone == const.IMAGEBIND:
            step_features, step_labels = self._get_imagebind_features(recording_id, step_start_end_list)
        elif self._backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST]:
            step_features, step_labels = self._get_video_features(recording_id, step_start_end_list)

        assert step_features is not None, f"Features not found for recording_id: {recording_id}"
        assert step_labels is not None, f"Labels not found for recording_id: {recording_id}"

        return step_features, step_labels


def collate_fn(batch):
    # batch is a list of tuples, and each tuple is (step_features, step_labels)
    step_features, step_labels = zip(*batch)

    # Stack the step_features and step_labels
    step_features = torch.cat(step_features, dim=0)
    step_labels = torch.cat(step_labels, dim=0)

    return step_features, step_labels
