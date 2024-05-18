# from torch.utils.data import Dataset
# import os
# import json
#
#
# class CC(Dataset):
#
#     def __init__(self, phase, config=None):
#         # self._config = config
#         # self._backbone = self._config.backbone
#         # self._modality = self._config.modality
#         self._phase = phase
#         # self._segment_length = self._config.segment_length
#
#         # assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"
#         # self.features_directory = os.path.join('/data/rohith/captaincook/features/gopro/', self._backbone)
#
#         with open('../annotations/data_splits/environment_data_split_combined.json', 'r') as file:
#             self.recording_ids = json.load(file)[self._phase]
#
#         with open('../annotations/annotation_json/step_annotations.json', 'r') as f:
#             self.annotations = json.load(f)
#
#         step_dict = {}
#         index_id = 0
#         for recording in self.recording_ids:
#             for step in self.annotations[recording]['steps']:
#                 step_dict[index_id] = (recording, (step['start_time'], step['end_time']), step['has_errors'])
#                 index_id += 1
#
#         print(index_id)
#
#
# if __name__ == '__main__':
#     newclass = CC('train')