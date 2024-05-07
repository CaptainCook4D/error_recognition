from torch.utils.data import Dataset


class CaptainCookSegmentDataset(Dataset):

    def __init__(self, config):
        self._config = config
        self._backbone = self._config.backbone
        self._modality = self._config.modality
        self._phase = self._config.phase
        self._segment_length = self._config.segment_length

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"

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
        pass

    def __len__(self):
        # Return the total number of segments from the created dictionaries
        pass

    def __getitem__(self, idx):
        # Return the segment features and gt_annotations for the given idx
        # Use the segment unique id to fetch the features path and gt_annotations from the dictionaries
        # Use the method created in __init__ to fetch the features from the unique step id
        # Output: Features - (1024), Gt Annotations - (num_error_categories)
        pass
