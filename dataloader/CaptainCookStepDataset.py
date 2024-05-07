from torch.utils.data import Dataset


class CaptainCookStepDataset(Dataset):

    def __init__(self, config):
        self._config = config
        self._modality = self._config.modality
        self._phase = self._config.phase
        self._segment_length = self._config.segment_length

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"

        # Only imagebind has support for multiple modalities

        # TODO: Implement the following
        # Aim: To load features for steps for all videos from the dataset
        # Steps:
        # 1. Load all features file paths for the dataset for a backbone and modality
        # 2. Load error annotations json from annotations directory
        # 3. Prepare step unique_id for each step feature :  "{recording_id}_{step_id}"
        # 4. Write a method to fetch specific features given the step unique id
        # 5. Prepare gt_annotations for each step : Include all categories of errors for each segment of the step
        # 6. Gt Annotations : For each step, include all categories of errors for each segment of the step
        # Format : [segments, num_error_categories] - 0/1 matrix
        # Output : Two dictionaries : features, gt_annotations with step unique id as key

    def __len__(self):
        # Return the total number of steps from the created dictionaries
        pass

    def __getitem__(self, idx):
        # Return the step features and gt_annotations for the given idx
        # Output: Features - (segments x 1024), Gt Annotations - (segments x num_categories)
        pass
