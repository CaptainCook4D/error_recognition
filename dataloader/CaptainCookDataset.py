from torch.utils.data import Dataset


class CaptainCookDataset(Dataset):
	
	def __init__(
			self,
			feature_dir,
			label_dir,
			backbone,
			modality,
			phase="train",
			segment_length=1
	):
		self._feature_dir = feature_dir
		self._label_dir = label_dir
		self._backbone = backbone
		self._modality = modality
		self._phase = phase
		self._segment_length = segment_length
		
		assert phase in ["train", "val", "test"], f"Invalid phase: {phase}"
		
	def __len__(self):
		pass
	
	def __getitem__(self, idx):
		pass
	
