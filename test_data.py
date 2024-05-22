import numpy as np


def fetch_numpy_array(data, filename):
    numpy_data = np.frombuffer(data[f"{filename}/data/0"], dtype=np.float32).reshape((-1, 1024))
    return numpy_data


def check_file_match(recording_id):
    audio_feature_path = f"/data/rohith/captain_cook/features/gopro/audios/imagebind/{recording_id}.wav.npz"
    video_feature_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"

    audio_data = np.load(audio_feature_path)
    video_data = np.load(video_feature_path)

    audio_numpy_data = fetch_numpy_array(audio_data, f"{recording_id}.wav")
    video_numpy_data = fetch_numpy_array(video_data, f"{recording_id}_360p.mp4")

    print(audio_numpy_data.shape)
    print(video_numpy_data.shape)

    assert audio_numpy_data.shape == video_numpy_data.shape
    return audio_numpy_data, video_numpy_data


def load_video_embeddings(video_feature_path):
    video_data = np.load(video_feature_path)
    video_numpy_data = video_data["video_embeddings"]
    return video_numpy_data

def load_audio_embeddings(audio_feature_path):
    audio_data = np.load(audio_feature_path)
    audio_numpy_data = audio_data["video_embeddings"]
    return audio_numpy_data


def test_imagebind():
    recording_id = "29_49"
    video_features_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"
    video_features = load_video_embeddings(video_features_path)

    audio_features_path = f"/data/rohith/captain_cook/features/gopro/audios/imagebind/{recording_id}.wav.npz"
    audio_features = load_audio_embeddings(audio_features_path)

    return video_features, audio_features


def test_npz():
    import numpy as np

    # Load the .npz file
    npz_file = np.load('/data/rohith/captain_cook/features/gopro/frames/tsm/1_7_360p.npz')

    # List all files/arrays in the npz file
    print("Contents of the NPZ file:")
    for file in npz_file.files:
        print(file)
        print(npz_file[file])
        print(npz_file[file].shape)


def test_pkl():
    import os
    import pickle as pkl

    def is_pickle_empty(pickle_file_path):
        """Check if a pickle file is empty or not."""
        # Check if the file exists and is not empty
        if os.path.exists(pickle_file_path) and os.path.getsize(pickle_file_path) > 0:
            return False  # File exists and has content
        else:
            return True  # File does not exist or is empty

    # Replace with the path to your pickle file
    pickle_file_path = '/data/bhavya/splits/ce_wts.pkl'

    # Check if the pickle is empty
    empty = is_pickle_empty(pickle_file_path)
    print(f"The pickle file is {'empty' if empty else 'not empty'}.")

    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        data = pkl.load(file)
        print(data)


def test_npz():
    recording_id = "10_16"
    # check_file_match(recording_id)
    # video_feature_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"
    # video_feature_path = "/data/rohith/captain_cook/features/gopro/segments/slowfast/10_16_360p.mp4_1s_1s.npz"
    video_feature_path = "/data/rohith/captain_cook/features/gopro/segments/x3d/1_33_360p.mp4_1s_1s.npz"

    features_data = np.load(video_feature_path)
    recording_features = features_data['arr_0']

    print(recording_features.shape)
    return recording_features


def test_npy():
    # Load the .npy file
    # npy_file = np.load('/data/rohith/stuff/slowfast_old/10_16_360p/10_16_360p_0.0_1.0.npy')
    npy_file = np.load('/data/rohith/stuff/x3d_old/1_33_360p/1_33_360p_0.0_1.0.npy')

    # Print the shape of the numpy array
    print(npy_file.shape)
    return npy_file


def compare_old_new():
    new_data = test_npz()
    old_data = test_npy()

    print(np.allclose(old_data, new_data))


if __name__ == "__main__":
    test_imagebind()
