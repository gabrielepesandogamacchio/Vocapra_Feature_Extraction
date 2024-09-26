import numpy as np
import os
import librosa
import h5py

DATASET_PATH = './Dataset_V2_official_classes'
hdf5_path = "dataset_logMelSpec_5000HZ_maxFreq_v0.h5"

SAMPLE_RATE = 22050
DURATION = 2 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

n_mels = 128
hop_length = 512
n_fft = 2048

# Data augmentation parameters
time_stretch_rates = [0.8, 1.0, 1.2]
pitch_shift_steps = [-2, -1, 0, 1, 2]

# Perform time stretching and pitch shifting - Data Augmentation
def augment_data(signal, sr, time_stretch_rates, pitch_shift_steps):
    augmented_data = []
    for ts_rate in time_stretch_rates:
        ts_signal = librosa.effects.time_stretch(signal, rate=ts_rate)
        for ps_step in pitch_shift_steps:
            ps_signal = librosa.effects.pitch_shift(ts_signal, sr=sr, n_steps=ps_step)
            augmented_data.append(ps_signal)
    return augmented_data

def save_logMel_dataset(dataset_path, n_mels=n_mels, n_fft=2048, hop_length=hop_length, fmax=5000, num_segments=2, overlap=0.5):
    # Structure for saving data
    data = {
        "mapping": [],
        "logMelSpec": [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    frame_length = samples_per_segment
    frame_shift = int(frame_length * overlap)

    label_mapping = {}

    # Cycle through the folders divided by goat vocalizations
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath != dataset_path:
            # Save the mapping names
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]

            # Add the semantic label to the mapping and keep track of the index
            if semantic_label not in label_mapping:
                label_mapping[semantic_label] = len(label_mapping)  # Assign a unique integer to each label

            label_index = label_mapping[semantic_label]

            if semantic_label not in data["mapping"]:
                data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Process files for a specific class (goat vocalization)
            for f in filenames:
                print(f)
                # Load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Augment data
                augmented_data = augment_data(signal, sr, time_stretch_rates, pitch_shift_steps)

                # Frame the signal and process segments by extracting log Mel spectrogram and save data
                for aug_signal in augmented_data:
                    frames = librosa.util.frame(aug_signal, frame_length=frame_length, hop_length=frame_shift).T
                    for frame in frames:
                        # Modify this line to limit frequency to 7000 Hz
                        S = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmax=fmax)
                        log_S = librosa.power_to_db(S, ref=np.max)

                        data["logMelSpec"].append(log_S.tolist())
                        data["labels"].append(label_index)  # Assign correct label for each spectrogram


    # Save data in HDF5 format
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset("mapping", data=np.array(data["mapping"], dtype='S'))
        f.create_dataset("logMelSpec", data=np.array(data["logMelSpec"]))
        f.create_dataset("labels", data=np.array(data["labels"]))

if __name__ == "__main__":
    save_logMel_dataset(DATASET_PATH, num_segments=2, overlap=0.5)
