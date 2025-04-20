import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import TensorDataset, random_split, DataLoader

# Set torchaudio backend to 'sox_io'
torchaudio.set_audio_backend("sox_io")  # You can also try "soundfile"

def load_mel_audio_dataset(sample_rate=16000, max_len=16000 * 5):  # 5 seconds
    base_dir = os.path.dirname(__file__)
    normal_dir = os.path.join(base_dir, "normal")
    abnormal_dir = os.path.join(base_dir, "abnormal")

    # Print out paths to check correctness
    print(f"Normal directory: {normal_dir}")
    print(f"Abnormal directory: {abnormal_dir}")
    
    def load_audio_files(directory, label):
        mel_features = []  # Changed variable name to 'mel_features'
        mel_labels = []    # Changed variable name to 'mel_labels'
        for fname in os.listdir(directory):
            path = os.path.join(directory, fname)
            if not fname.endswith(".wav"):
                continue
            try:
                waveform, sr = torchaudio.load(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            # Convert to mono (if not already)
            waveform = torch.mean(waveform, dim=0)  # Convert to mono

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            # Pad/truncate to a fixed length
            if waveform.shape[0] < max_len:
                pad = max_len - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad))  # Padding at the end
            else:
                waveform = waveform[:max_len]

            # Extract Mel spectrogram
            mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64, hop_length=512)
            mel_spectrogram = mel_transform(waveform)

            # Append the Mel spectrogram and label
            mel_features.append(mel_spectrogram)  # Storing Mel spectrograms
            mel_labels.append(label)              # Storing labels

        return mel_features, mel_labels

    # Load normal and abnormal audio files
    mel_normal_features, mel_normal_labels = load_audio_files(normal_dir, 1)
    mel_abnormal_features, mel_abnormal_labels = load_audio_files(abnormal_dir, 0)

    # Combine all features and labels
    all_mel_features = torch.stack(mel_normal_features + mel_abnormal_features)
    all_mel_labels = torch.tensor(mel_normal_labels + mel_abnormal_labels, dtype=torch.int64)

    # Combine data into a TensorDataset
    all_mel_data = TensorDataset(all_mel_features, all_mel_labels)

    # Split the data for training and testing
    normal_mask = all_mel_labels == 1
    normal_data = TensorDataset(all_mel_features[normal_mask], all_mel_labels[normal_mask])

    # Train = half of normal samples
    train_size = len(normal_data) // 2
    _, train_dataset = random_split(normal_data, [len(normal_data) - train_size, train_size])

    # Test = 50% of total data
    test_size = len(all_mel_data) // 2
    _, test_dataset = random_split(all_mel_data, [len(all_mel_data) - test_size, test_size])

    # Shuffle datasets using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = load_mel_audio_dataset()
    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
