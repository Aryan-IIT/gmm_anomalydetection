import os
import torch
import torchaudio
from torch.utils.data import TensorDataset, random_split, DataLoader

# Set torchaudio backend to 'sox_io'
torchaudio.set_audio_backend("sox_io")  # You can also try "soundfile"

def load_audio_dataset(sample_rate=16000, max_len=16000 * 5):  # 5 seconds
    base_dir = os.path.dirname(__file__)
    normal_dir = os.path.join(base_dir, "normal")
    abnormal_dir = os.path.join(base_dir, "abnormal")

    # Print out paths to check correctness
    print(f"Normal directory: {normal_dir}")
    print(f"Abnormal directory: {abnormal_dir}")
    
    def load_audio_files(directory, label):
        features = []
        labels = []
        for fname in os.listdir(directory):
            path = os.path.join(directory, fname)
            if not fname.endswith(".wav"):
                continue
            try:
                waveform, sr = torchaudio.load(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            waveform = torch.mean(waveform, dim=0)  # Convert to mono

            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            # Pad/truncate
            if waveform.shape[0] < max_len:
                pad = max_len - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad))  # Padding at the end
            else:
                waveform = waveform[:max_len]

            features.append(waveform)
            labels.append(label)
        return features, labels

    # Load normal and abnormal audio files
    normal_features, normal_labels = load_audio_files(normal_dir, 1)
    abnormal_features, abnormal_labels = load_audio_files(abnormal_dir, 0)

    all_features = torch.stack(normal_features + abnormal_features)
    all_labels = torch.tensor(normal_labels + abnormal_labels, dtype=torch.int64)

    # Combine data
    all_data = TensorDataset(all_features, all_labels)

    # Split the data for training and testing
    normal_mask = all_labels == 1
    normal_data = TensorDataset(all_features[normal_mask], all_labels[normal_mask])

    # Train = half of normal samples
    train_size = len(normal_data) // 2
    _, train_dataset = random_split(normal_data, [len(normal_data) - train_size, train_size])

    # Test = 50% of total data
    test_size = len(all_data) // 2
    _, test_dataset = random_split(all_data, [len(all_data) - test_size, test_size])

    # Shuffle datasets using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = load_audio_dataset()
    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
