import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from time import perf_counter_ns

class ThereminDataset(Dataset):
    def __init__(self, training=True):
        audio_feats = []
        mocap_feats = []

        mode = "train" if training else "test"

        for root, dirs, files in os.walk(f"out/{mode}"):
            files.sort()
            for f in files:
                data = np.load(f"out/{mode}/{f}")
                print(f"Loaded {f} with shape {data.shape}")
                # if "audio" in f:
                #     audio_feats.append(data)
                # else:
                #     mocap_feats.append(data)
                if "audio" not in f:
                    mocap_data = data
                    audio_data = np.load(f"out/{mode}/{f.replace('.npy', '_audio.npy')}")
                    max_len = min(len(mocap_data), len(audio_data))
                    audio_feats.append(audio_data[:max_len])
                    mocap_feats.append(mocap_data[:max_len, :15])

        # Convert lists to NumPy arrays
        if audio_feats:
            audio_feats = np.concatenate(audio_feats, axis=0)  # Stack along first axis

        if mocap_feats:
            mocap_feats = np.concatenate(mocap_feats, axis=0)  # Stack along first axis

        self.audio_feats = audio_feats #/ 3000.0
        self.mocap_feats = mocap_feats

        print(f"Loaded {len(self.audio_feats)} audio features and {len(self.mocap_feats)} mocap features")
    
    def __len__(self):
        return len(self.audio_feats)
    
    def __getitem__(self, idx):
        start_time = perf_counter_ns()
        audio = self.audio_feats[idx]
        mocap = self.mocap_feats[idx]

        # Normalize mocap features dividing by 3000 if they are not -10000, else put them to -1
        mocap_out = np.where(mocap != -10000, mocap / 3000, -1)

        loading_time = perf_counter_ns() - start_time

        return audio, mocap_out, loading_time
    

# dataset = ThereminDataset()
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for i, (audio, mocap) in enumerate(dataloader):
#     print(f"Batch {i}:")
#     print(f"Audio shape: {audio.shape}")
#     print(f"Mocap shape: {mocap.shape}")
#     break