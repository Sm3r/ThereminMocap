import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import perf_counter_ns
from utils.config import config

class ThereminDataset(Dataset):
    def __init__(self, training=True):

        mode = "train" if training else "test"
        take_name = config.take_name
        
        audio = np.load(f"out/{mode}/{take_name}_audio.npy")
        mocap = np.load(f"out/{mode}/{take_name}.npy")
        
        #TO renove already checked before
        if audio.shape[0] != mocap.shape[0]:
            raise ValueError(f"Mismatch between audio and mocap frames")

        self.audio_feats = audio
        self.mocap_feats = mocap

        print(f"\nLoaded audio and mocap features")
        print(f"\nFrame size: {self.audio_feats.shape[0]}")
    
    def __len__(self):
        return len(self.audio_feats)
    
    def __getitem__(self, idx):
        start_time = perf_counter_ns()
        audio = self.audio_feats[idx]
        mocap = self.mocap_feats[idx]

        loading_time = perf_counter_ns() - start_time
        
        return audio, mocap, loading_time
    

# dataset = ThereminDataset()
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for i, (audio, mocap) in enumerate(dataloader):
#     print(f"Batch {i}:")
#     print(f"Audio shape: {audio.shape}")
#     print(f"Mocap shape: {mocap.shape}")
#     break