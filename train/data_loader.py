import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from ..utils.config import config


def normalize_01_colwise(x, eps=1e-8):
    min_vals = x.min(axis=0, keepdims=True)
    max_vals = x.max(axis=0, keepdims=True)
    return (x - min_vals) / (max_vals - min_vals + eps)



class ThereminDataset(Dataset):
    def __init__(self, training=True):

        mode = "train" if training else "test"
        take_name = "TestZED" #config.take_name
        
        audio = np.load(f"out/{mode}/{take_name}_audio.npy")
        mocap = np.load(f"out/{mode}/{take_name}.npy")
        
        #TO renove already checked before
        if audio.shape[0] != mocap.shape[0]:
            raise ValueError(f"Mismatch between audio and mocap frames")
        
        audio_norm = normalize_01_colwise(audio)

        self.audio_feats = audio_norm
        self.mocap_feats = mocap

        #print(f"\nLoaded audio and mocap features")
        #print(f"\nFrame size: {self.audio_feats.shape[0]}")
    
    def __len__(self):
        return len(self.audio_feats)
    
    def __getitem__(self, idx):
        audio = self.audio_feats[idx]
        mocap = self.mocap_feats[idx]
        
        return audio, mocap 
    

dataset = ThereminDataset()

data0 = dataset[0]
print(f"Sample 0 - Audio shape: {data0[0].shape}, Mocap shape: {data0[1].shape}")
print(f"Sample 0 - Audio data: {data0[0][:]}")

print(data0[1][:-18])

data1 = dataset[1]
print(data1[1][:-18])

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for i, (audio, mocap) in enumerate(dataloader):
#     print(f"Batch {i}:")
#     print(f"Audio shape: {audio.shape}")
#     print(f"Mocap shape: {mocap.shape}")
#     break