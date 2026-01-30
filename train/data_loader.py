from torch.utils.data import Dataset
import numpy as np
from utils.config import config
from utils.plotter import plot_hands, plot_audio_correlation

def calculate_median(mocap, start_idx, end_idx):
    return np.median(mocap[:, start_idx:end_idx], axis=0) 

class ThereminDataset(Dataset):
    def __init__(self, training=True):

        mode = "train" if training else "test"
        take_name = config.take_name
        
        # Loading the data
        audio = np.load(f"out/{mode}/{take_name}_audio.npy")
        mocap = np.load(f"out/{mode}/{take_name}.npy")

        # Calculate medians for pitch and volume markers
        pitch_medians = calculate_median(mocap, 27, 36)
        volume_medians = calculate_median(mocap, 36, 45)
        
        # Reshape into a (x,y,z;x,y,z;x,y,z) and average each column
        pitch_avg = pitch_medians.reshape(3, 3).mean(axis=0)
        volume_avg = volume_medians.reshape(3, 3).mean(axis=0)
        
        left_hand_feats = mocap[:, 0:6]    # First 2 markers
        right_hand_feats = mocap[:, 6:27]  # Second 7 markers
        
        # Center hands to the new origin defined by the mean value of the antennas
        left_hand_feats = left_hand_feats - np.tile(volume_avg, 2)
        right_hand_feats = right_hand_feats - np.tile(pitch_avg, 7)
        
        mocap = np.hstack((left_hand_feats, right_hand_feats))
        mocap_normalized = (mocap - np.mean(mocap, axis=0)) / np.std(mocap, axis=0)
        
        self.audio_feats = audio
        self.mocap_feats = mocap_normalized
        
    def __len__(self):
        return len(self.audio_feats)
    
    def __getitem__(self, idx):
        audio = self.audio_feats[idx]
        mocap = self.mocap_feats[idx]
        
        return audio, mocap 
    

# Example usage
dataset = ThereminDataset()
plot_hands(dataset)
plot_audio_correlation(dataset)
