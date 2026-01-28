import numpy as np
import os
from utils.config import config

# Configuration
TRAIN_RATIO = 0.8
take_name = config.take_name

# Load the preprocessed data
audio_data = np.load(f"out/train/{take_name}_audio.npy")
mocap_data = np.load(f"out/train/{take_name}.npy")

print(f"\nLoaded data:")
print(f"  Audio: {audio_data.shape}")
print(f"  Mocap: {mocap_data.shape}")

# Verify they have the same number of frames
assert audio_data.shape[0] == mocap_data.shape[0], "Audio and mocap frame count mismatch!"

# Calculate split index
total_frames = audio_data.shape[0]
split_idx = int(total_frames * TRAIN_RATIO)

# Split both datasets
train_audio = audio_data[:split_idx]
test_audio = audio_data[split_idx:]

train_mocap = mocap_data[:split_idx]
test_mocap = mocap_data[split_idx:]

print(f"\nSplit into:")
print(f"  Train: {train_audio.shape[0]} frames ({TRAIN_RATIO*100:.0f}%)")
print(f"  Test:  {test_audio.shape[0]} frames ({(1-TRAIN_RATIO)*100:.0f}%)")

# Create directories
os.makedirs("out/train", exist_ok=True)
os.makedirs("out/test", exist_ok=True)

# Save train split (overwrite existing)
np.save(f"out/train/{take_name}_audio.npy", train_audio)
np.save(f"out/train/{take_name}.npy", train_mocap)

# Save test split
np.save(f"out/test/{take_name}_audio.npy", test_audio)
np.save(f"out/test/{take_name}.npy", test_mocap)

print(f"\nSaved to:")
print(f"  out/train/{take_name}_audio.npy")
print(f"  out/train/{take_name}.npy")
print(f"  out/test/{take_name}_audio.npy")
print(f"  out/test/{take_name}.npy")
print("\nDone!")
