import numpy as np
import os
from utils.config import config

TRAIN_RATIO = 0.8
take_name = config.take_name

audio_data = np.load(f"data/dataframes/{take_name}_audio.npy")
mocap_data = np.load(f"data/dataframes/{take_name}.npy")
assert audio_data.shape[0] == mocap_data.shape[0], "Audio and mocap frames mismatch!"

print(f"\nLoaded data:")
print(f"  Audio: {audio_data.shape}")
print(f"  Mocap: {mocap_data.shape}")


total_frames = audio_data.shape[0]
split_idx = int(total_frames * TRAIN_RATIO)

train_audio = audio_data[:split_idx]
test_audio = audio_data[split_idx:]

train_mocap = mocap_data[:split_idx]
test_mocap = mocap_data[split_idx:]

print(f"\nSplit into:")
print(f"  Train: {train_audio.shape[0]} frames ({TRAIN_RATIO*100:.0f}%)")
print(f"  Test:  {test_audio.shape[0]} frames ({(1-TRAIN_RATIO)*100:.0f}%)")

os.makedirs("out/train", exist_ok=True)
os.makedirs("out/test", exist_ok=True)

np.save(f"out/train/{take_name}_audio.npy", train_audio)
np.save(f"out/train/{take_name}.npy", train_mocap)

np.save(f"out/test/{take_name}_audio.npy", test_audio)
np.save(f"out/test/{take_name}.npy", test_mocap)

