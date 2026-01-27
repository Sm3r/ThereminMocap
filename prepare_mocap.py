import numpy as np
import os
from utils.mocap_parser import Take
from utils.tak_to_csv import convert_tak_to_csv
from utils.cleaning import clean_mocap_csv
from utils.plotter import plot_3d_animation
from utils.config import config


#convert_tak_to_csv()
#clean_mocap_csv()

take = Take(frame_rate=441)
take_name = config.take_name
take.readCSV(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv")

raw_markers = take.markers

# Swap Y and Z axes
for marker in raw_markers.values():
    for i, pos in enumerate(marker.positions):
        if pos is not None:
            pos[1], pos[2] = -pos[2], pos[1]



all_data = take.get_all_marker_data_as_array()
print(f"\nOriginal shape: {all_data.shape}")

# Downsample 2x to match audio rate
num_frames = all_data.shape[0]
downsampled_frames = num_frames // 2
all_data = (all_data[0::2][:downsampled_frames] + all_data[1::2][:downsampled_frames]) / 2
print(f"\nDownsampled shape: {all_data.shape}")

os.makedirs("out/train", exist_ok=True)
np.save(f"out/train/{take_name}.npy", all_data)

# plot_3d_animation(pitch_marker_1, pitch_marker_2, pitch_marker_3, volume_marker_1, volume_marker_2, volume_marker_3, all_markers)

