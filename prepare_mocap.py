import numpy as np
from mocap_tools import Take, clean_mocap_csv, convert_tak_to_csv
from config import config


print("\nPreparing mocap data...")
#convert_tak_to_csv()
print("Cleaning mocap CSV...")
clean_mocap_csv()
print("Loading mocap data...")
take = Take(frame_rate=config.rates.mocap_fps)
take_name = config.take_name
take.readCSV(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv")

raw_markers = take.markers

# Swap Y and Z axes
'''for marker in raw_markers.values():
    for i, pos in enumerate(marker.positions):
        if pos is not None:
            pos[1], pos[2] = -pos[2], pos[1]'''



all_data = take.get_markers()
print(f"\nDownsampling the mocap data:")
print(f"Original shape: {all_data.shape}")

# Downsample to target fps
ds_factor = config.rates.mocap_fps // config.rates.target_fps
num_frames = all_data.shape[0]
downsampled_frames = num_frames // ds_factor
all_data = sum(all_data[i::ds_factor][:downsampled_frames] for i in range(ds_factor)) / ds_factor
print(f"Downsampled shape: {all_data.shape}")

np.save(f"data/dataframes/{take_name}.npy", all_data)

