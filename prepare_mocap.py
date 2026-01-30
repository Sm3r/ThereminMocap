import numpy as np
from utils.mocap_parser import Take
from utils.tak_to_csv import convert_tak_to_csv
from utils.cleaning import clean_mocap_csv
from utils.config import config


print("\nPreparing mocap data...")
#convert_tak_to_csv()
print("Cleaning mocap CSV...")
clean_mocap_csv()
print("Loading mocap data...")
take = Take()
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

# Downsample 2x to match audio rate
num_frames = all_data.shape[0]
downsampled_frames = num_frames // 2
all_data = (all_data[0::2][:downsampled_frames] + all_data[1::2][:downsampled_frames]) / 2
print(f"Downsampled shape: {all_data.shape}")

np.save(f"data/dataframes/{take_name}.npy", all_data)

