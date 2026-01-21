import numpy as np
import os
from utils.mocap_parser import Take
from utils.tak_to_csv import convert_tak_to_csv
from plotter import plot_3d_animation
from utils.config import config

def fill_none_with_previous(lst, default=[-10000, -10000, -10000]):
    filled = []
    for i, x in enumerate(lst):
        if x is not None:
            filled.append(x)
        elif i == 0:
            filled.append(default)
        else:
            filled.append(filled[-1])
    return filled


convert_tak_to_csv()

take = Take(frame_rate=441)
take_name = config.take_name
take.readCSV(f"data/dataframes/MOCAP_{take_name}.csv")

raw_markers = take.markers

# Swap Y and Z axes
for marker in raw_markers.values():
    for i, pos in enumerate(marker.positions):
        if pos is not None:
            pos[1], pos[2] = -pos[2], pos[1]

pitch_marker_1 = raw_markers['pitch:Marker 001'].positions
pitch_marker_2 = raw_markers['pitch:Marker 002'].positions
pitch_marker_3 = raw_markers['pitch:Marker 003'].positions
volume_marker_1 = raw_markers['volume:Marker 001'].positions
volume_marker_2 = raw_markers['volume:Marker 002'].positions
volume_marker_3 = raw_markers['volume:Marker 003'].positions

pitch_marker_1 = fill_none_with_previous(pitch_marker_1)
pitch_marker_2 = fill_none_with_previous(pitch_marker_2)
pitch_marker_3 = fill_none_with_previous(pitch_marker_3)
volume_marker_1 = fill_none_with_previous(volume_marker_1)
volume_marker_2 = fill_none_with_previous(volume_marker_2)
volume_marker_3 = fill_none_with_previous(volume_marker_3)

del raw_markers['pitch:Marker 001']
del raw_markers['pitch:Marker 002']
del raw_markers['pitch:Marker 003']
del raw_markers['volume:Marker 001']
del raw_markers['volume:Marker 002']
del raw_markers['volume:Marker 003']

pitch_marker_1 = np.array(pitch_marker_1)
pitch_marker_2 = np.array(pitch_marker_2)
pitch_marker_3 = np.array(pitch_marker_3)
volume_marker_1 = np.array(volume_marker_1)
volume_marker_2 = np.array(volume_marker_2)
volume_marker_3 = np.array(volume_marker_3)

pitch_markers = np.stack((pitch_marker_1, pitch_marker_2, pitch_marker_3), axis=1)
volume_markers = np.stack((volume_marker_1, volume_marker_2, volume_marker_3), axis=1)

print(pitch_markers.shape) 

all_markers = []
max_num_markers = 0
for frame_id in range(len(pitch_marker_1)):
    all_markers.append([])
    for value in raw_markers.values():
        if value.positions[frame_id] != None:
            all_markers[-1].append(value.positions[frame_id])
    if len(all_markers[-1]) > max_num_markers:
        max_num_markers = len(all_markers[-1])

for i in range(len(all_markers)):
    while len(all_markers[i]) < max_num_markers:
        all_markers[i].append([-10000, -10000, -10000])

all_markers = np.array(all_markers)

all_data = np.concatenate((pitch_markers, volume_markers, all_markers), axis=1)

os.makedirs("out/train", exist_ok=True)
np.save(f"out/train/{take_name}.npy", all_data)

# plot_3d_animation(pitch_marker_1, pitch_marker_2, pitch_marker_3, volume_marker_1, volume_marker_2, volume_marker_3, all_markers)

print("Done")