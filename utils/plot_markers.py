import numpy as np
import matplotlib.pyplot as plt
from config import config
from mocap_parser import Take

take_name = config.take_name

# Load mocap data and get marker names
take = Take()
take.readCSV(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv", config=config.__dict__)

mocap_data = np.load(f"data/dataframes/{take_name}.npy")
marker_names = sorted(take.markers.keys())  # Get actual marker names


# Calculate number of markers (each has X, Y, Z)
num_features = mocap_data.shape[1]
num_markers = num_features // 3


# Create combined plot for all markers
fig = plt.figure(figsize=(14, 10))
fig.suptitle('All Markers - Combined View', fontsize=16, fontweight='bold')

ax_x = plt.subplot(3, 1, 1)
ax_y = plt.subplot(3, 1, 2)
ax_z = plt.subplot(3, 1, 3)

# Color palettes for each category
right_hand_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
left_hand_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
pitch_colors = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
volume_colors = plt.cm.Purples(np.linspace(0.4, 0.9, 10))

right_hand_idx = 0
left_hand_idx = 0
pitch_idx = 0
volume_idx = 0

# Plot all markers
for marker_idx in range(num_markers):
    marker_name = marker_names[marker_idx] if marker_idx < len(marker_names) else f"Marker_{marker_idx + 1}"
    
    # Determine category and color - only plot hand markers
    if config.right_hand_name in marker_name:
        color = right_hand_colors[right_hand_idx % len(right_hand_colors)]
        right_hand_idx += 1
    elif config.left_hand_name in marker_name:
        color = left_hand_colors[left_hand_idx % len(left_hand_colors)]
        left_hand_idx += 1
    elif config.pitch_antenna_name in marker_name.lower():
        pitch_idx += 1
        continue  # Skip pitch markers in figure 1
    elif config.volume_antenna_name in marker_name.lower():
        volume_idx += 1
        continue  # Skip volume markers in figure 1
    else:
        continue  # Skip markers not in any category
    
    # Extract X, Y, Z for this marker
    x_col = marker_idx * 3
    y_col = marker_idx * 3 + 1
    z_col = marker_idx * 3 + 2
    
    x_data = mocap_data[:, x_col].copy()
    y_data = mocap_data[:, y_col].copy()
    z_data = mocap_data[:, z_col].copy()
    
    # Replace 0 values with NaN so they don't show up in the plot
    x_data[x_data == 0] = np.nan
    y_data[y_data == 0] = np.nan
    z_data[z_data == 0] = np.nan
    
    # Plot on combined axes with alpha (only add label to first axis)
    ax_x.plot(x_data, linewidth=0.8, alpha=0.7, label=marker_name, color=color)
    ax_y.plot(y_data, linewidth=0.8, alpha=0.7, color=color)
    ax_z.plot(z_data, linewidth=0.8, alpha=0.7, color=color)

ax_x.set_ylabel('X Position', fontsize=12)
ax_x.grid(True, alpha=0.3)
ax_x.set_title('X Axis')

ax_y.set_ylabel('Y Position', fontsize=12)
ax_y.grid(True, alpha=0.3)
ax_y.set_title('Y Axis')

ax_z.set_ylabel('Z Position', fontsize=12)
ax_z.set_xlabel('Frame Number', fontsize=12)
ax_z.grid(True, alpha=0.3)
ax_z.set_title('Z Axis')

# Single legend outside the subplots
fig.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend on the right
plt.show()