# Plots all OptiTrack hand markers (X/Y/Z) from the cleaned mocap
# CSV. Used to verify spike removal and cleaning quality.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from mocap_tools import Take
from config import config


def plot_all_markers(mocap_data=None, marker_names=None,
                     left_name="", right_name="",
                     pitch_name="", volume_name="", show=True):
    if mocap_data is None or marker_names is None:
        take_name = config.take_name
        take = Take()
        take.readCSV(f"data/features/MOCAP_{take_name}_CLEAN.csv")
        mocap_data = np.load(f"data/features/{take_name}.npy")
        marker_names = list(take.markers.keys())
        left_name = config.names.left_hand
        right_name = config.names.right_hand
        pitch_name = config.names.pitch_antenna
        volume_name = config.names.volume_antenna

    num_features = mocap_data.shape[1]
    num_markers = num_features // 3

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('All Markers - Combined View', fontsize=16, fontweight='bold')

    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)

    right_hand_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    left_hand_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))

    right_hand_idx = 0
    left_hand_idx = 0

    for marker_idx in range(num_markers):
        marker_name = marker_names[marker_idx] if marker_idx < len(marker_names) else f"Marker_{marker_idx + 1}"

        if right_name in marker_name:
            color = right_hand_colors[right_hand_idx % len(right_hand_colors)]
            right_hand_idx += 1
        elif left_name in marker_name:
            color = left_hand_colors[left_hand_idx % len(left_hand_colors)]
            left_hand_idx += 1
        elif pitch_name in marker_name.lower():
            continue
        elif volume_name in marker_name.lower():
            continue
        else:
            continue

        x_data = mocap_data[:, marker_idx * 3].copy()
        y_data = mocap_data[:, marker_idx * 3 + 1].copy()
        z_data = mocap_data[:, marker_idx * 3 + 2].copy()

        x_data[x_data == 0] = np.nan
        y_data[y_data == 0] = np.nan
        z_data[z_data == 0] = np.nan

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

    fig.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot all mocap markers from cleaned data"
    )
    parser.add_argument("--take", default=None, help="Take name (default: config)")
    args = parser.parse_args()

    if args.take:
        config.take_name = args.take

    plot_all_markers()
