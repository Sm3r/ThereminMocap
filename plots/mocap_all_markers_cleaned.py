import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from config import config


TARGETS = ["pitch", "volume"]
HAND_MAP = {"pitch": "right", "volume": "left"}


def _bone_prefixes(target):
    hand = HAND_MAP[target]
    if hand == "right":
        return [f"Bone_{hand}_{hand}", f"Bone{hand}_001", f"Bone{hand}_002",
                f"Bone{hand}_003", f"Bone{hand}_004", f"Bone{hand}_005",
                f"{hand}_001", f"{hand}_002", f"{hand}_003",
                f"{hand}_004", f"{hand}_005"]
    else:
        return [f"Bone_{hand}_{hand}", f"Bone{hand}_001", f"Bone{hand}_002",
                f"Bone{hand}_003",
                f"{hand}_001", f"{hand}_002", f"{hand}_003"]


def plot_markers(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    csv_path = f"data/features/mocap/{take_name}_cleaned.csv"

    if not os.path.exists(csv_path):
        print(f"  Skipping {target}: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    bone_prefs = _bone_prefixes(target)

    marker_cols = []
    marker_labels = []
    for pref in bone_prefs:
        xc, yc, zc = f"{pref}_X", f"{pref}_Y", f"{pref}_Z"
        if xc in df.columns and yc in df.columns and zc in df.columns:
            marker_cols.append((xc, yc, zc))
            short = pref.replace(f"Bone_{hand}_", "Bone_").replace(f"Bone{hand}_", "Bone_").replace(f"{hand}_", "M_")
            marker_labels.append(short)

    if not marker_cols:
        print(f"  Skipping {target}: no bone marker columns found in {csv_path}")
        return

    num_markers = len(marker_cols)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{take_name} — {hand} hand (cleaned)', fontsize=16, fontweight='bold')

    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)

    colors = plt.cm.Set1(np.linspace(0, 0.9, num_markers))

    for mi, (xc, yc, zc) in enumerate(marker_cols):
        color = colors[mi]
        label = marker_labels[mi]
        for row, (ax, col) in enumerate([(ax_x, xc), (ax_y, yc), (ax_z, zc)]):
            vals = df[col].values
            ax.plot(vals, linewidth=0.8, alpha=0.7, color=color,
                    label=label if row == 0 else "")

    ax_x.set_ylabel('X Position')
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('X Axis')
    ax_x.legend(loc='upper right', fontsize=8)

    ax_y.set_ylabel('Y Position')
    ax_y.grid(True, alpha=0.3)
    ax_y.set_title('Y Axis')

    ax_z.set_ylabel('Z Position')
    ax_z.set_xlabel('Frame Number')
    ax_z.grid(True, alpha=0.3)
    ax_z.set_title('Z Axis')

    fig.tight_layout()
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    figs = []
    for target in TARGETS:
        fig = plot_markers(target)
        if fig is not None:
            figs.append(fig)
    if figs:
        plt.show()
