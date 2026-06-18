import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from config import config


TARGETS = ["pitch", "volume"]
HAND_MAP = {"pitch": "right", "volume": "left"}


def plot_markers(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    npy_path = f"data/features/mocap/TRAIN_{take_name}_hands.npy"

    if not os.path.exists(npy_path):
        print(f"  Skipping {target}: {npy_path} not found")
        return

    data = np.load(npy_path)
    num_markers = data.shape[1] // 3

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{take_name} — {hand} hand', fontsize=16, fontweight='bold')

    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)

    colors = plt.cm.Set1(np.linspace(0, 0.9, num_markers))

    for mi in range(num_markers):
        color = colors[mi]
        for row, (ax, axis) in enumerate([(ax_x, 'X'), (ax_y, 'Y'), (ax_z, 'Z')]):
            vals = data[:, mi * 3 + row]
            ax.plot(vals, linewidth=0.8, alpha=0.7, color=color,
                    label=f"Marker {mi}" if row == 0 else "")

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
