import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from config import config


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_MAP = {"pitch": "right", "volume": "left"}
TARGETS = ["pitch", "volume"]


def _to_display(pts):
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return out


def animate_target(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    npy_path = f"data/features/{take_name}_hand.npy"

    if not os.path.exists(npy_path):
        print(f"  Skipping {target}: {npy_path} not found")
        return

    data = np.load(npy_path)
    data = data.reshape(len(data), 21, 3)

    nan_mask = ~np.any(np.isnan(data), axis=(1, 2))
    valid_frames = np.where(nan_mask)[0]
    if len(valid_frames) == 0:
        print(f"  Skipping {target}: no valid frames in {npy_path}")
        return

    print(f"  {take_name} ({hand} hand): {len(valid_frames)} valid / {len(data)} frames")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title(f"{take_name} — {hand} hand")
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect((1, 1, 1))

    scat = ax.scatter([], [], [], c="blue" if hand == "left" else "red", s=20, label=f"{hand.title()} Hand")
    lines = [ax.plot([], [], [], c="blue" if hand == "left" else "red", lw=1, alpha=0.6)[0]
             for _ in HAND_CONNECTIONS]
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend()

    bounds = _to_display(data[valid_frames])
    margin = 0.1
    mins = bounds.min(axis=(0, 1)) - margin
    maxs = bounds.max(axis=(0, 1)) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    def update(idx):
        pts = data[idx]
        d = _to_display(pts)
        scat.set_offsets(d[:, :2])
        scat.set_3d_properties(d[:, 2], zdir="z")
        for (i, j), line in zip(HAND_CONNECTIONS, lines):
            line.set_data([d[i, 0], d[j, 0]], [d[i, 1], d[j, 1]])
            line.set_3d_properties([d[i, 2], d[j, 2]], zdir="z")
        frame_text.set_text(f"Frame: {idx}")
        return [scat, *lines, frame_text]

    anim = FuncAnimation(fig, update, frames=valid_frames,
                         interval=1000 / 30, cache_frame_data=False)
    fig.tight_layout()
    return anim


if __name__ == "__main__":
    animations = []
    for target in TARGETS:
        anim = animate_target(target)
        if anim is not None:
            animations.append(anim)

    if animations:
        plt.show()
