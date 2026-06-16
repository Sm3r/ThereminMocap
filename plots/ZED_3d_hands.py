# 3D animation of both hands from ZED hand-tracking CSV.
# Plots 21 MediaPipe landmarks with skeleton connections.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from config import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _get_hand_data(df: pd.DataFrame, hand: str, frame: int) -> np.ndarray | None:
    row = df[df["Frame"] == frame]
    if row.empty:
        return None
    coords = np.full((21, 3), np.nan)
    for i in range(21):
        x = row[f"{hand}_{i:02d}_X"].values[0]
        y = row[f"{hand}_{i:02d}_Y"].values[0]
        z = row[f"{hand}_{i:02d}_Z"].values[0]
        coords[i] = [x, y, z]
    if np.any(np.isnan(coords)):
        return None
    return coords


def _to_display(pts):
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return out


def update(frame, df, left_scat, right_scat, left_lines, right_lines, frame_text):
    frame_text.set_text(f"Frame: {frame}")

    for hand, scat, lines in [("left", left_scat, left_lines),
                               ("right", right_scat, right_lines)]:
        pts = _get_hand_data(df, hand, frame)
        if pts is not None:
            d = _to_display(pts)
            scat.set_offsets(d[:, :2])
            scat.set_3d_properties(d[:, 2], zdir="z")
            scat.set_visible(True)
            for (i, j), line in zip(HAND_CONNECTIONS, lines):
                line.set_data([d[i, 0], d[j, 0]],
                              [d[i, 1], d[j, 1]])
                line.set_3d_properties([d[i, 2], d[j, 2]], zdir="z")
                line.set_visible(True)
        else:
            scat.set_visible(False)
            for line in lines:
                line.set_visible(False)

    return [left_scat, right_scat, *left_lines, *right_lines, frame_text]


def main():
    parser = argparse.ArgumentParser(
        description="3D animation of both hands from cam1 CSV"
    )
    parser.add_argument("--take", default=None, help="Take name (default: config)")
    parser.add_argument("--csv", default=None, help="Path to specific CSV file")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        take = args.take or config.take_name
        data_dir = "data/dataframes"
        csv_path = os.path.join(data_dir, f"{take}_cam2_preprocessed.csv")
        if not os.path.exists(csv_path):
            fallback = os.path.join(data_dir, f"{take}_cam2.csv")
            if os.path.exists(fallback):
                csv_path = fallback
            else:
                print(f"Neither {csv_path} nor {fallback} found")
                sys.exit(1)

    if not csv_path.lower().endswith(".csv"):
        print(f"Error: expected a .csv file, got '{csv_path}'")
        sys.exit(1)

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    frames = df["Frame"].values
    print(f"  {len(frames)} frames loaded")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m) — towards you")
    ax.set_zlabel("Y (m) — up")
    ax.set_title(f"Both Hands — {os.path.basename(csv_path)}")
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect((1, 1, 1))

    left_scat = ax.scatter([], [], [], c="blue", s=20, label="Left Hand")
    right_scat = ax.scatter([], [], [], c="red", s=20, label="Right Hand")
    left_lines = [ax.plot([], [], [], c="blue", lw=1, alpha=0.6)[0]
                  for _ in HAND_CONNECTIONS]
    right_lines = [ax.plot([], [], [], c="red", lw=1, alpha=0.6)[0]
                   for _ in HAND_CONNECTIONS]
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend()

    sample = _get_hand_data(df, "left", frames[0])
    if sample is not None:
        all_d = _to_display(sample)
        r_sample = _get_hand_data(df, "right", frames[0])
        if r_sample is not None:
            all_d = np.vstack([all_d, _to_display(r_sample)])
        margin = 0.1
        mins = all_d.min(axis=0) - margin
        maxs = all_d.max(axis=0) + margin
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    ani = FuncAnimation(
        fig, update, frames=frames,
        fargs=(df, left_scat, right_scat, left_lines, right_lines, frame_text),
        interval=1000 / args.fps, cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()
