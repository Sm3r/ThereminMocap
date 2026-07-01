import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config


SIGNIFICANT_JOINTS = [0, 5, 17]
JOINT_LABELS = ["Wrist (0)", "Index MCP (5)", "Pinky MCP (17)"]
COLORS = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
HAND_MAP = {"pitch": "right", "volume": "left"}
TARGETS = ["pitch", "volume"]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def plot_target(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    data_dir = "data/features"

    csv_path = os.path.join(data_dir, f"{take_name}_webcam.csv")
    npy_path = os.path.join(data_dir, f"{take_name}_webcam_hand.npy")

    if not os.path.exists(csv_path):
        print(f"  Skipping {target}: {csv_path} not found")
        return
    if not os.path.exists(npy_path):
        print(f"  Skipping {target}: {npy_path} not found")
        return

    df_raw = pd.read_csv(csv_path)
    hand_cols = sorted(c for c in df_raw.columns
                       if c.startswith(f"{hand}_") and "_detected" not in c)
    if not hand_cols:
        print(f"  Skipping {target}: no {hand} hand columns in CSV")
        return

    raw_data = df_raw[hand_cols].values
    proc_data = np.load(npy_path)

    n = min(len(raw_data), len(proc_data))
    raw_data = raw_data[:n]
    proc_data = proc_data[:n]

    # Figure 1: joint traces raw vs processed
    fig1, axes = plt.subplots(2, len(SIGNIFICANT_JOINTS),
                              figsize=(14, 6), squeeze=False)
    fig1.suptitle(f"{take_name} — {hand} hand: Raw (CSV) vs Processed (NPY)",
                  fontsize=14, fontweight="bold")

    for stage_idx, (data, label) in enumerate([(raw_data, "raw"), (proc_data, "proc")]):
        for col_idx, joint_idx in enumerate(SIGNIFICANT_JOINTS):
            ax = axes[stage_idx, col_idx]
            if stage_idx == 0:
                ax.set_title(JOINT_LABELS[col_idx])
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Frame")
            ax.grid(True, alpha=0.3)

            for ax_idx, axis_label in enumerate(["X", "Y", "Z"]):
                col_in_npy = joint_idx * 3 + ax_idx
                vals = data[:, col_in_npy]
                ax.plot(range(len(vals)), vals,
                        color=COLORS[axis_label],
                        linewidth=0.8, alpha=0.85,
                        label=axis_label if col_idx == 0 else "")

        axes[stage_idx, 0].legend(loc="upper right", fontsize=8)

    fig1.tight_layout()

    # Figure 2: 2D skeleton samples from processed data
    valid = ~np.isnan(proc_data).any(axis=1)
    valid_idxs = np.where(valid)[0]
    n_valid = len(valid_idxs)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle(f"{take_name} — {hand} hand: 2D skeleton (XY normalized)",
                  fontsize=14, fontweight="bold")

    sample_frames = [
        valid_idxs[0] if n_valid > 0 else 0,
        valid_idxs[n_valid // 2] if n_valid > 0 else 0,
        valid_idxs[-1] if n_valid > 0 else 0,
    ]

    for ax, frame_idx in zip(axes2, sample_frames):
        pts = proc_data[frame_idx].reshape(21, 3)
        for (i, j) in HAND_CONNECTIONS:
            if np.isnan(pts[i]).any() or np.isnan(pts[j]).any():
                continue
            ax.plot([pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    "b-", lw=1.5, alpha=0.7)
        ax.scatter(pts[:, 0], pts[:, 1], c="red", s=15, zorder=5)
        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("X (norm)")
        ax.set_ylabel("Y (norm)")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.1, -0.1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig2.tight_layout()
    plt.show(block=False)
    return fig1, fig2


if __name__ == "__main__":
    figs = []
    for target in TARGETS:
        result = plot_target(target)
        if result is not None:
            figs.extend(result)

    any_exists = any(
        os.path.exists(f"data/features/{config.get_take_name(t)}_webcam.csv")
        for t in TARGETS
    )
    if figs and any_exists:
        plt.show()
