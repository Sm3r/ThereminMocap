# Overlays audio pitch/volume CV signals with mocap hand markers
# from the training dataset. Used to verify audio-mocap alignment.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from config import config


def _plot_3panel_signal(frames, marker_data, marker_count, color, left_n=0,
                        overlay=None, overlay_label="", title=""):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    for row, axis_name in enumerate(['X', 'Y', 'Z']):
        for m in range(marker_count):
            data = marker_data[:, m * 3 + row]
            label = f"{'Left' if m < left_n else 'Right'}_{m + 1:03d}" if left_n else f"Marker_{m + 1:03d}"
            axes[row].plot(frames, data, linewidth=0.8, alpha=0.7, color=color, label=label)

        axes[row].set_ylabel(f"{axis_name} Position", fontsize=12)
        axes[row].grid(True, alpha=0.3)
        axes[row].legend(loc="upper right", fontsize=8)

        if overlay is not None:
            ax2 = axes[row].twinx()
            ax2.plot(frames, overlay, linewidth=1.5, alpha=0.8,
                     color="purple" if overlay_label == "Volume" else "green",
                     linestyle="--", label=overlay_label)
            ax2.set_ylabel(overlay_label, fontsize=12)
            ax2.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Frame Number", fontsize=12)
    plt.tight_layout()
    return fig


def plot_hands(dataset, left_n=2, right_n=7, show=True):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Hand Markers (Centered & Normalized)', fontsize=16, fontweight='bold')

    frames = np.arange(len(dataset.mocap_feats))
    left_data = dataset.mocap_feats[:, :left_n * 3]
    right_data = dataset.mocap_feats[:, left_n * 3:(left_n + right_n) * 3]

    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)

    for m in range(left_n):
        color = 'blue'
        label = f'LeftHand_{m + 1:03d}'
        ax_x.plot(frames, left_data[:, m * 3], linewidth=0.8, alpha=0.5, color=color, label=label)
        ax_y.plot(frames, left_data[:, m * 3 + 1], linewidth=0.8, alpha=0.5, color=color)
        ax_z.plot(frames, left_data[:, m * 3 + 2], linewidth=0.8, alpha=0.5, color=color)

    for m in range(right_n):
        color = 'red'
        label = f'RightHand_{m + 1:03d}'
        ax_x.plot(frames, right_data[:, m * 3], linewidth=0.8, alpha=0.5, color=color, label=label)
        ax_y.plot(frames, right_data[:, m * 3 + 1], linewidth=0.8, alpha=0.5, color=color)
        ax_z.plot(frames, right_data[:, m * 3 + 2], linewidth=0.8, alpha=0.5, color=color)

    ax_x.set_ylabel('X Position', fontsize=12)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('X Axis')
    ax_x.legend(loc='upper right', fontsize=8)

    ax_y.set_ylabel('Y Position', fontsize=12)
    ax_y.grid(True, alpha=0.3)
    ax_y.set_title('Y Axis')

    ax_z.set_ylabel('Z Position', fontsize=12)
    ax_z.set_xlabel('Frame Number', fontsize=12)
    ax_z.grid(True, alpha=0.3)
    ax_z.set_title('Z Axis')

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_audio_correlation(dataset, left_n=2, right_n=7, show=True):
    frames = np.arange(len(dataset.audio_feats))
    volume = dataset.audio_feats[:, 1] if dataset.audio_feats.shape[1] > 0 else np.zeros(len(frames))
    pitch = dataset.audio_feats[:, 0] if dataset.audio_feats.shape[1] > 1 else np.zeros(len(frames))

    left_data = dataset.mocap_feats[:, :left_n * 3]
    right_data = dataset.mocap_feats[:, left_n * 3:(left_n + right_n) * 3]

    fig1 = _plot_3panel_signal(frames, left_data, left_n, "blue",
                               left_n=left_n, overlay=volume,
                               overlay_label="Volume",
                               title="Audio Volume vs Left Hand Movement")
    fig2 = _plot_3panel_signal(frames, right_data, right_n, "red",
                               left_n=left_n, overlay=pitch,
                               overlay_label="Pitch",
                               title="Audio Pitch vs Right Hand Movement")

    if show:
        plt.show()
    return fig1, fig2


if __name__ == "__main__":
    import argparse
    from train.data_loader import ThereminDataset

    parser = argparse.ArgumentParser(
        description="Plot dataset features: hand markers and audio correlation"
    )
    parser.add_argument("--mode", required=True,
                        choices=["hands", "correlation"],
                        help="Plot mode")
    parser.add_argument("--take", default=None, help="Take name (default: config)")
    parser.add_argument("--left-n", type=int, default=2, help="Left hand marker count")
    parser.add_argument("--right-n", type=int, default=7, help="Right hand marker count")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Dataset split (for hands/correlation)")
    args = parser.parse_args()

    if args.take:
        config.take_name = args.take

    dataset = ThereminDataset(training=(args.split == "train"))

    if args.mode == "hands":
        plot_hands(dataset, left_n=args.left_n, right_n=args.right_n)
    elif args.mode == "correlation":
        plot_audio_correlation(dataset, left_n=args.left_n, right_n=args.right_n)
