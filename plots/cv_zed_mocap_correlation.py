import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import config


TARGETS = ["pitch", "volume"]
CV_COL_MAP = {"pitch": "pitch_norm_volts", "volume": "volume_norm_volts"}
COLORS = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}

for target in TARGETS:
    take_name = config.get_take_name(target)

    cv_path = f"data/features/{take_name}_audio.npy"
    zed_path = f"data/features/{take_name}_hand.npy"
    mocap_path = f"data/features/{take_name}_hand_mocap.npy"

    if not os.path.exists(cv_path):
        print(f"  Skipping {target}: {cv_path} not found")
        continue
    if not os.path.exists(zed_path):
        print(f"  Skipping {target}: {zed_path} not found")
        continue
    if not os.path.exists(mocap_path):
        print(f"  Skipping {target}: {mocap_path} not found")
        continue

    cv = np.load(cv_path)
    zed = np.load(zed_path)
    mocap = np.load(mocap_path)

    cv = cv[::2]
    zed_avg = zed.reshape(zed.shape[0], -1, 3).mean(axis=1)
    mocap_avg = mocap.reshape(mocap.shape[0], -1, 3).mean(axis=1)
    mocap_avg = mocap_avg[::2]

    min_len = min(len(cv), len(zed_avg), len(mocap_avg))
    cv = cv[:min_len]
    zed_avg = zed_avg[:min_len]
    mocap_avg = mocap_avg[:min_len]
    frames = np.arange(min_len)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title(f"{take_name} — {target}")

    ax1.plot(frames, cv, linewidth=1.0, color="tab:blue")
    ax1.set_ylabel(f"{target.capitalize()} (norm volts)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, zed_avg[:, 0], color=COLORS["X"], lw=0.8, label="X")
    ax2.plot(frames, zed_avg[:, 1], color=COLORS["Y"], lw=0.8, label="Y")
    ax2.plot(frames, zed_avg[:, 2], color=COLORS["Z"], lw=0.8, label="Z")
    ax2.set_ylabel("ZED hand avg (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    ax3.plot(frames, mocap_avg[:, 0], color=COLORS["X"], lw=0.8, label="X")
    ax3.plot(frames, mocap_avg[:, 1], color=COLORS["Y"], lw=0.8, label="Y")
    ax3.plot(frames, mocap_avg[:, 2], color=COLORS["Z"], lw=0.8, label="Z")
    ax3.set_ylabel("Mocap hand avg (mm)")
    ax3.set_xlabel("Frame (30 fps)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    fig.tight_layout()
    plt.show(block=False)

plt.show()
