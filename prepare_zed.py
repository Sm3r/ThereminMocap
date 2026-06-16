import os
import numpy as np
import pandas as pd
from hand_tracking_ZED6D.capture import capture_to_csv
from config import config


os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)

    svo_path = f"data/recordings/{take_name}_cam1.svo2"
    if not os.path.exists(svo_path):
        print(f"  Skipping {take_name}: {svo_path} not found")
        continue

    csv_path = f"data/features/{take_name}_cam1.csv"

    if not os.path.exists(csv_path):
        print(f"  Processing {take_name} ...")
        capture_to_csv(
            filename=svo_path,
            output_csv=csv_path,
            show_windows=True,
            fps=config.rates.zed_fps,
            use_triangulation=(config.depth_mode == "triangulation"),
        )

    df = pd.read_csv(csv_path)
    left_cols = sorted(c for c in df.columns
                       if c.startswith("left_") and "_detected" not in c)
    right_cols = sorted(c for c in df.columns
                        if c.startswith("right_") and "_detected" not in c)

    if left_cols:
        np.save(f"data/features/{take_name}_left_hand.npy", df[left_cols].values)
    if right_cols:
        np.save(f"data/features/{take_name}_right_hand.npy", df[right_cols].values)

    print(f"  {take_name}: left {df[left_cols].shape if left_cols else (0,)}, "
          f"right {df[right_cols].shape if right_cols else (0,)}")
