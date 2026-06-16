import argparse
import glob
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

from hand_tracking_ZED6D.capture import capture_to_csv
from preprocessing import preprocess_csv, fix_hand_labels, drop_minority_hand
from config import config


parser = argparse.ArgumentParser()
parser.add_argument("--rebuild", action="store_true", help="Force rebuild all CSVs from SVO files")
args = parser.parse_args()

take_name = config.take_name
data_dir = "data/dataframes"
pattern = os.path.join("data", "takes", f"{take_name}_cam*.svo2")
svo_files = sorted(glob.glob(pattern))

if not svo_files:
    print(f"No SVO files found matching {pattern}")
    sys.exit(1)

# Filter which SVO files actually need processing
need_process = []
for svo in svo_files:
    csv_path = os.path.join(data_dir, os.path.splitext(os.path.basename(svo))[0] + ".csv")
    if args.rebuild or not os.path.exists(csv_path):
        need_process.append(svo)
    else:
        print(f"  Skipping {svo} (CSV exists, use --rebuild to reprocess)")

if need_process:
    print(f"Processing {len(need_process)} camera(s) in parallel ...\n")
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=len(need_process)) as pool:
        fut_to_svo = {
            pool.submit(
                capture_to_csv,
                filename=svo,
                show_windows=True,
                fps=config.rates.zed_fps,
                use_triangulation=(config.depth_mode == "triangulation"),
                stop_event=stop_event,
            ): svo
            for svo in need_process
        }
        try:
            while fut_to_svo and not stop_event.is_set():
                done, _ = wait(fut_to_svo, timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for f in done:
                    svo = fut_to_svo.pop(f)
                    try:
                        f.result()
                        print(f"  Done: {svo}")
                    except Exception as e:
                        print(f"  Failed: {svo} — {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted, signalling all tasks to stop ...")
            stop_event.set()
            # Still collect results from tasks that finish before exit
            for f in fut_to_svo:
                try:
                    f.result(timeout=2)
                except Exception:
                    pass
else:
    print("All CSVs are up to date.")

print("\nPreprocessing CSVs ...")
for csv_path in sorted(glob.glob(os.path.join(data_dir, f"{take_name}_cam*.csv"))):
    base, _ = os.path.splitext(csv_path)
    if base.endswith("_preprocessed"):
        continue
    print(f"  {os.path.basename(csv_path)}:")
    df = pd.read_csv(csv_path)
    df = fix_hand_labels(df)
    #df = drop_minority_hand(df)
    df = preprocess_csv(df)
    # df.to_csv(f"{base}_preprocessed{ext}", index=False)


print("\nExtracting hand data to npy files ...")
cam1_csv = os.path.join(data_dir, f"{take_name}_cam1_preprocessed.csv")
if os.path.exists(cam1_csv):
    df = pd.read_csv(cam1_csv)
    left_cols = sorted(c for c in df.columns if c.startswith("left_"))
    right_cols = sorted(c for c in df.columns if c.startswith("right_"))
    left_data = df[left_cols].values
    right_data = df[right_cols].values
    np.save(f"data/dataframes/{take_name}_cam1_left_hand.npy", left_data)
    np.save(f"data/dataframes/{take_name}_cam1_right_hand.npy", right_data)
    print(f"  Saved left hand:  {left_data.shape}")
    print(f"  Saved right hand: {right_data.shape}")
else:
    print(f"  Skipping — {os.path.basename(cam1_csv)} not found")

print("\n=== Detection Summary ===")
for csv_path in sorted(glob.glob(os.path.join(data_dir, f"{take_name}_cam*.csv"))):
    base, _ = os.path.splitext(csv_path)
    if base.endswith("_preprocessed"):
        continue
    df = pd.read_csv(csv_path)
    total = len(df)

    has_2d_cols = 'left_2d_detected' in df.columns and 'right_2d_detected' in df.columns

    landmark_cols = sorted(c for c in df.columns
                           if (c.startswith("left_") or c.startswith("right_"))
                           and "_detected" not in c)
    depth_valid = df[landmark_cols].notna().any(axis=1).sum() if landmark_cols else 0

    basename = os.path.basename(csv_path)
    pct_depth = f"{depth_valid / total * 100:5.1f}%" if total else "N/A"
    print(f"  {basename}:")
    print(f"    Total frames:            {total}")
    if has_2d_cols:
        mp_detected = df[['left_2d_detected', 'right_2d_detected']].any(axis=1).sum()
        pct_mp = f"{mp_detected / total * 100:5.1f}%" if total else "N/A"
        print(f"    MediaPipe 2D detected:   {mp_detected:>6} / {total} ({pct_mp})")
    print(f"    Wrist depth valid:       {depth_valid:>6} / {total} ({pct_depth})")

