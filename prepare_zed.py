import glob
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from hand_tracking_ZED6D.capture import capture_to_csv
from utils.config import config


take_name = config.take_name
pattern = os.path.join("data", "takes", f"{take_name}_cam*.svo2")
svo_files = sorted(glob.glob(pattern))

if not svo_files:
    print(f"No SVO files found matching {pattern}")
    sys.exit(1)

n_cams = len(svo_files)
print(f"Processing {n_cams} camera(s) in parallel ...\n")

with ThreadPoolExecutor(max_workers=n_cams) as pool:
    fut_to_svo = {
        pool.submit(capture_to_csv, filename=svo, show_windows=False): svo
        for svo in svo_files
    }
    try:
        while fut_to_svo:
            done, _ = wait(fut_to_svo, timeout=0.5, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for f in done:
                svo = fut_to_svo.pop(f)
                try:
                    f.result()
                    print(f"  ✓ Done: {svo}")
                except Exception as e:
                    print(f"  ✗ Failed: {svo} — {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted, cancelling remaining tasks ...")
        for f in fut_to_svo:
            f.cancel()
