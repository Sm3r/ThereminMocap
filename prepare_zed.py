import glob
import os
from hand_tracking_ZED6D.capture import capture_to_csv
from utils.config import config

take_name = config.take_name
pattern = os.path.join("data", "takes", f"{take_name}_cam*.svo2")
svo_files = sorted(glob.glob(pattern))

for svo in svo_files:
    print(f"Processing {svo}...")
    capture_to_csv(filename=svo)
