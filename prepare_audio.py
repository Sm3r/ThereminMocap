import os
import numpy as np
import pandas as pd
from config import config


_TARGET_COLUMNS = {"pitch": "pitch_norm_volts", "volume": "volume_norm_volts"}


os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)
    cv_path = f"data/recordings/{take_name}_cv.csv"
    if not os.path.exists(cv_path):
        print(f"  Skipping {take_name}: {cv_path} not found")
        continue

    col = _TARGET_COLUMNS[target]
    values = pd.read_csv(cv_path)[col].to_numpy(dtype=np.float64)
    np.save(f"data/features/{take_name}_audio.npy", values)
    nan_count = int(np.isnan(values).sum())
    print(f"  {take_name}: {len(values)} samples, {nan_count} NaNs, "
          f"range [{values.min():.4f}, {values.max():.4f}]")
