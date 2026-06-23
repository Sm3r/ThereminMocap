import os
import numpy as np
import pandas as pd
from scipy import interpolate
from config import config


_TARGET_COLUMNS = {"pitch": "pitch_norm_volts", "volume": "volume_norm_volts"}


os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)
    cv_path = f"data/recordings/{take_name}_cv.csv"
    if not os.path.exists(cv_path):
        print(f"  Skipping {take_name}: {cv_path} not found")
        continue

    df = pd.read_csv(cv_path)
    raw = df[_TARGET_COLUMNS[target]].to_numpy(dtype=np.float64)
    time_ms = df["time_ms"].to_numpy(dtype=np.float64)

    if len(raw) < 2:
        print(f"  Skipping {take_name}: not enough samples ({len(raw)})")
        continue

    target_fps = config.rates.there_fps
    interval_ms = 1000.0 / target_fps
    uniform_t = np.arange(0.0, time_ms[-1], interval_ms)

    valid = ~np.isnan(raw)
    if valid.sum() < 2:
        uniform = np.full_like(uniform_t, np.nan)
    else:
        f = interpolate.interp1d(
            time_ms[valid], raw[valid], kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        uniform = f(uniform_t)

    np.save(f"data/features/{take_name}_audio.npy", uniform)
    nan_count = int(np.isnan(uniform).sum())
    print(f"  {take_name}: {len(raw)} raw → {len(uniform)} uniform at {target_fps} fps, "
          f"{nan_count} NaNs, range [{uniform.min():.4f}, {uniform.max():.4f}]")
