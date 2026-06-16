import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import csv
from collections import defaultdict


def create_outlier_mask(df, columns, deviation_threshold=0.3):
    df_normalized = df.copy()
    for col in columns:
        if col in df_normalized.columns:
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max - col_min != 0:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)

    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in columns:
        if col in df_normalized.columns:
            column_median = df_normalized[col].median()
            distance_from_median = (df_normalized[col] - column_median).abs()
            outlier_mask[col] = distance_from_median > deviation_threshold

    return outlier_mask


def clean_mocap_csv(take_name, target, config):
    raw_path = f"data/features/MOCAP_{take_name}.csv"
    clean_path = f"data/features/{take_name}_mocap_clean.csv"

    with open(raw_path, "r") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(8)]

    raw_types = header_rows[2]
    raw_names = header_rows[3]
    raw_cols  = header_rows[7]

    # Determine which entities to keep based on target
    entities_to_keep = set()
    entities_to_keep.add(config.get_mocap_markerset(target))
    entities_to_keep.add(config.get_mocap_rigid_body(target))
    entities_to_keep.add(config.get_mocap_camera(target))
    entities_to_keep.add(config.get_mocap_webcam())

    config_antennas = {'pitch', 'volume'}
    for ant in config_antennas:
        label = getattr(config.mocap.rigid_bodies, ant, None)
        if label:
            entities_to_keep.add(label)

    print(f"  Keeping entities: {entities_to_keep}")

    new_columns_all = []
    name_counts = defaultdict(int)

    for i in range(len(raw_cols)):
        t = raw_types[i] if i < len(raw_types) else ""
        n = raw_names[i] if i < len(raw_names) else ""
        c = raw_cols[i]

        parts = []
        if t and t != "Type":
            parts.append(t)
        if n and n != "Name":
            parts.append(n)
        if c:
            parts.append(c)

        col_name = "_".join(parts)

        if name_counts[col_name] > 0:
            col_name = f"{col_name}_{name_counts[col_name]}"
        name_counts[col_name] += 1

        new_columns_all.append(col_name)

    # Determine which columns to keep
    keep_mask = [False] * len(new_columns_all)
    for i, col in enumerate(new_columns_all):
        if col == 'Frame':
            keep_mask[i] = True
        elif any(entity in col for entity in entities_to_keep if entity):
            keep_mask[i] = True

    kept_columns = [c for c, k in zip(new_columns_all, keep_mask) if k]
    print(f"  Raw columns: {len(new_columns_all)} -> Kept: {len(kept_columns)}")

    df = pd.read_csv(raw_path, skiprows=8, names=new_columns_all)
    df = df.loc[:, keep_mask]
    df = df.reset_index(drop=True)
    df['Frame'] = range(len(df))

    # Drop quaternion columns (W columns) and their blocks
    cols_to_drop = []
    for i in range(len(df.columns)):
        col_name = df.columns[i]
        if col_name.endswith("_W"):
            block = df.columns[i-3 : i+1]
            cols_to_drop.extend(block)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Clean up column names
    prefix_pattern = r"Rigid Body Marker_|Rigid Body_|Marker_"
    df.columns = df.columns.str.replace(prefix_pattern, "", regex=True)
    df.columns = df.columns.str.replace(":", "_", regex=False)
    df.columns = df.columns.str.replace(" ", "", regex=False)
    df.columns = df.columns.str.replace(r"_1$", "", regex=True)
    df.columns = df.columns.str.replace("Marker", "")

    df = df.reset_index(drop=True)
    df['Frame'] = range(len(df))

    # Spike removal on hand markers
    markerset_label = config.get_mocap_markerset(target)
    hand_cols = [c for c in df.columns if c.startswith(markerset_label) and c != 'Frame']

    if hand_cols:
        outlier_mask = create_outlier_mask(df, hand_cols, deviation_threshold=0.3)
        for col in hand_cols:
            if col in df.columns:
                df.loc[outlier_mask[col], col] = np.nan
                df[col] = df[col].interpolate(method='pchip', limit_direction='both')

    df.to_csv(clean_path, index=False)
    print(f"  Cleaned CSV saved: {clean_path}  shape={df.shape}")
    return clean_path
