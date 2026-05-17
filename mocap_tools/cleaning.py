import pandas as pd
import numpy as np
import csv
from utils.config import config
from collections import defaultdict

def create_outlier_mask(df, columns, deviation_threshold=0.3):

    # Create normalized copy for outlier detection
    df_normalized = df.copy()
    for col in columns:
        if col in df_normalized.columns:
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max - col_min != 0:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    # Detect outliers in normalized data
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in columns:
        if col in df_normalized.columns:
            column_median = df_normalized[col].median()
            distance_from_median = (df_normalized[col] - column_median).abs()
            outlier_mask[col] = distance_from_median > deviation_threshold
    
    return outlier_mask

def clean_mocap_csv():
    take_name = config.take_name

    # Renaming the columns using the metadata to get a better understanding of the values
    with open(f"data/dataframes/MOCAP_{take_name}.csv", "r") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(8)]

    raw_types = header_rows[2]     # Type
    raw_names = header_rows[3]     # Name
    raw_cols  = header_rows[7]     # Actual column

    new_columns = []

    for i in range(len(raw_cols)):
        t = raw_types[i] if i < len(raw_types) else ""
        n = raw_names[i] if i < len(raw_names) else ""
        c = raw_cols[i]

        parts = []

        # Filter
        if t and t != "Type":
            parts.append(t)
        if n and n != "Name":
            parts.append(n)
        if c:
            parts.append(c)

        new_columns.append("_".join(parts))

    # Adding count to equally named columns
    final_columns = []
    name_counts = defaultdict(int)

    for col in new_columns:
        if name_counts[col] > 0:
            final_columns.append(f"{col}_{name_counts[col]}")
        else:
            final_columns.append(col)

        name_counts[col] += 1

    df = pd.read_csv(f"data/dataframes/MOCAP_{take_name}.csv", skiprows=8, names=final_columns)

    # Deleting rigid bodies quaternions since I don't need them
    cols_to_drop = []
    for i in range(3, len(df.columns)):
        col_name = df.columns[i]

        if col_name.endswith("_W"):
            block = df.columns[i-3 : i+1]
            cols_to_drop.extend(block)

    df = df.drop(columns=cols_to_drop)

    # Removing unwanted columns (but keep Frame column)
    df = df.loc[:, ~df.columns.str.contains('Unlabeled|Bone Marker|Time|Bone|Rigid Body Marker|Rigid', case=False) | (df.columns == 'Frame')]

    # Better naming for clarity
    prefix_pattern = r"Rigid Body Marker_|Rigid Body_|Marker_"

    df.columns = df.columns.str.replace(prefix_pattern, "", regex=True)
    df.columns = df.columns.str.replace(":", "_", regex=False)
    df.columns = df.columns.str.replace(" ", "", regex=False)
    df.columns = df.columns.str.replace(r"_1$", "", regex=True)
    df.columns = df.columns.str.replace("Marker", "")

    # Remove the first 100 frames and the last 1000 frames
    df = df.iloc[config.start_trim_frames : -config.end_trim_frames]
    df = df.reset_index(drop=True)
    df['Frame'] = range(len(df))
    
    # Removing spikes in the hand markers using normalized copy for detection
    hand_names = ['RightHand_001', 'RightHand_002', 'RightHand_003', 'RightHand_004', 'RightHand_005', 'RightHand_006', 'RightHand_007', 'LeftHand_001', 'LeftHand_002']
    hand_cols = []
    for name in hand_names:
        hand_cols.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])

    # Create outlier mask
    outlier_mask = create_outlier_mask(df, hand_cols, deviation_threshold=0.3)
    
    # Apply mask to original data and interpolate
    for col in hand_cols:
        if col in df.columns:
            df.loc[outlier_mask[col], col] = np.nan
            df[col] = df[col].interpolate(method='pchip', limit_direction='both')
        
    
    df.to_csv(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv", index=False)
    