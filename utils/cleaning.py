import pandas as pd
import csv
from utils.config import config
from collections import defaultdict

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

    # Removing unwanted columns
    df_clean = df.loc[:, ~df.columns.str.contains('Unlabeled|Bone Marker|Time|Bone|Rigid Body Marker', case=False)]

    print(f"Original Count: {len(df.columns)}, Final Count:    {len(df_clean.columns)}")

    # Better naming for clarity
    df = df_clean

    prefix_pattern = r"Rigid Body Marker_|Rigid Body_|Marker_"

    df.columns = df.columns.str.replace(prefix_pattern, "", regex=True)
    df.columns = df.columns.str.replace(":", "_", regex=False)
    df.columns = df.columns.str.replace(" ", "", regex=False)
    df.columns = df.columns.str.replace(r"_1$", "", regex=True)
    df.columns = df.columns.str.replace("Marker", "")

    df.to_csv(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv", index=False)