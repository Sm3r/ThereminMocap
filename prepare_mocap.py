import argparse
import os
import sys

import numpy as np

from mocap_tools import Take, clean_mocap_csv, convert_tak_to_csv
from config import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-name", type=str, default=None,
                        help="Take name (default: from config based on --target)")
    parser.add_argument("--target", type=str, required=True,
                        choices=["pitch", "volume"],
                        help="Which target/hand to process")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force reprocess even if cleaned CSV exists")
    args = parser.parse_args()

    take_name = args.take_name or config.get_take_name(args.target)
    target = args.target

    raw_csv = f"data/features/MOCAP_{take_name}.csv"
    clean_csv = f"data/features/{take_name}_mocap_clean.csv"
    tak_path = f"data/recordings/{take_name}.tak"

    os.makedirs("data/features", exist_ok=True)

    # Step 1: Convert .tak to raw CSV if needed
    if not os.path.exists(raw_csv):
        if not os.path.exists(tak_path):
            print(f"ERROR: Neither raw CSV nor .tak file found for '{take_name}'")
            print(f"  Tried: {raw_csv}")
            print(f"  Tried: {tak_path}")
            sys.exit(1)
        print(f"Converting {tak_path} -> {raw_csv} ...")
        convert_tak_to_csv(take_name)
    else:
        print(f"  Raw CSV exists: {raw_csv}")

    # Step 2: Clean (filter to relevant entities)
    if os.path.exists(clean_csv) and not args.rebuild:
        print(f"  Clean CSV exists: {clean_csv} (use --rebuild to reprocess)")
    else:
        print(f"Cleaning mocap CSV for target='{target}' ...")
        clean_mocap_csv(take_name, target, config)

    # Step 3: Parse with Take
    print("Parsing cleaned CSV ...")
    take = Take(frame_rate=config.rates.mocap_fps)
    take.readCSV(clean_csv)

    # Step 4: Extract per-entity data
    markerset_label = config.get_mocap_markerset(target)
    antenna_label = config.get_mocap_rigid_body(target)
    camera_label = config.get_mocap_camera(target)
    webcam_label = config.get_mocap_webcam()

    print(f"  Hand markerset:     {markerset_label}")
    print(f"  Antenna rigid body: {antenna_label}")
    print(f"  Camera rigid body:  {camera_label}")
    print(f"  Webcam rigid body:  {webcam_label}")

    def entity_markers_data(entity_dict, label):
        matches = {k: v for k, v in entity_dict.items() if label in k}
        if not matches:
            print(f"  Warning: No '{label}' markers found")
            return None
        names = list(matches.keys())
        num_frames = len(matches[names[0]].positions)
        data = np.zeros((num_frames, len(names) * 3))
        for mi, name in enumerate(names):
            for fi, pos in enumerate(matches[name].positions):
                if pos is not None:
                    data[fi, mi * 3:mi * 3 + 3] = pos
        return data

    def rigid_body_data(bodies_dict, label):
        body = bodies_dict.get(label)
        if body is None:
            print(f"  Warning: Rigid body '{label}' not found")
            return None
        num_frames = len(body.positions)
        data = np.zeros((num_frames, 3))
        for fi, pos in enumerate(body.positions):
            if pos is not None:
                data[fi] = pos
        return data

    ds_factor = config.rates.mocap_fps // config.rates.target_fps

    def downsample(arr):
        if arr is None:
            return None
        n = arr.shape[0]
        ds_n = n // ds_factor
        if ds_n == 0:
            return arr
        return sum(arr[i::ds_factor][:ds_n] for i in range(ds_factor)) / ds_factor

    # Hand markers
    hand_data = downsample(entity_markers_data(take.markers, markerset_label))
    if hand_data is not None:
        np.save(f"data/features/{take_name}_hand_mocap.npy", hand_data)
        print(f"  Hand mocap: {hand_data.shape}")

    # Antenna rigid body
    ant_data = downsample(rigid_body_data(take.rigid_bodies, antenna_label))
    if ant_data is not None:
        np.save(f"data/features/{take_name}_antenna.npy", ant_data)
        print(f"  Antenna: {ant_data.shape}")

    # Camera rigid body
    cam_data = downsample(rigid_body_data(take.rigid_bodies, camera_label))
    if cam_data is not None:
        np.save(f"data/features/{take_name}_camera.npy", cam_data)
        print(f"  Camera: {cam_data.shape}")

    # Webcam rigid body
    webcam_data = downsample(rigid_body_data(take.rigid_bodies, webcam_label))
    if webcam_data is not None:
        np.save(f"data/features/{take_name}_webcam.npy", webcam_data)
        print(f"  Webcam: {webcam_data.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
