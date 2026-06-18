import os

import numpy as np

from mocap_tools import Take, clean_mocap_csv, convert_tak_to_csv
from config import config


TARGETS = ["pitch", "volume"]


def process_target(target):
    take_name = config.get_take_name(target)

    raw_csv = f"data/features/mocap/OPTITRACK_{take_name}_raw.csv"
    clean_csv = f"data/features/mocap/OPTITRACK_{take_name}_cleaned.csv"
    tak_path = f"data/recordings/{take_name}_solved.tak"

    print("\n")
    if not os.path.exists(raw_csv):
        if not os.path.exists(tak_path):
            print(f"  Skipping {target}: {tak_path} not found")
            return
        print(f"  Converting {tak_path} ...")
        convert_tak_to_csv(take_name)

    print(f"  Cleaning {take_name} ...")
    clean_mocap_csv(take_name, target, config)

    print(f"  Parsing {take_name} ...")
    take = Take(frame_rate=config.rates.mocap_fps)
    take.readCSV(clean_csv)

    markerset_label = config.get_mocap_markerset(target)
    antenna_label = config.get_mocap_rigid_body(target)
    camera_label = config.get_mocap_camera(target)
    webcam_label = config.get_mocap_webcam()

    print(f"    Hand markerset:     {markerset_label}")
    print(f"    Antenna rigid body: {antenna_label}")
    print(f"    Camera rigid body:  {camera_label}")
    print(f"    Webcam rigid body:  {webcam_label}")

    def entity_markers_data(entity_dict, label):
        matches = {k: v for k, v in entity_dict.items() if label in k}
        if not matches:
            print(f"    Warning: no '{label}' markers found")
            return None
        names = list(matches.keys())
        num_frames = len(matches[names[0]].positions)
        data = np.zeros((num_frames, len(names) * 3))
        for mi, name in enumerate(names):
            for fi, pos in enumerate(matches[name].positions):
                if pos is not None:
                    data[fi, mi * 3:mi * 3 + 3] = pos
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

    hand = downsample(entity_markers_data(take.markers, markerset_label))
    if hand is not None:
        np.save(f"data/features/mocap/TRAIN_{take_name}_hands.npy", hand)
        print(f"    Hand mocap: {hand.shape}")

    def mean_position(body):
        if body is None:
            return None
        valid = [p for p in body.positions if p is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)

    rigids_path = f"data/features/mocap/TRAIN_{take_name}_rigids.csv"
    with open(rigids_path, "w") as f:
        f.write("name,x,y,z\n")
        for label, name in [(antenna_label, "antenna"),
                            (camera_label, "camera"),
                            (webcam_label, "webcam")]:
            body = take.rigid_bodies.get(label)
            pos = mean_position(body)
            if pos is not None:
                f.write(f"{name},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
                print(f"    {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            else:
                print(f"    Warning: rigid body '{label}' not found")


if __name__ == "__main__":
    os.makedirs("data/features/mocap", exist_ok=True)
    for target in TARGETS:
        process_target(target)
    print("Done.")
