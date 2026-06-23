import os
import shutil

import numpy as np

from mocap_tools import Take, clean_mocap_csv
from config import config


TARGETS = ["pitch", "volume"]


def process_target(target):
    take_name = config.get_take_name(target)

    solved_src = f"data/features/{take_name}_solved.csv"
    solved_dst = f"data/features/mocap/{take_name}_solved.csv"
    clean_csv = f"data/features/mocap/{take_name}_cleaned.csv"

    print("\n")
    if not os.path.exists(clean_csv):
        if os.path.exists(solved_dst):
            solved_csv = solved_dst
        elif os.path.exists(solved_src):
            print(f"  Moving {solved_src} -> {solved_dst} ...")
            shutil.move(solved_src, solved_dst)
            solved_csv = solved_dst
        else:
            print(f"  Skipping {target}: solved CSV not found")
            return

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
        np.save(f"data/features/{take_name}_hand_mocap.npy", hand)
        print(f"    Hand mocap: {hand.shape}")

    def rigid_body_position(label):
        body = take.rigid_bodies.get(label)
        if body is not None:
            valid = [p for p in body.positions if p is not None]
            if valid:
                return np.mean(valid, axis=0)
        markers = [m for m in take.markers if m.startswith(label)]
        if markers:
            all_pos = []
            for m in markers:
                for p in take.markers[m].positions:
                    if p is not None:
                        all_pos.append(p)
            if all_pos:
                return np.mean(all_pos, axis=0)
        return None

    rigids_path = f"data/features/{take_name}_rigids.csv"
    with open(rigids_path, "w") as f:
        f.write("name,x,y,z\n")
        for label, name in [(antenna_label, "antenna"),
                            (camera_label, "camera"),
                            (webcam_label, "webcam")]:
            pos = rigid_body_position(label)
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
