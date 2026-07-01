import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from preprocessing import fix_hand_labels, remove_spikes
from config import config


# MediaPipe settings mirroring the ZED pipeline
_MAX_HANDS = 1
_DETECTION_CON = 0.2
_TRACK_CON = 0.8
_COMPLEXITY = 1

fix_hand_label = True
remove_spike = False
show_windows = True

_HAND_MAP = {"pitch": "right", "volume": "left"}


def _filter_hallucinations(df, hand, threshold=5.0):
    """Nullify frames where the wrist is a spatial outlier (MAD-based).

    With maxHands >= 2, MediaPipe sometimes hallucinates a second instance of
    the same hand.  These hallucinations typically appear at random wrist
    positions far from the real hand's median position.
    """
    wx = df[f"{hand}_00_X"].values
    wy = df[f"{hand}_00_Y"].values
    valid = ~np.isnan(wx) & ~np.isnan(wy)
    if valid.sum() < 5:
        return df

    med_x = np.nanmedian(wx)
    med_y = np.nanmedian(wy)

    dist = np.sqrt((wx - med_x) ** 2 + (wy - med_y) ** 2)
    med_dist = np.nanmedian(dist)

    if med_dist < 1e-8:
        return df

    outlier = valid & (dist > threshold * med_dist)
    if outlier.any():
        hand_cols = [c for c in df.columns if c.startswith(f"{hand}_")]
        df.loc[outlier, hand_cols] = np.nan
        print(f"  Nullified {outlier.sum()} hallucination frames for {hand} hand")

    return df


os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)
    target_hand = _HAND_MAP[target]

    avi_path = f"data/recordings/{take_name}_webcam.avi"
    if not os.path.exists(avi_path):
        print(f"  Skipping {take_name}: {avi_path} not found")
        continue

    csv_path = f"data/features/{take_name}_webcam.csv"

    if not os.path.exists(csv_path):
        print(f"  Processing {take_name} ...")
        cap = cv2.VideoCapture(avi_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=_MAX_HANDS,
            min_detection_confidence=_DETECTION_CON,
            min_tracking_confidence=_TRACK_CON,
            model_complexity=_COMPLEXITY,
        )
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        columns = ["Frame", "Timestamp_ns"]
        for hand in ("left", "right"):
            for i in range(21):
                for axis in ("X", "Y", "Z"):
                    columns.append(f"{hand}_{i:02d}_{axis}")
            columns.append(f"{hand}_2d_detected")

        rows = []
        frame_w = None
        frame = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break

            if frame_w is None:
                frame_h, frame_w = img.shape[:2]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row = {"Frame": frame, "Timestamp_ns": None}
            for col in columns[2:]:
                row[col] = np.nan

            if results.multi_hand_landmarks:
                # Per label, keep only the highest-confidence detection
                best = {}
                for lm, hd in zip(results.multi_hand_landmarks,
                                  results.multi_handedness):
                    label = "right" if hd.classification[0].index == 0 else "left"
                    score = hd.classification[0].score
                    if label not in best or score > best[label][0]:
                        best[label] = (score, lm)

                for label, (_, lm) in best.items():
                    for i, pt in enumerate(lm.landmark):
                        row[f"{label}_{i:02d}_X"] = pt.x
                        row[f"{label}_{i:02d}_Y"] = pt.y
                        row[f"{label}_{i:02d}_Z"] = pt.z
                    row[f"{label}_2d_detected"] = 1

            rows.append(row)
            frame += 1

            if show_windows:
                display = img.copy()
                h, w, _ = display.shape
                if results.multi_hand_landmarks:
                    for lm, hd in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
                        mp_draw.draw_landmarks(
                            display, lm, mp.solutions.hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                        wrist = lm.landmark[0]
                        wx, wy = int(wrist.x * w), int(wrist.y * h)
                        label = "Right" if hd.classification[0].index == 0 else "Left"
                        cv2.circle(display, (wx, wy), 8, (255, 0, 0), -1)
                        cv2.putText(display, label, (wx + 12, wy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow(f"Webcam {take_name}", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pct = frame / total * 100 if total else 0
            filled = int(25 * frame / total) if total else 0
            bar = "\u2588" * filled + "\u2591" * (25 - filled)
            print(f"\r  [{bar}] {frame:>6}/{total} ({pct:5.1f}%)", end="", flush=True)

        cap.release()
        hands.close()
        if show_windows:
            cv2.destroyAllWindows()

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

    df = pd.read_csv(csv_path)

    # Post-hoc: remove hallucination frames via wrist outlier detection
    df = _filter_hallucinations(df, target_hand)
    other_hand = "left" if target_hand == "right" else "right"
    both = (df[f"{target_hand}_2d_detected"] == 1) & (df[f"{other_hand}_2d_detected"] == 1)
    if both.any():
        other_cols = [c for c in df.columns if c.startswith(f"{other_hand}_")]
        df.loc[both, other_cols] = np.nan
        n_both = both.sum()
        print(f"  Nullified {other_hand} on {n_both}/{len(df)} dual-detect frames"
              f" ({n_both/len(df)*100:.1f}%)")

    print(f"\n  === Detection summary: {take_name} ({target_hand} hand) ===")
    if fix_hand_label:
        df = fix_hand_labels(df, target_hand=target_hand)
    if remove_spike:
        df = remove_spikes(df)

    total_frames = len(df)
    det_col = f"{target_hand}_2d_detected"
    detected = df[det_col].sum() if det_col in df.columns else total_frames
    pct_det = f"{detected / total_frames * 100:5.1f}%" if total_frames else "N/A"

    hand_cols = sorted(c for c in df.columns
                       if c.startswith(f"{target_hand}_") and "_detected" not in c)

    print(f"    Total frames:        {total_frames}")
    print(f"    MediaPipe detected:   {detected:>6} / {total_frames} ({pct_det})")

    if hand_cols:
        np.save(f"data/features/{take_name}_webcam_hand.npy", df[hand_cols].values)
        print(f"    Saved {target_hand} hand: {df[hand_cols].shape}")
