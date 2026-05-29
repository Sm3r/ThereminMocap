import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
import torch
from dotenv import load_dotenv
from pythonosc.udp_client import SimpleUDPClient

# Add project root so that hand_tracking_ZED6D, train, etc. resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hand_tracking_ZED6D.tracking import HandTracking
from hand_tracking_ZED6D.zed import Zed
from train.network import HandNet

dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path)

SC_IP = "127.0.0.1"
SC_PORT = 57120


def _load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}

    model = HandNet(
        input_dim=63,
        coord_mlp_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def _flatten_hand(hand_data: np.ndarray) -> np.ndarray | None:
    if hand_data.shape == (21, 3):
        return hand_data.flatten().astype(np.float64)
    return None


def _project_centroid(point_3d: np.ndarray, cam: Zed) -> tuple[int, int] | None:
    fx = cam.camera_params.fx
    fy = cam.camera_params.fy
    cx = cam.camera_params.cx
    cy = cam.camera_params.cy
    x, y, z = point_3d
    if np.any(np.isnan(point_3d)) or abs(z) < 1e-6:
        return None
    return int(fx * x / z + cx), int(fy * y / z + cy)


def _detect_one_frame(cam: Zed, detector: HandTracking) -> tuple:
    """Grab + detect + findpostion — returns (right_feat, left_feat, img)."""
    err = cam.zed.grab(cam.runtime_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, None, None

    cam.get_image()
    img = cam.img.copy()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    pcl = cam.point_cloud

    det_img = detector.findHands(img)
    left_data, right_data = detector.findpostion(det_img, pcl, cam.camera_params)

    return _flatten_hand(right_data), _flatten_hand(left_data), det_img


def _inference_thread(cam: Zed, detector: HandTracking, model: HandNet,
                      centroid_3d: np.ndarray, hand_label: str,
                      window_name: str, osc_path: str, device: torch.device,
                      running: threading.Event):
    """Full pipeline: grab → detect → center → buffer → infer → OSC → draw."""
    origin_offset = np.tile(centroid_3d, 21)
    buf = deque(maxlen=3)
    last_feat = None

    extract = (lambda r, l: r) if hand_label == "right" else (lambda r, l: l)

    osc_client = SimpleUDPClient(SC_IP, SC_PORT)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while running.is_set():
        err = cam.zed.grab(cam.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        cam.get_image()
        img = cam.img.copy()
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        pcl = cam.point_cloud

        det_img = detector.findHands(img)
        left_data, right_data = detector.findpostion(det_img, pcl, cam.camera_params)

        hand_data = extract(right_data, left_data)
        if hand_data is not None and hand_data.shape == (21, 3):
            feat = hand_data.flatten().astype(np.float64)
            if not np.any(np.isnan(feat)):
                last_feat = feat

        if last_feat is not None:
            centered = last_feat - origin_offset
            if not np.any(np.isnan(centered)):
                buf.append(centered)

        if len(buf) == 3:
            seq = torch.tensor(np.array(buf), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                val = model(seq).item()
            val = max(0.0, min(1.0, val))
            if osc_path == "/pitch":
                freq = 130.0 * (2093.0 / 130.0) ** val * 3.0
                osc_client.send_message(osc_path, [freq])
            else:
                amp = val * 3.0
                osc_client.send_message(osc_path, [amp])

        pt = _project_centroid(centroid_3d, cam)
        if pt is not None:
            cv2.drawMarker(det_img, pt, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.imshow(window_name, det_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running.clear()
            break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load models ------------------------------------------------------------
    pitch_ckpt = os.path.join(PROJECT_ROOT, "train", "checkpoints", "pitch_model.pt")
    volume_ckpt = os.path.join(PROJECT_ROOT, "train", "checkpoints", "volume_model.pt")
    print("Loading pitch model (right hand → pitch) ...")
    pitch_model = _load_model(pitch_ckpt, device)
    print("Loading volume model (left hand → volume) ...")
    volume_model = _load_model(volume_ckpt, device)

    # ---- Open ZED cameras -------------------------------------------------------
    serial_1 = int(os.getenv("ZED_SERIAL_1"))
    serial_2 = int(os.getenv("ZED_SERIAL_2"))

    print(f"\nOpening ZED camera 1 (serial: {serial_1}) ...")
    cam0 = Zed(None, camera_serial=serial_1)
    print(f"Opening ZED camera 2 (serial: {serial_2}) ...")
    cam1 = Zed(None, camera_serial=serial_2)
    cams = [cam0, cam1]

    detectors = [
        HandTracking(maxHands=1, detectionCon=0.2, trackCon=0.8,
                     complexity=0, draw=False)
        for _ in range(2)
    ]

    # ---- Auto-detect hand → camera mapping --------------------------------------
    cam_hand = [None, None]
    print("\n=== AUTO-DETECT HANDS ===")
    print("Show one hand in each camera. Detection runs until both hands are found.")
    print("(Press Ctrl+C to abort)\n")

    while True:
        any_new = False
        for i, (cam, det) in enumerate(zip(cams, detectors)):
            if cam_hand[i] is not None:
                continue
            right_feat, left_feat, _ = _detect_one_frame(cam, det)
            if right_feat is not None:
                cam_hand[i] = "right"
                print(f"  ✓ Camera {i} (serial {[serial_1, serial_2][i]}) → right hand")
                any_new = True
            elif left_feat is not None:
                cam_hand[i] = "left"
                print(f"  ✓ Camera {i} (serial {[serial_1, serial_2][i]}) → left hand")
                any_new = True

        if all(cam_hand):
            break

        status = []
        for i in range(2):
            if cam_hand[i] is None:
                status.append(f"Camera {i}: waiting...")
            else:
                status.append(f"Camera {i}: {cam_hand[i]} hand ✓")
        if not any_new:
            print(f"\r  {' | '.join(status)}", end="", flush=True)
    print()

    if "right" in cam_hand and "left" in cam_hand:
        pitch_cam_idx = cam_hand.index("right")
        volume_cam_idx = cam_hand.index("left")
        print(f"\nRoute: Camera {pitch_cam_idx} (right hand) → Pitch model, Camera {volume_cam_idx} (left hand) → Volume model")
    else:
        print(f"\n  Both cameras detected the same hand type ({cam_hand[0]}/{cam_hand[1]}).")
        print("  Falling back to camera index: Camera 0 → Pitch, Camera 1 → Volume")
        pitch_cam_idx = 0
        volume_cam_idx = 1

    # ---- Calibration phase --------------------------------------------------------
    CALIBRATION_FRAMES = 60
    print("\n=== CALIBRATION ===")
    print(f"Keep your hands still at your chosen origin position for {CALIBRATION_FRAMES // (30 // 2)} seconds ...")
    print(f"Each hand will define a 3D origin point used to center all subsequent positions.\n")

    pitch_feats_calib = []
    volume_feats_calib = []

    for cal_idx in range(CALIBRATION_FRAMES):
        pitch_raw = None
        volume_raw = None

        for i in range(2):
            right_feat, left_feat, _ = _detect_one_frame(cams[i], detectors[i])
            if i == pitch_cam_idx and right_feat is not None:
                pitch_raw = right_feat
            if i == volume_cam_idx and left_feat is not None:
                volume_raw = left_feat

        if pitch_raw is not None and not np.any(np.isnan(pitch_raw)):
            pitch_feats_calib.append(pitch_raw)
        if volume_raw is not None and not np.any(np.isnan(volume_raw)):
            volume_feats_calib.append(volume_raw)

        remaining = CALIBRATION_FRAMES - cal_idx - 1
        pitch_ok = len(pitch_feats_calib)
        vol_ok = len(volume_feats_calib)
        print(f"\r  Calibrating... {remaining:2d}s  |  pitch frames: {pitch_ok:3d}  volume frames: {vol_ok:3d}", end="", flush=True)
        time.sleep(1 / 30)
    print()

    if len(pitch_feats_calib) < 10 or len(volume_feats_calib) < 10:
        print(f"ERROR: Not enough calibration frames (pitch: {len(pitch_feats_calib)}, volume: {len(volume_feats_calib)}). Need at least 10 each.")
        for cam in cams:
            cam.zed.close()
        sys.exit(1)

    # Compute centroid: mean of all 21 landmarks across all calibration frames → (3,)
    all_pitch = np.array(pitch_feats_calib)   # (N, 63)
    all_volume = np.array(volume_feats_calib)  # (N, 63)

    pitch_centroid_3d = all_pitch.reshape(-1, 21, 3).mean(axis=(0, 1))  # (3,)
    volume_centroid_3d = all_volume.reshape(-1, 21, 3).mean(axis=(0, 1))  # (3,)

    print(f"\n  Pitch origin (right hand): ({pitch_centroid_3d[0]:.4f}, {pitch_centroid_3d[1]:.4f}, {pitch_centroid_3d[2]:.4f})")
    print(f"  Volume origin (left hand):  ({volume_centroid_3d[0]:.4f}, {volume_centroid_3d[1]:.4f}, {volume_centroid_3d[2]:.4f})")
    print("Calibration complete.\n")

    # ---- Start independent inference threads ---------------------------------------
    running = threading.Event()
    running.set()

    threads = []

    t = threading.Thread(
        target=_inference_thread,
        args=(cams[pitch_cam_idx], detectors[pitch_cam_idx], pitch_model,
              pitch_centroid_3d, "right",
              "Pitch Camera (right hand — press Q to quit)", "/pitch",
              device, running),
    )
    t.start()
    threads.append(t)

    t = threading.Thread(
        target=_inference_thread,
        args=(cams[volume_cam_idx], detectors[volume_cam_idx], volume_model,
              volume_centroid_3d, "left",
              "Volume Camera (left hand — press Q to quit)", "/volume",
              device, running),
    )
    t.start()
    threads.append(t)

    print("Real-time prediction running (press Q in either window or Ctrl+C to stop) ...\n")

    try:
        while running.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping ...")
    finally:
        running.clear()
        for t in threads:
            t.join(timeout=2)

        for cam in cams:
            cam.zed.close()
        cv2.destroyAllWindows()
        print("Cameras closed. Done.")


if __name__ == "__main__":
    main()
