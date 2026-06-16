import csv
import os
import sys
import threading
import time

import cv2
import pyzed.sl as sl
import serial

from dotenv import load_dotenv
import fnmatch
import shutil

from mocap_tools.natnet.NatNetClient import NatNetClient
from config import config

# ==========================
# CONSTANTS
# ==========================
_SVO_CODECS = {
    "H264": sl.SVO_COMPRESSION_MODE.H264,
    "H265": sl.SVO_COMPRESSION_MODE.H265,
    "LOSSLESS": sl.SVO_COMPRESSION_MODE.LOSSLESS,
    "LOSSLESS_H264": sl.SVO_COMPRESSION_MODE.H264_LOSSLESS,
    "LOSSLESS_H265": sl.SVO_COMPRESSION_MODE.H265_LOSSLESS,
}

# ==========================
# CONFIG
# ==========================
record_cv = True
record_motion = True
record_zed = True
record_webcam = True

load_dotenv()

if config.check_files_exist():
    print("[ERROR] Cannot start recording - files would be overwritten.")
    sys.exit(1)

os.makedirs("data/recordings", exist_ok=True)

name = config.take_name
cv_filename = f"data/recordings/{name}_cv.csv"
tak_filename = name
webcam_output = f"data/recordings/{name}_webcam.avi"

stop_event = threading.Event()

# ==========================
# CV THREAD (Crow module)
# ==========================
def cv_thread_fn():
    print("[CV] Initializing")
    port = config.crow_port
    if not port:
        print("[CV] No crow_port set in config, skipping CV recording")
        return

    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()

    with open(cv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "crow_frame",
            "volume_volts",
            "pitch_volts",
            "volume_norm_volts",
            "pitch_norm_volts",
        ])

        print(f"[CV] Connected to {port}")
        print("[CV] Ready, waiting for all streams…")
        try:
            start_barrier.wait()
        except threading.BrokenBarrierError:
            print("[CV] Barrier broken, aborting.")
            ser.close()
            return

        print("[CV] Recording")
        try:
            while not stop_event.is_set():
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line.startswith("cv,"):
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                _, frame, v1, v2, n1, n2 = parts
                writer.writerow([
                    int(frame),
                    float(v1),
                    float(v2),
                    float(n1),
                    float(n2),
                ])
                f.flush()
        finally:
            ser.close()

    print("[CV] Stopped")


# ==========================
# ZED THREAD
# ==========================
def zed_thread_fn(serial_number, output_file):
    print(f"[ZED {serial_number}] Initializing")

    cam = sl.Camera()

    init = sl.InitParameters()
    init.set_from_serial_number(serial_number)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = config.rates.zed_fps
    init.async_image_retrieval = False

    status = cam.open(init)
    zed_ok = (status == sl.ERROR_CODE.SUCCESS)
    if not zed_ok:
        print(f"[ZED {serial_number}] Open failed: {status} — will not record")
    else:
        print(f"[ZED {serial_number}] Camera opened successfully")

        codec = _SVO_CODECS.get(config.zed.svo_codec, sl.SVO_COMPRESSION_MODE.H264)
        recording_param = sl.RecordingParameters(output_file, codec)

        rec_err = cam.enable_recording(recording_param)
        if rec_err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED {serial_number}] Recording error ({rec_err}) — will not record")
            cam.close()
            zed_ok = False
        else:
            print(f"[ZED {serial_number}] Recording enabled (codec: {config.zed.svo_codec})")

    runtime = sl.RuntimeParameters()

    print(f"[ZED {serial_number}] Ready, waiting for all streams…")
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        print(f"[ZED {serial_number}] Barrier broken, aborting.")
        if zed_ok:
            cam.disable_recording()
        cam.close()
        return

    if not zed_ok:
        cam.close()
        return

    frames = 0
    print(f"[ZED {serial_number}] Recording")
    while not stop_event.is_set():
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frames += 1
            print(f"[ZED {serial_number}] Frames: {frames}", end="\r")

    cam.disable_recording()
    cam.close()
    print(f"\n[ZED {serial_number}] Stopped — {frames} frames recorded")


# ==========================
# WEBCAM THREAD
# ==========================
def webcam_thread_fn():
    print("[WEBCAM] Initializing")
    cap = cv2.VideoCapture(config.webcam.index, cv2.CAP_MSMF)

    ret, frame = cap.read()
    if not ret:
        print("[WEBCAM] Failed to read first frame")
        cap.release()
        return
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"[WEBCAM] Camera provides {w}x{h} at {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    out = cv2.VideoWriter(webcam_output, fourcc, fps, (w, h))

    print("[WEBCAM] Ready, waiting for all streams…")
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        print("[WEBCAM] Barrier broken, aborting.")
        cap.release()
        out.release()
        return

    frames = 0
    print("[WEBCAM] Recording")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frames += 1

    cap.release()
    out.release()
    print(f"\n[WEBCAM] Stopped — {frames} frames written")


# ==========================
# NATNET THREAD
# ==========================
def natnet_thread_fn():
    print("[MOTION] Initializing NatNet thread")

    client = NatNetClient()
    client.set_server_address('127.0.0.1')
    client.set_client_address('127.0.0.1')
    client.set_use_multicast(True)

    if not client.run():
        print("[MOTION] NatNet failed to start")
        return

    client.send_command(f"SetRecordTakeName,{tak_filename}")

    print("[MOTION] Ready, waiting for all streams…")
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        print("[MOTION] Barrier broken, aborting.")
        client.shutdown()
        return

    client.send_command("StartRecording")
    print(f"[MOTION] Recording '{tak_filename}.tak'")

    while not stop_event.is_set():
        time.sleep(0.01)

    client.send_command("StopRecording")
    time.sleep(2)
    client.shutdown()
    print("[MOTION] NatNet thread stopped")


# ==========================
# MAIN
# ==========================
threads = []

if record_cv:
    threads.append(threading.Thread(target=cv_thread_fn, daemon=True))

if record_zed:
    serial_1 = int(os.getenv("ZED_SERIAL_1"))
    threads.append(threading.Thread(
        target=zed_thread_fn,
        args=(serial_1, f"data/recordings/{name}_cam1.svo"),
        daemon=True
    ))

if record_motion:
    threads.append(threading.Thread(target=natnet_thread_fn, daemon=True))

if record_webcam:
    threads.append(threading.Thread(target=webcam_thread_fn, daemon=True))

start_barrier = threading.Barrier(len(threads), timeout=45)

for t in threads:
    t.start()

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopping…")
    stop_event.set()
    if start_barrier.broken:
        start_barrier.reset()
    else:
        try:
            start_barrier.abort()
        except ValueError:
            pass

try:
    for t in threads:
        t.join(timeout=5)
except KeyboardInterrupt:
    pass

print("All recordings stopped cleanly.")
