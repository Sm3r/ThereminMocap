import csv
import threading
import time

import cv2
import pyzed.sl as sl
import serial

from cv_reader import find_crow_port
from mocap_tools.natnet.NatNetClient import NatNetClient
import sys
import os
from dotenv import load_dotenv
import shutil
from config import config

# ==========================
# CONFIG
# ==========================
record_cv = True
record_motion = True
record_zed = True
record_webcam = True


load_dotenv()

# Prevent overwriting
if config.check_files_exist():
    print("[ERROR] Cannot start recording - files would be overwritten.")
    sys.exit(1)

name = config.take_name
os.makedirs("data/takes", exist_ok=True)
cv_filename = f"data/takes/{name}_cv.csv"
tak_filename = name
output_svo_file = f"data/takes/{name}.svo"
webcam_output = f"data/takes/{name}_webcam.avi"

stop_event = threading.Event()

# ==========================
# CV THREAD (Crow module)
# ==========================

def cv_thread_fn():
    print("[CV] Initializing")
    port = find_crow_port(port=config.crow_port)

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



_SVO_CODECS = {
    "H264": sl.SVO_COMPRESSION_MODE.H264,
    "H265": sl.SVO_COMPRESSION_MODE.H265,
    "LOSSLESS": sl.SVO_COMPRESSION_MODE.LOSSLESS,
    "LOSSLESS_H264": sl.SVO_COMPRESSION_MODE.H264_LOSSLESS,
    "LOSSLESS_H265": sl.SVO_COMPRESSION_MODE.H265_LOSSLESS,
}


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

    # Read first frame to discover actual camera resolution
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
    # probe frame discarded — recording starts after sync

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
    threads.append(threading.Thread(
        target=cv_thread_fn,
        daemon=True
    ))

if record_zed:
    serial_1 = int(os.getenv("ZED_SERIAL_1"))
    serial_2 = int(os.getenv("ZED_SERIAL_2"))

    threads.append(threading.Thread(
        target=zed_thread_fn,
        args=(serial_1, f"data/takes/{name}_cam1.svo"),
        daemon=True
    ))

    threads.append(threading.Thread(
        target=zed_thread_fn,
        args=(serial_2, f"data/takes/{name}_cam2.svo"),
        daemon=True
    ))

if record_motion:
    threads.append(threading.Thread(
        target=natnet_thread_fn,
        daemon=True
    ))

if record_webcam:
    threads.append(threading.Thread(
        target=webcam_thread_fn,
        daemon=True
    ))

start_barrier = threading.Barrier(len(threads), timeout=45)

for t in threads:
    t.start()

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopping…")
    stop_event.set()
    # Reset the barrier so waiting threads can proceed to exit
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

if record_motion:
    import fnmatch
    tak_dst = f"data/takes/{tak_filename}.tak"

    search_root = os.path.join(os.path.expanduser("~"), "Documents", "OptiTrack")
    pattern = f"{tak_filename}_*.tak"
    tak_src = None
    tak_src_mtime = 0
    for _ in range(30):
        for root, dirs, files in os.walk(search_root):
            for f in fnmatch.filter(files, pattern):
                candidate = os.path.join(root, f)
                mtime = os.path.getmtime(candidate)
                if mtime > tak_src_mtime:
                    tak_src = candidate
                    tak_src_mtime = mtime
        if tak_src:
            break
        time.sleep(1.0)

    if tak_src:
        shutil.move(tak_src, tak_dst)
        size_mb = os.path.getsize(tak_dst) / 1_000_000
        print(f"[MOTION] .tak saved to {tak_dst} ({size_mb:.1f} MB)")
    else:
        print(f"[MOTION] WARNING: .tak matching '{pattern}' not found under {search_root}")
        print(f"[MOTION] Check Motive → View → Data View → Recording for the save path")

print("All recordings stopped cleanly.")
