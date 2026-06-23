import argparse
import csv
import os
import sys
import threading
import time

import cv2
import pyzed.sl as sl
import serial

from dotenv import load_dotenv

from mocap_tools.natnet.NatNetClient import NatNetClient
from config import config


_SVO_CODECS = {
    "H264": sl.SVO_COMPRESSION_MODE.H264,
    "H265": sl.SVO_COMPRESSION_MODE.H265,
    "LOSSLESS": sl.SVO_COMPRESSION_MODE.LOSSLESS,
    "LOSSLESS_H264": sl.SVO_COMPRESSION_MODE.H264_LOSSLESS,
    "LOSSLESS_H265": sl.SVO_COMPRESSION_MODE.H265_LOSSLESS,
}

record_cv = True
record_motion = True
record_zed = True
record_webcam = True

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--pitch", action="store_true", help="Record pitch take (default)")
group.add_argument("--volume", action="store_true", help="Record volume take")
args = parser.parse_args()

target = "volume" if args.volume else "pitch"
name = config.get_take_name(target)

load_dotenv()

if config.check_files_exist(name):
    print("[ERROR] Cannot start recording - files would be overwritten.")
    sys.exit(1)

os.makedirs("data/recordings", exist_ok=True)

print(f"Recording {target} take: {name}")

cv_filename = f"data/recordings/{name}_cv.csv"
tak_filename = name
webcam_output = f"data/recordings/{name}_webcam.avi"
zed_output = f"data/recordings/{name}_cam1.svo"
zed_timestamp_output = f"data/recordings/{name}_cam1_timestamps.csv"
webcam_timestamp_output = f"data/recordings/{name}_webcam_timestamps.csv"

stop_event = threading.Event()
recording_start_ns = None
start_barrier = None


def mark_recording_start():
    global recording_start_ns
    recording_start_ns = time.perf_counter_ns()


# ==========================
# CV THREAD
# ==========================
def cv_thread_fn():
    print("[CV] Initializing")

    port = config.crow_port
    if not port:
        print("[CV] No crow_port set in config, skipping CV recording")
        return

    def parse_cv_line(line):
        parts = line.split(",")
        if len(parts) not in (5, 6):
            return None
        if parts[0] != "cv":
            return None
        try:
            v1, v2, n1, n2 = parts[-4:]
            return (float(v1), float(v2), float(n1), float(n2))
        except ValueError:
            return None

    ser = None
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(0.5)

        with open(cv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_ms",
                "volume_volts",
                "pitch_volts",
                "volume_norm_volts",
                "pitch_norm_volts",
            ])

            print(f"[CV] Connected to {port}")
            print("[CV] Ready, waiting for all streams...")

            try:
                start_barrier.wait()
            except threading.BrokenBarrierError:
                print("[CV] Barrier broken, aborting.")
                return

            ser.reset_input_buffer()

            print("[CV] Recording")

            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                now_ns = time.perf_counter_ns()
                line = raw.decode("utf-8", errors="replace").strip()
                parsed = parse_cv_line(line)
                if parsed is None:
                    continue
                if recording_start_ns is not None:
                    time_ms = (now_ns - recording_start_ns) * 1e-6
                else:
                    time_ms = 0.0
                v1, v2, n1, n2 = parsed
                writer.writerow([time_ms, v1, v2, n1, n2])

            print("[CV] Stopped")

    except Exception as e:
        print(f"[CV] Error: {e}")

    finally:
        if ser is not None:
            ser.close()


# ==========================
# ZED THREAD
# ==========================
def zed_thread_fn(serial_number, output_file, timestamp_file):
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
        print(f"[ZED {serial_number}] Open failed: {status} - will not record")
    else:
        print(f"[ZED {serial_number}] Camera opened successfully")

        codec = _SVO_CODECS.get(config.zed.svo_codec, sl.SVO_COMPRESSION_MODE.H264)
        recording_param = sl.RecordingParameters(output_file, codec)
        rec_err = cam.enable_recording(recording_param)

        if rec_err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED {serial_number}] Recording error ({rec_err}) - will not record")
            cam.close()
            zed_ok = False
        else:
            print(f"[ZED {serial_number}] Recording enabled, codec: {config.zed.svo_codec}")

    runtime = sl.RuntimeParameters()

    print(f"[ZED {serial_number}] Ready, waiting for all streams...")

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

    try:
        with open(timestamp_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zed_frame", "host_time_ns", "rel_time_s", "zed_image_time_ns"])

            while not stop_event.is_set():
                grab_status = cam.grab(runtime)

                if grab_status == sl.ERROR_CODE.SUCCESS:
                    host_time_ns = time.perf_counter_ns()

                    if recording_start_ns is not None:
                        rel_time_s = (host_time_ns - recording_start_ns) * 1e-9
                    else:
                        rel_time_s = 0.0

                    try:
                        zed_image_time_ns = cam.get_timestamp(
                            sl.TIME_REFERENCE.IMAGE
                        ).get_nanoseconds()
                    except Exception:
                        zed_image_time_ns = ""

                    writer.writerow([frames, host_time_ns, rel_time_s, zed_image_time_ns])
                    frames += 1
                    print(f"[ZED {serial_number}] Frames: {frames}", end="\r")

    finally:
        cam.disable_recording()
        cam.close()

    print(f"\n[ZED {serial_number}] Stopped - {frames} frames recorded")


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

    if not out.isOpened():
        print("[WEBCAM] Failed to open VideoWriter")
        cap.release()
        return

    print("[WEBCAM] Ready, waiting for all streams...")

    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        print("[WEBCAM] Barrier broken, aborting.")
        cap.release()
        out.release()
        return

    frames = 0
    print("[WEBCAM] Recording")

    try:
        with open(webcam_timestamp_output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["webcam_frame", "host_time_ns", "rel_time_s"])

            while not stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    host_time_ns = time.perf_counter_ns()

                    if recording_start_ns is not None:
                        rel_time_s = (host_time_ns - recording_start_ns) * 1e-9
                    else:
                        rel_time_s = 0.0

                    out.write(frame)
                    writer.writerow([frames, host_time_ns, rel_time_s])
                    frames += 1
    finally:
        cap.release()
        out.release()

    print(f"\n[WEBCAM] Stopped - {frames} frames written")


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

    print("[MOTION] Ready, waiting for all streams...")

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
def main():
    global start_barrier

    threads = []

    if record_cv:
        threads.append(threading.Thread(target=cv_thread_fn, daemon=True))

    if record_zed:
        zed_serial_env = os.getenv("ZED_SERIAL_1")
        if not zed_serial_env:
            print("[ERROR] ZED_SERIAL_1 is not set in .env")
            sys.exit(1)
        serial_1 = int(zed_serial_env)
        threads.append(threading.Thread(
            target=zed_thread_fn,
            args=(serial_1, zed_output, zed_timestamp_output),
            daemon=True,
        ))

    if record_motion:
        threads.append(threading.Thread(target=natnet_thread_fn, daemon=True))

    if record_webcam:
        threads.append(threading.Thread(target=webcam_thread_fn, daemon=True))

    if not threads:
        print("[ERROR] No recording streams enabled.")
        sys.exit(1)

    start_barrier = threading.Barrier(
        len(threads),
        timeout=45,
        action=mark_recording_start,
    )

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        try:
            start_barrier.abort()
        except Exception:
            pass

    for t in threads:
        t.join(timeout=5)

    print("All recordings stopped cleanly.")


if __name__ == "__main__":
    main()
