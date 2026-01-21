import threading
import time
import wave
import pyaudio
import pyzed.sl as sl
from utils.natnet.NatNetClient import NatNetClient
import array
import sys
import os
import shutil
from utils.config import config

# ==========================
# CONFIG
# ==========================
record_audio = True
record_motion = True
record_zed = True
DEBUG_DEVICES = True

SAMPLE_WIDTH = 2
CHANNELS_TO_SAVE = 3
INPUT_CHANNELS = 8

chunk = 1024
sample_format = pyaudio.paInt16
channels = 8
fs = 44100


# Prevent overwriting
if config.check_files_exist():
    print("[ERROR] Cannot start recording - files would be overwritten.")
    sys.exit(1)

name = config.take_name
os.makedirs("data/takes", exist_ok=True)
audio_filename = f"data/takes/{name}.wav"
tak_filename = name
output_svo_file = f"data/takes/{name}.svo"

stop_event = threading.Event()

# ==========================
# AUDIO THREAD
# ==========================

def audio_thread_fn():
    print("[AUDIO] Starting")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=sample_format,
        channels=INPUT_CHANNELS,
        rate=fs,
        input=True,
        input_device_index=21,
        frames_per_buffer=1024
    )

    recorded_frames = []

    while not stop_event.is_set():
        try:
            data = stream.read(1024, exception_on_overflow=False)

            # Convert bytes -> array of 16-bit ints
            samples = array.array('h', data)

            # Each "frame" is INPUT_CHANNELS samples
            frames_per_buffer = len(samples) // INPUT_CHANNELS

            # Extract only first 3 channels
            new_samples = array.array('h', [samples[i*INPUT_CHANNELS + ch]
                                            for i in range(frames_per_buffer)
                                            for ch in range(CHANNELS_TO_SAVE)])

            recorded_frames.append(new_samples.tobytes())

        except Exception as e:
            print("[AUDIO] Error:", e)
            break

    stream.stop_stream()
    stream.close()

    # Save WAV
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS_TO_SAVE)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(fs)
    wf.writeframes(b''.join(recorded_frames))
    wf.close()

    p.terminate()
    print("[AUDIO] Stopped")



# ==========================
# ZED THREAD
# ==========================
def zed_thread_fn():
    print("[ZED] Starting ZED thread")

    cam = sl.Camera()

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_resolution = sl.RESOLUTION.VGA
    init.camera_fps = 60
    init.async_image_retrieval = False

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ZED] Camera open failed:", status)
        return

    recording_param = sl.RecordingParameters(
        output_svo_file,
        sl.SVO_COMPRESSION_MODE.H264
    )

    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("[ZED] Recording error:", err)
        cam.close()
        return

    runtime = sl.RuntimeParameters()
    frames_recorded = 0

    while not stop_event.is_set():
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frames_recorded += 1
            print(f"[ZED] Frames: {frames_recorded}", end="\r")

    cam.disable_recording()
    cam.close()
    print("\n[ZED] ZED thread stopped")

# ==========================
# NATNET THREAD
# ==========================
def natnet_thread_fn():
    print("[MOTION] Starting NatNet thread")

    client = NatNetClient()
    client.set_server_address('127.0.0.1')
    client.set_client_address('127.0.0.1')
    client.set_use_multicast(True)

    if not client.run():
        print("[MOTION] NatNet failed to start")
        return

    client.send_command(f"SetRecordTakeName,{tak_filename}")
    client.send_command("StartRecording")
    print(f"[MOTION] Recording '{tak_filename}.tak'")

    while not stop_event.is_set():
        time.sleep(0.01)

    client.send_command("StopRecording")
    client.shutdown()
    print("[MOTION] NatNet thread stopped")

# ==========================
# MAIN
# ==========================

threads = []

if record_audio:
    threads.append(threading.Thread(
        target=audio_thread_fn,
        daemon=True
    ))

if record_zed:
    threads.append(threading.Thread(
        target=zed_thread_fn,
        daemon=True
    ))

if record_motion:
    threads.append(threading.Thread(
        target=natnet_thread_fn,
        daemon=True
    ))

for t in threads:
    t.start()

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopping…")
    stop_event.set()


for t in threads:
    t.join()
    
time.sleep(1)
shutil.move(os.path.join(os.path.expanduser("~"), "Documents", "OptiTrack", "Default", f"{tak_filename}.tak"), f"data/takes/{tak_filename}.tak")
print("All recordings stopped cleanly.")
