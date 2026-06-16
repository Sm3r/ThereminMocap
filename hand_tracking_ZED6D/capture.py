import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd

from .tracking import HandTracking
from .triangulation import stereo_detect
from .zed import Zed
from config import config
import pyzed.sl as sl


def _initialize_name_dict():
    name_dict = {'Frame': []}
    for hand in ['left', 'right']:
        for i in range(21):
            for axis in ['X', 'Y', 'Z']:
                name_dict[f"{hand}_{i:02d}_{axis}"] = []
    name_dict['left_2d_detected'] = []
    name_dict['right_2d_detected'] = []
    return name_dict


def capture_to_csv(filename=None, output_csv=None, window_title='Image',
                   timestamped=False, show_windows=False, print_fps=False,
                   fps=None, use_triangulation=None, stop_event=None):
    """Capture hand keypoints from a ZED camera or SVO and save to CSV.

    - `filename`: path to SVO file. If None, uses live camera.
    - `output_csv`: optional path to save CSV. If omitted, a sensible default is used.
    - `window_title`: title for the OpenCV window.
    - `timestamped`: when True and no explicit `output_csv` is given, create a timestamped live filename.
    - `show_windows`: when False, disable OpenCV windows and landmark drawing to improve FPS.
    - `print_fps`: when True and `show_windows` is False, print per-camera FPS to the terminal.
    - `use_triangulation`: when True, use stereo triangulation instead of ZED point cloud.
      Defaults to (config.depth_mode == "triangulation") when not passed.
    - `stop_event`: shared threading.Event for coordinated cancellation across threads.
      When set, the frame loop exits at the next iteration.
    """
    if use_triangulation is None:
        use_triangulation = (config.depth_mode == "triangulation")

    os.makedirs('data/features', exist_ok=True)

    detector = HandTracking(maxHands=1, detectionCon=0.2, trackCon=0.8, complexity=1, draw=show_windows)

    # If using SVO file, keep existing single-stream behavior
    if filename:
        cam = Zed(filename, fps=fps)
        cam.print_information()
        try:
            final_frame = cam.zed.get_svo_number_of_frames()
        except Exception:
            final_frame = float('inf')
        camera_params = cam.camera_params
        zed_list = [cam]
        live_mode = False
    else:
        # Live mode: detect number of connected ZED cameras and use up to 2
        live_mode = True
        zed_list = []

        for cam_index in (0, 1):
            try:
                cam_instance = Zed(None, fps=fps)
                zed_list.append(cam_instance)
                if len(zed_list) >= 2:
                    break
            except Exception:
                continue

        if len(zed_list) == 0:
            raise RuntimeError('No ZED cameras available for live capture')

        # Use camera params from each opened camera
        camera_params_list = [z.camera_params for z in zed_list]

        # Live streams are open indefinitely unless SVO provided
        final_frame = float('inf')

    columns = list(_initialize_name_dict().keys())
    rows = []

    # ensure camera params list is available for per-camera processing
    camera_params_list = [z.camera_params for z in zed_list]

    # create per-camera detectors so FPS and detection are tracked per stream
    if live_mode:
        detectors = [HandTracking(maxHands=1, detectionCon=0.2, trackCon=0.8, complexity=1, draw=show_windows) for _ in zed_list]
    else:
        detectors = [detector]

    # Separate detectors per stereo view for triangulation
    if use_triangulation:
        left_detectors = [HandTracking(maxHands=1, detectionCon=0.2, trackCon=0.8, complexity=1, draw=False) for _ in zed_list]
        right_detectors = [HandTracking(maxHands=1, detectionCon=0.2, trackCon=0.8, complexity=1, draw=False) for _ in zed_list]
    else:
        left_detectors = None
        right_detectors = None

    frame = 0
    first_print = True

    # Loop: grab from each opened camera and produce one row per frame containing all camera columns
    num_cams = len(zed_list)
    fps_values = [None] * num_cams

    use_parallel = live_mode and num_cams > 1 and not show_windows
    executor = ThreadPoolExecutor(max_workers=num_cams) if use_parallel else None

    def _process_camera(idx, active_cam):
        err = active_cam.zed.grab(active_cam.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            return {"idx": idx, "success": False}

        active_cam.get_image()

        img = active_cam.img.copy()
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        depth_img = active_cam.depth_img
        pcl = active_cam.point_cloud

        if show_windows:
            img_processed = detectors[idx].findHands(img)
        else:
            detectors[idx].findHands(img)
            img_processed = None

        camera_params = camera_params_list[idx]
        data_left, data_right = detectors[idx].findpostion(img, pcl, camera_params)

        # Determine MediaPipe 2D detection status (independent of depth success)
        mp_left = False
        mp_right = False
        if detectors[idx].results and detectors[idx].results.multi_hand_landmarks:
            for i, _ in enumerate(detectors[idx].results.multi_hand_landmarks):
                handedness = detectors[idx].results.multi_handedness[i].classification[0].index
                if handedness == 1:
                    mp_left = True
                else:
                    mp_right = True

        entry = {
            'img': img_processed,
            'left_data': data_left,
            'right_data': data_right,
            'mp_left_detected': mp_left,
            'mp_right_detected': mp_right,
        }
        fps_value = detectors[idx].get_fps() if (not show_windows and print_fps) else None
        return {"idx": idx, "success": True, "entry": entry, "fps": fps_value}

    def _process_camera_triangulation(idx, active_cam):
        err = active_cam.zed.grab(active_cam.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            return {"idx": idx, "success": False}

        active_cam.get_image()

        img_left = active_cam.img.copy()
        if img_left.ndim == 3 and img_left.shape[2] == 4:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2BGR)

        img_right = active_cam.img_right.copy()
        if img_right.ndim == 3 and img_right.shape[2] == 4:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

        result = stereo_detect(
            left_detectors[idx].hands, right_detectors[idx].hands,
            img_left, img_right,
            active_cam.cam_left, active_cam.cam_right,
            active_cam.stereo_transform,
        )

        if show_windows:
            detectors[idx].findHands(img_left)

        entry = {
            'img': img_left if show_windows else None,
            'left_data': result["left_data"],
            'right_data': result["right_data"],
            'mp_left_detected': result["mp_left_detected"],
            'mp_right_detected': result["mp_right_detected"],
        }
        fps_value = detectors[idx].get_fps() if (not show_windows and print_fps) else None
        return {"idx": idx, "success": True, "entry": entry, "fps": fps_value}

    interrupted = False
    try:
        while frame <= final_frame and not stop_event.is_set():
            any_success = False
            per_cam = [None] * num_cams

            process_fn = _process_camera_triangulation if use_triangulation else _process_camera

            # grab/process each camera once per frame
            if use_parallel:
                futures = [executor.submit(process_fn, idx, cam) for idx, cam in enumerate(zed_list)]
                for fut in futures:
                    result = fut.result()
                    if not result.get("success"):
                        continue
                    idx = result["idx"]
                    any_success = True
                    per_cam[idx] = result["entry"]
                    if print_fps:
                        fps_values[idx] = result["fps"]
            else:
                for idx, active_cam in enumerate(zed_list):
                    result = process_fn(idx, active_cam)
                    if not result.get("success"):
                        continue
                    any_success = True
                    entry = result["entry"]
                    per_cam[idx] = entry

                    if show_windows:
                        # overlay FPS for this camera stream
                        img_processed = detectors[idx].displayFPS(entry['img'])
                        cv2.imshow(f"{window_title}_{idx}", img_processed)
                    elif print_fps:
                        fps_values[idx] = result.get("fps")

            if not any_success:
                break

            # advance frame once and append a single combined row
            frame += 1
            if filename and not show_windows:
                print(f"\r  Processed {frame} / {final_frame} frames", end="", flush=True)
            row = {}
            row['Frame'] = frame

            if num_cams == 1:
                entry = per_cam[0]
                dl = entry['left_data'] if (entry is not None) else None
                dr = entry['right_data'] if (entry is not None) else None

                if isinstance(dl, (list, tuple)):
                    dl = np.array(dl)
                if isinstance(dr, (list, tuple)):
                    dr = np.array(dr)

                for i in range(21):
                    for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                        col = f"left_{i:02d}_{axis}"
                        if isinstance(dl, np.ndarray) and dl.shape == (21, 3):
                            row[col] = float(dl[i, axis_idx])
                        else:
                            row[col] = np.nan

                    for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                        col = f"right_{i:02d}_{axis}"
                        if isinstance(dr, np.ndarray) and dr.shape == (21, 3):
                            row[col] = float(dr[i, axis_idx])
                        else:
                            row[col] = np.nan

                row['left_2d_detected'] = int(entry.get('mp_left_detected', False))
                row['right_2d_detected'] = int(entry.get('mp_right_detected', False))

            rows.append(row)

            if show_windows:
                # Print detection and frame count on two fixed lines and refresh in-place
                # combine detector strings for all detectors
                detection = ' | '.join(getattr(d, 'detection_str', '') for d in detectors)
                frame_line = f"Frame count: {frame}" + (f" / {final_frame}" if filename else "")
                if not first_print:
                    print("\x1b[2A", end='')
                print(detection.ljust(80), flush=True)
                print(frame_line.ljust(80), flush=True)
                first_print = False

                # throttle to real-time when playing back an SVO with windows
                wait_ms = max(1, int(1000 / fps)) if (show_windows and fps) else 1
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
            elif print_fps:
                fps_parts = []
                for idx, fps in enumerate(fps_values):
                    if fps is None:
                        fps_parts.append(f"cam{idx}: --")
                    else:
                        fps_parts.append(f"cam{idx}: {fps:5.1f} fps")
                print("\r" + " | ".join(fps_parts).ljust(60), end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()

    if interrupted and print_fps:
        print("", flush=True)

    # Save final results (build DataFrame from row buffer)
    df = pd.DataFrame(rows, columns=columns)
    if output_csv is None:
        if filename:
            base = os.path.splitext(os.path.basename(filename))[0]
            output_csv = os.path.join('data', 'dataframes', base + '.csv')
        elif timestamped:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = os.path.join('data', 'dataframes', f'live_{timestamp}.csv')
        else:
            output_csv = os.path.join('data', 'dataframes', 'output.csv')

    df.to_csv(output_csv, index=False)
    print(f"\n\nResults saved to: {output_csv}")
    print(f"Total frames recorded: {frame}")
    cv2.destroyAllWindows()

    if executor is not None:
        executor.shutdown(wait=True)

    return output_csv, frame
