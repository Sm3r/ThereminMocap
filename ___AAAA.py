def cv_thread_fn():
    print("[CV] Initializing")

    port = config.crow_port
    if not port:
        print("[CV] No crow_port set in config, skipping CV recording")
        return

    ser = serial.Serial(port, 115200, timeout=0.05)
    time.sleep(0.5)

    with open(cv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cv_sample_idx",       # Python-side recording sample index
            "host_time_ns",
            "host_time_s",
            "crow_frame_raw",      # Crow's free-running frame counter
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

        # Critical: remove all CV lines emitted before synchronized start.
        ser.reset_input_buffer()

        print("[CV] Recording")

        cv_sample_idx = 0

        try:
            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue

                host_time_ns = time.perf_counter_ns()

                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("cv,"):
                    continue

                parts = line.split(",")
                if len(parts) != 6:
                    continue

                _, crow_frame_raw, v1, v2, n1, n2 = parts

                writer.writerow([
                    cv_sample_idx,
                    host_time_ns,
                    host_time_ns * 1e-9,
                    int(crow_frame_raw),
                    float(v1),
                    float(v2),
                    float(n1),
                    float(n2),
                ])

                cv_sample_idx += 1

        finally:
            ser.close()

    print(f"[CV] Stopped — {cv_sample_idx} samples written")