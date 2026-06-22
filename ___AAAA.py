def cv_thread_fn():
    print("[CV] Initializing")

    port = config.crow_port
    if not port:
        print("[CV] No crow_port set in config, skipping CV recording")
        return

    # Choose the CV recording rate.
    # Use config.rates.cv_fps if you have it.
    # Otherwise, use the ZED fps so CV rows match camera-frame cadence.
    target_fps = getattr(config.rates, "cv_fps", config.rates.zed_fps)
    sample_period_ns = int(1_000_000_000 / target_fps)

    ser = serial.Serial(port, 115200, timeout=0)
    time.sleep(0.5)

    latest_cv = None
    latest_serial_time_ns = None
    serial_lines_read = 0
    serial_lines_bad = 0

    def parse_cv_line(line):
        parts = line.split(",")
        if len(parts) != 5:
            return None

        tag, v1, v2, n1, n2 = parts
        if tag != "cv":
            return None

        try:
            return (
                float(v1),
                float(v2),
                float(n1),
                float(n2),
            )
        except ValueError:
            return None

    def drain_serial():
        nonlocal latest_cv
        nonlocal latest_serial_time_ns
        nonlocal serial_lines_read
        nonlocal serial_lines_bad

        while ser.in_waiting > 0:
            raw = ser.readline()
            if not raw:
                break

            now_ns = time.perf_counter_ns()
            line = raw.decode("utf-8", errors="replace").strip()

            parsed = parse_cv_line(line)
            if parsed is None:
                serial_lines_bad += 1
                continue

            latest_cv = parsed
            latest_serial_time_ns = now_ns
            serial_lines_read += 1

    with open(cv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cv_frame",
            "host_time_ns",
            "rel_time_s",
            "volume_volts",
            "pitch_volts",
            "volume_norm_volts",
            "pitch_norm_volts",
            "source_age_ms",
            "valid",
        ])

        print(f"[CV] Connected to {port}")
        print(f"[CV] Target recording rate: {target_fps} Hz")
        print("[CV] Ready, waiting for all streams…")

        try:
            start_barrier.wait()
        except threading.BrokenBarrierError:
            print("[CV] Barrier broken, aborting.")
            ser.close()
            return

        # Discard all old Crow messages emitted before recording start.
        ser.reset_input_buffer()

        print("[CV] Recording")

        cv_frame = 0
        next_sample_ns = time.perf_counter_ns()

        try:
            while not stop_event.is_set():
                # Always keep the newest incoming Crow value.
                drain_serial()

                now_ns = time.perf_counter_ns()

                if now_ns < next_sample_ns:
                    sleep_s = (next_sample_ns - now_ns) * 1e-9
                    time.sleep(min(sleep_s, 0.001))
                    continue

                sample_time_ns = now_ns

                if recording_start_ns is not None:
                    rel_time_s = (sample_time_ns - recording_start_ns) * 1e-9
                else:
                    rel_time_s = sample_time_ns * 1e-9

                if latest_cv is None:
                    writer.writerow([
                        cv_frame,
                        sample_time_ns,
                        rel_time_s,
                        "",
                        "",
                        "",
                        "",
                        "",
                        0,
                    ])
                else:
                    v1, v2, n1, n2 = latest_cv
                    source_age_ms = (
                        (sample_time_ns - latest_serial_time_ns) * 1e-6
                        if latest_serial_time_ns is not None
                        else ""
                    )

                    writer.writerow([
                        cv_frame,
                        sample_time_ns,
                        rel_time_s,
                        v1,
                        v2,
                        n1,
                        n2,
                        source_age_ms,
                        1,
                    ])

                cv_frame += 1
                next_sample_ns += sample_period_ns

                # If Python got delayed badly, do not try to "catch up"
                # by writing a burst of old samples.
                if next_sample_ns < time.perf_counter_ns() - sample_period_ns:
                    next_sample_ns = time.perf_counter_ns() + sample_period_ns

        finally:
            ser.close()

    print(f"[CV] Stopped — {cv_frame} synchronized CV frames written")
    print(f"[CV] Serial lines read: {serial_lines_read}")
    print(f"[CV] Bad serial lines: {serial_lines_bad}")