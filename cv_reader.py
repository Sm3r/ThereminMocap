import csv
import time
from pathlib import Path

import serial
from serial.tools import list_ports


def find_crow_port(port=None):
    if port:
        return port

    ports = list(list_ports.comports())

    for p in ports:
        text = " ".join(
            str(x) for x in [p.device, p.description, p.manufacturer, p.product]
        )
        if "crow" in text.lower() or "monome" in text.lower():
            return p.device

    print("Could not auto-detect Crow.")
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device}: {p.description}")

    raise RuntimeError("Crow serial port not found.")


def main():
    port = find_crow_port()
    output_path = Path("crow_cv_log.csv")

    with serial.Serial(port, 115200, timeout=1) as ser:
        time.sleep(0.5)
        ser.reset_input_buffer()

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "crow_frame",
                "input1_volts",
                "input2_volts",
                "input1_norm_0_1",
                "input2_norm_0_1",
            ])

            print(f"Recording from {port}")
            print(f"Writing to {output_path}")
            print("Press Ctrl+C to stop.")

            try:
                while True:
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

            except KeyboardInterrupt:
                print("Stopped.")


if __name__ == "__main__":
    main()