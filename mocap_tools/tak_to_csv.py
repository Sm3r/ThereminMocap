import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import platform
from config import config
from pathlib import Path


def _to_windows_path(path: Path) -> str:
    path_str = str(path)
    if platform.system() == "Linux" and "/mnt/" in path_str:
        parts = path_str.split("/")
        if len(parts) > 2 and parts[1] == "mnt":
            drive = parts[2].upper()
            remaining = "\\".join(parts[3:])
            return f"{drive}:\\{remaining}"
    return path_str


def convert_tak_to_csv(take_name: str = None):
    if take_name is None:
        take_name = config.pitch_take_name

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    tak_file = project_root / "data" / "recordings" / f"{take_name}_solved.tak"
    csv_dir = project_root / "data" / "features" / "mocap"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / f"OPTITRACK_{take_name}_raw.csv"
    converter_exe = script_dir / "optitrack-motive-file-converter" / "bin" / "Release" / "net9.0" / "converter.exe"

    if not converter_exe.exists():
        raise FileNotFoundError(f"Converter executable not found: {converter_exe}")

    converter_exe_win = _to_windows_path(converter_exe)
    tak_file_win = _to_windows_path(tak_file)
    csv_file_win = _to_windows_path(csv_file)

    # print(f"Converting {tak_file} to CSV...")

    if platform.system() == "Linux":
        result = subprocess.run(
            ["cmd.exe", "/c", converter_exe_win, tak_file_win, csv_file_win, "0"],
            check=True
        )
    else:
        result = subprocess.run(
            [converter_exe_win, tak_file_win, csv_file_win, "0"],
            check=True
        )

    print(f"Conversion complete: {csv_file}")
    return str(csv_file)
