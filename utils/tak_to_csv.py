import subprocess
import os
import platform
from .config import config
from pathlib import Path, PureWindowsPath


def _to_windows_path(path: Path) -> str:
    """Convert a Path object to Windows format, handling WSL paths."""
    path_str = str(path)
    
    # If running on WSL, convert /mnt/c/... to C:\...
    if platform.system() == "Linux" and "/mnt/" in path_str:
        # Convert /mnt/c/Users/... to C:\Users\...
        parts = path_str.split("/")
        if len(parts) > 2 and parts[1] == "mnt":
            drive = parts[2].upper()
            remaining = "\\".join(parts[3:])
            return f"{drive}:\\{remaining}"
    
    return path_str


def convert_tak_to_csv(take_name: str = None):

    if take_name is None:
        take_name = config.take_name
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    tak_file = project_root / "data" / "takes" / f"{take_name}.tak"
    csv_file = project_root / "data" / "dataframes" / f"MOCAP_{take_name}.csv"
    converter_exe = script_dir / "optitrack-motive-file-converter" / "bin" / "Release" / "net9.0" / "converter.exe"
    
    if not converter_exe.exists():
        raise FileNotFoundError(f"Converter executable not found: {converter_exe}")
    
    # Convert paths to Windows format for the Windows executable
    converter_exe_win = _to_windows_path(converter_exe)
    tak_file_win = _to_windows_path(tak_file)
    csv_file_win = _to_windows_path(csv_file)
    
    print(f"Converting {tak_file} to CSV...")
    
    # If on Linux (WSL), run through cmd.exe; otherwise run directly
    if platform.system() == "Linux":
        # Run through Windows cmd.exe from WSL
        result = subprocess.run(
            ["cmd.exe", "/c", converter_exe_win, tak_file_win, csv_file_win, "0"],
            check=True
        )
    else:
        # Run directly on Windows
        result = subprocess.run(
            [converter_exe_win, tak_file_win, csv_file_win, "0"],
            check=True
        )
    
    print(f"Conversion complete: {csv_file}")
    return str(csv_file)
