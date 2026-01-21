import subprocess
from config import config
from pathlib import Path

take_name = config.take_name
TAK_FILE = rf"data\takes\{take_name}.tak"
CSV_FILE = rf"data\dataframes\{take_name}.csv"

subprocess.run([r"optitrack-motive-file-converter\bin\Release\net9.0\converter.exe", TAK_FILE, CSV_FILE, "0"])
