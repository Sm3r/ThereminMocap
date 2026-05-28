import glob
import json
import os
from types import SimpleNamespace


class Config:
    default = "take1"

    _DEFAULT_NAMES = {
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'pitch_antenna': 'pitch',
        'volume_antenna': 'volume',
    }

    _DEFAULT_RATES = {
        'mocap_fps': 120,
        'target_fps': 60,
        'audio_sr': 44100,
        'audio_hop': 245,
        'zed_fps': 30,
    }

    def __init__(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(project_root, "config.json")

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.take_name = data.get('take_name', self.default)
                self.names = SimpleNamespace(**(data.get('names', self._DEFAULT_NAMES)))
                self.rates = SimpleNamespace(**(data.get('rates', self._DEFAULT_RATES)))
        else:
            self.take_name = self.default
            self.names = SimpleNamespace(**self._DEFAULT_NAMES)
            self.rates = SimpleNamespace(**self._DEFAULT_RATES)

    def set_take_name(self, name):
        self.take_name = name
        with open(self.config_file, 'w') as f:
            json.dump({
                'take_name': name,
                'names': {k: getattr(self.names, k) for k in self._DEFAULT_NAMES},
                'rates': {k: getattr(self.rates, k) for k in self._DEFAULT_RATES},
            }, f)

    def check_files_exist(self):
        pattern = f"data/takes/{self.take_name}.*"
        existing = glob.glob(pattern)
        return bool(existing)


config = Config()
