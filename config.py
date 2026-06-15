import glob
import json
import os
from types import SimpleNamespace


_VALID_DEPTH_MODES = ("point_cloud", "triangulation")


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
        'zed_fps': 60,
    }

    _DEFAULT_WEBCAM = {
        'index': 1,
    }

    _DEFAULT_ZED = {
        'svo_codec': 'H264',
    }

    def __init__(self):
        self.depth_mode = "point_cloud"
        self.webcam = SimpleNamespace(**self._DEFAULT_WEBCAM)
        self.zed = SimpleNamespace(**self._DEFAULT_ZED)

        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(project_root, "config.json")

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.take_name = data.get('take_name', self.default)
                self.names = SimpleNamespace(**(data.get('names', self._DEFAULT_NAMES)))
                self.rates = SimpleNamespace(**(data.get('rates', self._DEFAULT_RATES)))

            depth_mode = data.get('depth_mode', 'point_cloud')
            if depth_mode not in _VALID_DEPTH_MODES:
                raise ValueError(
                    f"Invalid depth_mode '{depth_mode}' in config.json. "
                    f"Must be one of {_VALID_DEPTH_MODES}"
                )
            self.depth_mode = depth_mode

            webcam_data = data.get('webcam', self._DEFAULT_WEBCAM)
            self.webcam = SimpleNamespace(**webcam_data)

            zed_data = data.get('zed', self._DEFAULT_ZED)
            self.zed = SimpleNamespace(**zed_data)
        else:
            self.take_name = self.default
            self.names = SimpleNamespace(**self._DEFAULT_NAMES)
            self.rates = SimpleNamespace(**self._DEFAULT_RATES)

    def set_take_name(self, name):
        self.take_name = name
        with open(self.config_file, 'w') as f:
            json.dump({
                'take_name': name,
                'depth_mode': self.depth_mode,
                'webcam': {
                    'index': self.webcam.index,
                },
                'zed': {
                    'svo_codec': self.zed.svo_codec,
                },
                'names': {k: getattr(self.names, k) for k in self._DEFAULT_NAMES},
                'rates': {k: getattr(self.rates, k) for k in self._DEFAULT_RATES},
            }, f)

    def check_files_exist(self):
        pattern = f"data/takes/{self.take_name}.*"
        existing = glob.glob(pattern)
        return bool(existing)


config = Config()
