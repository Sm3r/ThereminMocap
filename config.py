import glob
import json
import os

class Config: 
    config_file = ".config.json"
    
    def __init__(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.recording_name = data.get('recording_name', 'take1')
        else:
            self.recording_name = 'take1'
    
    def set_recording_name(self, name):
        self.recording_name = name
        with open(self.config_file, 'w') as f:
            json.dump({'recording_name': name}, f)
    
    def check_files_exist(self):
        pattern = f"data/takes/{self.recording_name}.*"
        existing = glob.glob(pattern)
        
        if existing:
            return True
        return False

config = Config()

