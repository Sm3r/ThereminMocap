import glob
import json
import os

class Config: 
    config_file = ".config.json"
    default = "take1"
    
    def __init__(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.take_name = data.get('take_name', self.default)
        else:
            self.take_name = self.default
    
    def set_take_name(self, name):
        self.take_name = name
        with open(self.config_file, 'w') as f:
            json.dump({'take_name': name}, f)
    
    def check_files_exist(self):
        pattern = f"data/takes/{self.take_name}.*"
        existing = glob.glob(pattern)
        
        if existing:
            return True
        return False

config = Config()

