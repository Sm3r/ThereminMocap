import glob

class Config: 
    def __init__(self):
        self.recording_name = None
    
    def set_recording_name(self, name):
        self.recording_name = name
    
    def check_files_exist(self):
        pattern = f"data/takes/{self.recording_name}.*"
        existing = glob.glob(pattern)
        
        if existing:
            return True
        return False

config = Config()

