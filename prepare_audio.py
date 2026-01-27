import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
import librosa
import os
from utils.config import config

take_name = config.take_name

# Load the audio file using librosa
audio_path = f"data/takes/{take_name}.wav"
audio, sample_rate = librosa.load(audio_path, sr=44100, mono=False)

# Split into 3 separate channels
pitch_antenna = audio[0, :]
volume_antenna = audio[1, :]
channels = [pitch_antenna, volume_antenna]

# Normalize audio to 0 dB
pitch_antenna = pitch_antenna / np.max(np.abs(pitch_antenna))
volume_antenna = volume_antenna / np.max(np.abs(volume_antenna))

# Frame settings
frame_size = 2048 # 4096
hop_size = 245

# Extract pitches from both channels
pitch_extractor = es.PitchYin(frameSize=frame_size)
pitches = []
confidences = []
for channel in channels:
    channel_pitches = []
    channel_confidences = []
    for frame in es.FrameGenerator(channel, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        pitch, confidence = pitch_extractor(frame)
        channel_pitches.append(pitch)
        channel_confidences.append(confidence)
    pitches.append(np.array(channel_pitches))
    confidences.append(np.array(channel_confidences))

theremin_pitch = pitches[0]
theremin_volume = pitches[1]

# Filter out unreliable pitch estimates and replace with average between first valid samples before and after
min_freq = 20.0
confidence_threshold = 0.5

for i in range(len(pitches)):
    pitch_data = pitches[i].copy()
    conf_data = confidences[i]
    
    # Create mask for valid pitches
    valid_mask = (pitch_data > min_freq) & (conf_data > confidence_threshold)
    
    if valid_mask.sum() > 0:
        # Interpolate invalid values
        for idx in range(len(pitch_data)):
            if not valid_mask[idx]:
                # Find nearest valid sample before
                before_val = None
                for j in range(idx - 1, -1, -1):
                    if valid_mask[j]:
                        before_val = pitch_data[j]
                        break
                
                # Find nearest valid sample after
                after_val = None
                for j in range(idx + 1, len(pitch_data)):
                    if valid_mask[j]:
                        after_val = pitch_data[j]
                        break
                
                # Replace with average of before and after
                if before_val is not None and after_val is not None:
                    pitch_data[idx] = (before_val + after_val) / 2.0
                elif before_val is not None:
                    pitch_data[idx] = before_val
                elif after_val is not None:
                    pitch_data[idx] = after_val
        
        pitches[i] = pitch_data

theremin_pitch = pitches[0]
theremin_volume = pitches[1]

# Normalize
theremin_pitch = (theremin_pitch - theremin_pitch.min()) / (theremin_pitch.max() - theremin_pitch.min())
theremin_volume = (theremin_volume - theremin_volume.min()) / (theremin_volume.max() - theremin_volume.min())

# Create DataFrame
df = pd.DataFrame({
    "Frame": np.arange(len(theremin_pitch)),
    "Pitch_CV": theremin_pitch,
    "Volume_CV": theremin_volume
})

print(df.head())

# Save the audio features
os.makedirs("data/dataframes", exist_ok=True)
df.to_csv(f"data/dataframes/CV_{take_name}.csv", index=False)

out_audio_feats = np.array([theremin_pitch, theremin_volume]).T
os.makedirs("out/train", exist_ok=True)
np.save(f"out/train/{take_name}_audio.npy", out_audio_feats)