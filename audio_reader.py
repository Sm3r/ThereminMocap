import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
import librosa



'''load audio file
 channel[0] = pitch antenna
 channel[1] = volume antenna
 channel[2] = audio (ignora)
'''

# Load the audio file using librosa
audio_path = "data/takes/TestZED.wav"
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
hop_size = 441

# Extract pitches from both channels
pitch_extractor = es.PitchYin(frameSize=frame_size)
pitches = []
for channel in channels:
    channel_pitches = []
    for frame in es.FrameGenerator(channel, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        pitch, _ = pitch_extractor(frame)
        channel_pitches.append(pitch)
    pitches.append(np.array(channel_pitches))

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

# Save DataFrame to CSV
df.to_csv("data/dataframes/TestZED.csv", index=False)
