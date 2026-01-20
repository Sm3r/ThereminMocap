Simply a mocap theremin.

Google Drive link to data/takes folder:
https://drive.google.com/drive/folders/1FwwdydfsW5zwJmulHlqtQbMZtq1t1s7j?usp=share_link


### Pipeline:
1. multirecorder.py - Records mocap, zed and audio data
2. audio_reader.py - Creates df out of audio data
3. preprocess_mocap.py - Creates df out of mocap data and combines with audio df
4. network.py - for training and testing the neural network

UPDATED:

natnet/multirecorder.py - for recording mocap, zed and audio data

audio_reader.py - for reading audio data and creating dataframe

TODO:

preprocess_mocap.py

network.py
