# ThereminMocap

Motion capture theremin system.

## Data

Google Drive: https://drive.google.com/drive/folders/1FwwdydfsW5zwJmulHlqtQbMZtq1t1s7j?usp=share_link

## Pipeline

1. `multi_recorder.py` - Record mocap, ZED, and audio data
2. `prepare_audio.py` - Process audio features to NPY
3. `prepare_mocap.py` - Process mocap markers to NPY
4. `data_loader.py` - Load data for training
5. `main.py` - Train neural network
6. `evaluate.py` - Test trained model