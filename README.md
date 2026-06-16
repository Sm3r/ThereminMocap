# ThereminMocap

Motion capture theremin system using machine learning to map hand movements to theremin audio parameters.

## Input data

### Dataset recording setup

- MOCAP: 8 Optitrack cameras motion capture system
- Stereo cameras: 2 ZED i2 cameras 
- Single RGB camera: 2MP webcam
- Thereming outputtting continuous voltage values for Pitch (V/Oct) and Volume

### Training settings
- 2 recordings one per [hand-antenna-pitch/volume value] using each input source defined above

## Project Structure

```
ThereminMocap/
├── config.json              
├── requirements.txt         
├── data/
│   ├── features/         
│   └── recordings/             
├── out/
│   ├── train/         
│   └── test/           
├── utils/
│   ├── cleaning.py         
│   ├── config.py           
│   ├── mocap_parser.py
│   ├── plotter.py
│   └── tak_to_csv.py
├── train/
│   ├── split_dataset.py         
│   ├── data_loader.py           
│   ├── network.py
│   └── train.py
├── multirecorder.py        # Mocap + ZED + Audio recorder
├── prepare_mocap.py        # Mocap preprocessing from TAK to NPY
├── prepare_audio.py        # Audio preprocessing from WAV to NPY
├── main.py                 # Training script
├── evaluate.py             # Evaluation script
└── playback_osc.py         # Playback for the theremin inferred parameters
```

## Running the project
### Windows Setup Instructions:


1. **Install and run WSL**

2. **Install Python 3.10**
   ```bash
   sudo apt update && sudo apt upgrade
   sudo apt install python3.10
   ```

3. **Install build dependencies for Essentia**
   ```bash
   sudo apt install build-essential libyaml-dev libfftw3-dev \
       libavcodec-dev libavformat-dev libavutil-dev \
       libavresample-dev libsamplerate0-dev libtag1-dev \
       libchromaprint-dev python3-numpy-dev python3-yaml -y
   ```

4. **Create and activate virtual environment, install dependencies**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Pipeline

#### Data Acquisition
1. **Edit the config file with desired naming**
2. **Record mocap, ZED and audio data**
    ```bash
    python3 multirecorder.py
    ```
3. **Save your Takes!**

    Go to the recordings folder, zip the takes and save them somewere safe.
## 

#### Data Preparation

4. **Manual motive take preparation**
    - Open Motive, load your take, and trim it to remove noisy frames at start and end.
    - Create one rigid body per antenna naming it as the config name you choose.
    - Create one markerset per hand naming it as the config name you choose.
    - Train the markerset
    - Go in to the labelling section and manually relabel the lost markers with the quick label tool.
    - Solve the rigid bodies.
    - Save the solved take as a .tak file in the recordings folder.

5. **Process data**

    - **<u>Theremin:</u>**
        ```bash
        python3 prepare_audio.py
        ```

    - **<u>ZED:</u>**
        ```bash
        python3 prepare_zed.py
        ```
    
    - **<u>Mocap:</u>**
        ```bash
        python3 prepare_mocap.py
        ```
        This script will:
        - Convert the tak file to CSV using the compiled file converter.
        - Clean the CSV file removing unwanted columns.
        - Parse the remaining bodies and markers from CSV file to classes.
        - Export them and save in a .npy file
    
  

### Training

6. **Train the neural network**:
   ```bash
   python3 -m train.train
   ```

### Evaluation

7. **Test the trained model**:
   ```bash
   python3 evaluate.py
   ```


