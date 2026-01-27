# ThereminMocap

Motion capture theremin system using machine learning to map hand movements to theremin audio parameters.

## Data

Google Drive: https://drive.google.com/drive/folders/1FwwdydfsW5zwJmulHlqtQbMZtq1t1s7j?usp=share_link

Jupyter Notebook: https://colab.research.google.com/drive/1qxv-aSJIAnVRj09gXxxJ_weP0UZtmfzq?usp=sharing

## Project Structure

```
ThereminMocap/
├── config.json              
├── requirements.txt         
├── data/
│   ├── dataframes/         
│   └── takes/             
├── out/
│   └── train/             
├── utils/
│   ├── cleaning.py         
│   ├── config.py           
│   ├── mocap_parser.py     
│   └── tak_to_csv.py
├── multirecorder.py        # Mocap + ZED + Audio recorder
├── prepare_mocap.py        # Mocap preprocessing from TAK to NPY
├── prepare_audio.py        # Audio preprocessing from WAV to NPY
├── data_loader.py          # PyTorch dataset
├── network.py              # Neural network architecture
├── main.py                 # Training script
└── evaluate.py             # Evaluation script
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

    Go to the takes folder, zip the takes and save them somewere safe.
## 

#### Data Preparation

4. **Manual motive take preparation**
    - Open Motive, load your take, and trim it to remove noisy frames at start and end.
    - Create one rigid body per antenna naming it as the config name you choose.
    - Create one markerset per hand naming it as the config name you choose.
    - Train the markerset
    - Go in to the labelling section and manually relabel the lost markers with the quick label tool.
    - Solve the rigid bodies.
    - Export the take as a .tak file in the takes folder.

5. **Process mocap data**

    ```bash
    python3 prepare_mocap.py
    ```
    This script will:
    - Convert the tak file to CSV using the compiled file converter.
    - Clean the CSV file removing unwanted columns.
    - Parse the remaining bodies and markers from CSV file to classes.
    - Export them and save in a .npy file
    
  

6. **Process audio data**
   ```bash
   python3 prepare_audio.py
   ```
    This script will:
    - Load the recorded WAV audio file and split it in channels.
    - Extract pitch and loudness features from the pitch of the audio channels.
    - Save the features in a CSV and .npy file.

### Training

7. **Train the neural network**:
   ```bash
   python3 main.py
   ```

### Evaluation

8. **Test the trained model**:
   ```bash
   python3 evaluate.py
   ```


