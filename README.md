<div align="center">

## **Theremin Mocap**

</div>

## 📖 Introduction
Motion capture theremin system using machine learning to map hand movements to theremin audio parameters.

## 📂 Dataset

### Dataset recording setup

- MOCAP: 8 Optitrack cameras motion capture system
- Stereo cameras: 2 ZED i2 cameras 
- Single RGB camera: 2MP webcam
- Thereming outputtting continuous voltage values for Pitch (V/Oct) and Volume

### Training settings
- 2 recordings one per [hand-antenna-pitch/volume value] using each input source defined above

## 🛠️ Installation

1. **Install and run WSL**

    ```bash
   wsl
   ```

2. **Install Python 3.10**
   ```bash
   sudo apt update && sudo apt upgrade
   sudo apt install python3.10
   ```

3. **Env setup**
   ```bash
   python3.10 -m venv venv
   exit
   ```

   ```bash
   .\venv\bin\activate
   py -m pip install --upgrade pip
   py -m pip install -r requirements.txt
   ```

### 🚀 Usage

#### Data Acquisition
1. **Edit the config file with desired naming**
2. **Record mocap, ZED and audio data**
    ```bash
    python3 multi_recorder.py
    ```
3. **Save your Takes!**

    Go to the recordings folder, zip the takes and save them somewere safe.
## 

#### Data Preparation

4. **Manual motive take preparation**
    - Open Motive, load your take.
    - Create one rigid body per antenna, and camera
    - Create one markerset per hand
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
         This script will:
        - Swap misslabelled hand, remove unconsistencies.
        - Save joints in a .npy file

    - **<u>Mocap:</u>**
        ```bash
        python3 prepare_mocap.py
        ```
        This script will:
        - Convert the tak file to CSV using the compiled file converter.
        - Clean the CSV file removing unwanted columns.
        - Parse the remaining bodies and markers from CSV file to classes.
        - Export them and save in a .npy file.
    
  

### Training

6. **Train the neural network**:
   ```bash
   python3 -m train.train
   ```

### Inference



