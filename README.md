# Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection

This repository contains the software components developed as part of the master's thesis titled **"Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection"**, authored by **B.Sc. Máté Gábor Seidl** for obtaining the academic degree **Master of Science (M.Sc.)** at the **TUM School of Engineering and Design** of the **Technical University of Munich**.

## Thesis Supervision

The thesis was supervised by **M.Sc. Celina Dettmering** from the **Institute for Machine Tools and Industrial Management (iwb)** of the **Technical University of Munich**.

**Thesis Submission Date:** 30.11.2024

## Intellectual Property Notice

Through the supervision of **B.Sc. Máté Gábor Seidl** intellectual property of the Institute for Machine Tools and Industrial Management (iwb) is incorporated in this work. Publication of the work or transfer to third parties requires the permission of the chair holder. I agree to the archiving of the work in the library of the iwb, which is only accessible to iwb staff, as inventory and in the digital student research project database of the iwb as a PDF document.

## Installation

### Dependencies

To run this project, install the followings:

- **PortAudio**
  ```bash
  sudo apt-get install portaudio19-dev

- **Python 3 -> check if it is installed**
  ```bash
  python3 --version

### Python Virtual Environment (on Mac/Linux)
1. Create a project folder in a selected location and navigate into it in the terminal:
    ```bash
    mkdir <project_name>
    cd <project_name>

2. Create a Python 3 virtual environment:
    ```bash
    python3 -m venv venv

3. Activate the virtual environment:
    ```bash
    source venv/bin/activate

3. Copy the project files into the project folder

4. Setup required folder structure and install the required Python packages from the `requirements.txt` file in the project folder:
    ```bash
    python setup.py

## Folder Structure
```
<Project Folder Name>/
└─── 01_Data/
    └─── 01_audioDatasets/ (excluded from git)
        ├─── 01_ethernet_without_additional_noise
        ├─── 02_ethernet_with_additional_noise
        ├─── ...
        ├─── 07_noise_samples
        ├─── [New dataset folders go here]
    └─── clickDatasetPreprocessing.ipynb (current notebook)
└─── ...
```

## References

- **Librosa Documentation:**
  - [Mel Frequencies](https://librosa.org/doc/main/generated/librosa.mel_frequencies.html)
  - [Power to Decibel Conversion](https://librosa.org/doc/main/generated/librosa.power_to_db.html)
  - [Power to Decibel Implementation](https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#power_to_db)

- **Thery and Implementation Examples:**
  - [Live Spectrogram Generation](https://github.com/srrtth/Spectrogram-generator-from-live-audio/blob/main/livegram.py)
  - [Real-Time Sound Event Detection Using Spectrogram - SED During Badminton Play](https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb)
  - [Theory of Spectrogram and Mel Spectrogram and Visualization Using Python](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
  - [Audio Labs Erlangen - Audio Visualisation Using Python](https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html)

- **Variational Autoencoder implementations for data generation:**
  - https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb 
  - https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
  - https://www.tensorflow.org/tutorials/generative/cvae
  - https://www.kaggle.com/code/darkrubiks/variational-autoencoder-with-pytorch

