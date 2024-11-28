# Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection

This repository contains the software components developed as part of the master's thesis titled **"Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection"**, authored by **B.Sc. Máté Gábor Seidl** for obtaining the academic degree **Master of Science (M.Sc.)** at the **TUM School of Engineering and Design** of the **Technical University of Munich**.

## Thesis Supervision

The thesis was supervised by **M.Sc. Celina Dettmering** from the **Institute for Machine Tools and Industrial Management (iwb)** of the **Technical University of Munich**.

**Thesis Submission Date:** 30.11.2024

## Intellectual Property Notice

Through the supervision of **B.Sc. Máté Gábor Seidl** intellectual property of the Institute for Machine Tools and Industrial Management (iwb) is incorporated in this work. Publication of the work or transfer to third parties requires the permission of the chair holder. I agree to the archiving of the work in the library of the iwb, which is only accessible to iwb staff, as inventory and in the digital student research project database of the iwb as a PDF document.

## Installation

To process audio input, install the following:

- **PortAudio**
  ```bash
  sudo apt-get install portaudio19-dev

The project was created using Python 3.12.5. Install it or check the installed python version.

- **Python 3 -> check if it is installed**
  ```bash
  python3 --version

### Create a Python Virtual Environment (on Mac/Linux)

1. Create a project folder in a selected location and navigate into it in the terminal:
    ```bash
    mkdir <project_name>
    cd <project_name>

2. Create a Python 3 virtual environment (with the name **venv** in this example):
    ```bash
    python3 -m venv venv

3. Activate the virtual environment:
    ```bash
    source venv/bin/activate

3. Copy the project files from the GitHub repository into the project folder (it can be done with *git clone* as well)

4. Setup required folder structure
    ```bash
    python setup.py

5. Check if pip3 installed, if not install it
    ```bash
    pip3 --version

6. Install the required Python packages
    ```bash
    pip3 install numpy matplotlib pandas scikit-learn torch torchvision torchaudio tqdm PyQt5

If there are problems with the python packages, check the **requirements.txt** file in the project folder containing all packages and version used in the project.

## Folder Structure
```
<Project Folder Name>
└─── 01_Dataset
    └─── 01_audioDatasets (excluded from git)
        ├─── 01_Ethernet
        ├─── 02_Ethernet_Test
        ├─── 03_HVA280
        ├─── 04_HVA280_Test
        ├─── 05_HVA630
        ├─── 06_HVA630_Test
        ├─── 07_Noise_Samples
        ├─── [New dataset folders can be placed here]
    └─── preprocessNoiseAudio.ipynb
    └─── preprocessSingleClickAudio.ipynb
    └─── preprocessSingleClickAudioTest.ipynb
    └─── preprocessTwoClickAudio.ipynb
    └─── preprocessTwoClickAudioTest.ipynb
└─── 02_Data_Augmentation
    └─── 01_augmentedDatasets (excluded from git)
        ├─── 01_Ethernet
        ├─── 02_HVA280
        ├─── 03_HVA630
        ├─── 04_Noise_Samples
        ├─── [New dataset folders can be placed here]
    └─── mixClicksWithGeneratedNoise.ipynb
    └─── mixClicksWithRecordedNose.ipynb
    └─── VAEforNoiseSampleGeneration.ipynb
└─── 03_Click_Detection_Model
    └─── 01_modelArchitectures (model architectures for each model types)
    └─── 02_savedWeights (excluded from git)
    └─── 03_trainingResults (excluded from git)
    └─── clickDetectionModelTraining.ipynb
└─── 04_Detection_Model_Test
    └─── 01_testResults (excluded from git)
    └─── detectionOnTestData.ipynb
└─── 05_Click_Detection_App
    └─── __pycache__ (excluded from git)
    └─── clickDetector.py
    └─── clickSenseGUI.py
    └─── clickSenseMain.py
    └─── visualizeAudioInputSpectrogram.py
└─── 06_Utilities
    └─── __pycache__ (excluded from git)
    └─── audioProcessing.py
    └─── sharedValues.py
    └─── spectrogramPlotting.py
```

## Folder Content Description (according to the **Table B.1** in the thesis)
- 01_Dataset
    - Folderscontainingaudiorecordingsformodel training and the **Test Dataset** for each connector type, including noise recordings.
    - Jupyternotebooksforpreprocessingtheaudio recordings and saving the labeled datasets into the corresponding folders.

- 02_Data_Augmentation
    - Jupyter notebooks for creating the augmented datasets.
    - Jupyter notebooks with the VAE model implementation for synthetic noise data generation.
    - Folders to store the augmented datasets.

- 03_Click_Detection_Model
    - Jupyter notebook for model training and hyperparameter tuning.
    -  Folder containing model architectures for different channel sizes and for the two different spectrogram chunk lengths.
    - Folders to store trained weights.

- 04_Detection_Model_Test
    - Jupyter notebook to simulate real-time detection and make predictions on the \textbf{Test Dataset}.
    - Folder to save test results.

- 05_Click_Detection_App
    - Python app for real-time click detection during electrical connector mating using a trained model.
    
- 05_Utilities
    - Functions for audio processing and visualization used in multiple modules in the project.
    - Shared values used in multiple modules in the project.


## Main References and Implementation Examples Used in the Project
(all accessed on 28.11.2024)

- **Librosa Documentation:**
  - [Mel Frequencies](https://librosa.org/doc/main/generated/librosa.mel_frequencies.html)
  - [Power to dB](https://librosa.org/doc/main/generated/librosa.power_to_db.html)
  - [Power to dB implementation](https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#power_to_db)

- **Sound Event Detection Theory and Implementation Examples:**
  - [Spectrogram-generator-from-live-audio/livegram.py](https://github.com/srrtth/Spectrogram-generator-from-live-audio/blob/main/livegram.py)
  - [machinehearing/handson/badminton/BadmintonSoundEvents.ipynb](https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb)
  - [Getting to Know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
  - [Python Visualization, Plotting a Spectrogram](https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html)

- **Variational Autoencoder implementations for data generation:**
  - [Pytorch-VAE-tutorial/01_Variational_AutoEncoder.ipynb](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb)
  - [pytorch-vae/vae-cnn.ipynb](https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb)
  - [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae)
  - [Variational Autoencoder with PyTorch](https://www.kaggle.com/code/darkrubiks/variational-autoencoder-with-pytorch)

- **ML model training**
  - [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
  - [Introduction to Deep Learning (I2DL 2023) course by Matthias Niessner (TU MUnich)](https://www.youtube.com/watch?v=piSpPhPmPBU&list=PLQ8Y4kIIbzy_pGm2QAwF625E6nmcRu2sU&pp=iAQB)

