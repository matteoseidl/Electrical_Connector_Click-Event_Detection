# Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection

This repository contains the software components developed as part of the master's thesis titled **"Quality Control in Robotic Peg-in-Hole Assembly Using Acoustic Event Detection"**, authored by **B.Sc. Máté Gábor Seidl** for obtaining the academic degree **Master of Science (M.Sc.)** at the **TUM School of Engineering and Design** of the **Technical University of Munich**.

## Thesis Supervision

The thesis was supervised by **M.Sc. Celina Dettmering** from the **Institute for Machine Tools and Industrial Management (iwb)** of the Technical University of Munich.

**Thesis Submission Date:** 30.11.2024

## Intellectual Property Notice

Through the supervision of **B.Sc. Máté Gábor Seidl** intellectual property of the Institute for Machine Tools and Industrial Management (iwb) is incorporated in this work. Publication of the work or transfer to third parties requires the permission of the chair holder. I agree to the archiving of the work in the library of the iwb, which is only accessible to iwb staff, as inventory and in the digital student research project database of the iwb as a PDF document.

## Installation

### Dependencies

To run this project, ensure you have the following installed:

- **PortAudio**
  ```bash
  sudo apt-get install portaudio19-dev

### Python Virtual Environment

1. Create a Python 3 virtual environment:
   ```bash
   python3 -m venv venv

2. Activate the virtual environment (on Mac/Linux):
    ```bash
    source venv/bin/activate

3. Install the required Python packages:
    ```bash
    pip install pyaudio numpy matplotlib librosa PyQt5

## References

- **Librosa Documentation:**
  - [Mel Frequencies](https://librosa.org/doc/main/generated/librosa.mel_frequencies.html)
  - [Power to Decibel Conversion](https://librosa.org/doc/main/generated/librosa.power_to_db.html)
  - [Power to Decibel Implementation](https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#power_to_db)

- **Implementation Examples:**
  - [Live Spectrogram Generation](https://github.com/srrtth/Spectrogram-generator-from-live-audio/blob/main/livegram.py)
  - [Real-Time Sound Event Detection Using Spectrogram - SED During Badminton Play](https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb)
  - [Theory of Spectrogram and Mel Spectrogram and Visualization Using Python](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
  - [Audio Labs Erlangen Resources](https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html)

