import pyaudio
import numpy as np
import signal
import sys
import threading
import time
import sys
import os
import torch
from torch import nn
from os.path import dirname, abspath
from pathlib import Path
import importlib

from visualizeAudioInputSpectrogram import AudioSpectrogramPlotter
from clickDetector import ClickDetector


sampling_rate_orig = 48000 # original sampling rate of the microphone, defined by using "system_profiler SPAudioDataType" in macOS terminal to list connected audio devices and their properties
channels = 1
format = pyaudio.paFloat32 # for librosa audio data must be floating-point

long_window = True

# path to the model architecture and weights
model_architectures_dir = "03_Click_Detection_Model/01_modelArchitectures"

if long_window:
    selected_model = "ClickDetectorCNN_v1_long_window"
else:
    selected_model = "ClickDetectorCNN_v1"

model_weights_path = "03_Click_Detection_Model/02_savedWeights/hva280_det_model_0.pt"

class ClickSense:
    def __init__(self):
        self.p = pyaudio.PyAudio() # instantiate PyAudio for audio capture

        # constant parameters for audio capture, same as used audio preprocessing and training data generation
        self.sampling_rate_downsampled = 32000
        self.chunk = 4096

        self.stream = None
        self.audio_capture = False
        self.lock = threading.Lock() # create lock object

        # number of chunks protted in one plot --> 16 chunks -> 16*4096 samples = 65536 samples = 2.048 seconds at 32kHz sampling rate
        self.chunks_per_plot = 16
        #self.mic_input_spec = np.zeros(self.chunk * self.chunks_per_plot)
        
        self.detector = ClickDetector() # instantiate ClickDetector class
        self.model = self.detector.load_model(model_architectures_dir, selected_model, model_weights_path) # load model with weights

        if long_window:
            self.window_size = 64
        else:
            self.window_size = 32

    def start_recording(self):
        self.stream = self.p.open(
            format=format,
            channels=channels,
            rate=self.sampling_rate_downsampled,
            input=True,
            frames_per_buffer=self.chunk # number of samples processed at a time
        )
        self.audio_capture = True
        
        while self.audio_capture:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=True)
                audio_data = np.frombuffer(data, dtype=np.float32) # convert audio input data to numpy array
                
                with self.lock:
                    self.mic_input = audio_data

            except Exception as e:
                print(f"Error in audio capture: {e}")
                break

    def stop_recording(self):
        self.audio_capture = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Recording stopped")

    def get_mic_input(self):
        with self.lock:
            return self.mic_input.copy() if hasattr(self, 'mic_input') else None

