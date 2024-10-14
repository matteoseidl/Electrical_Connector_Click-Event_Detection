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

model_architectures_dir = "03_Click_Detection_Model/01_modelArchitectures"
selected_model = "ClickDetectorCNN_v1"
model_weights_path = "03_Click_Detection_Model/02_savedWeights/ethernet_det_model_1.pt"

class ClickSense:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.sampling_rate_downsampled = int(sampling_rate_orig/3)
        self.chunk = 2048
        self.stream = None
        self.audio_capture = False
        self.lock = threading.Lock()
        self.chunks_per_plot = 16
        self.mic_input_spec = np.zeros(self.chunk * self.chunks_per_plot)
        
        self.detector = ClickDetector()
        self.model = self.detector.load_model(model_architectures_dir, selected_model, model_weights_path)

    def start_recording(self):
        self.stream = self.p.open(
            format=format,
            channels=channels,
            rate=self.sampling_rate_downsampled,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.audio_capture = True
        
        while self.audio_capture:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                with self.lock:
                    self.mic_input = audio_data
                    #print(f"mic_input: {self.mic_input}")
                    mic_input_arr = np.array(audio_data)
                    self.mic_input_spec = np.roll(self.mic_input_spec, -mic_input_arr.shape[0])
                    self.mic_input_spec[-mic_input_arr.shape[0]:] = mic_input_arr
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

    def get_mic_input_spec(self):
        with self.lock:
            return self.mic_input_spec.copy()

