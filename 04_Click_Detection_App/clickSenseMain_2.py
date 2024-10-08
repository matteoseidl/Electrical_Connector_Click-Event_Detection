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

from visualizeAudioInputSpectrogram_2 import AudioSpectrogramPlotter2
from clickDetector_2 import ClickDetector2


sampling_rate_orig = 48000 # original sampling rate of the microphone, defined by using "system_profiler SPAudioDataType" in macOS terminal to list connected audio devices and their properties
channels = 1
#format = pyaudio.paInt16
format = pyaudio.paFloat32 # for librosa audio data must be floating-point

model_architectures_dir = "03_Click_Detection_Model/01_modelArchitectures"
selected_model = "ClickDetectorCNN_v1"
model_weights_path = "03_Click_Detection_Model/02_savedWeights/ethernet_det_model_1.pt"

class ClickSense2:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.sampling_rate_downsampled = int(sampling_rate_orig/3) # downsampled to 16 kHz to reduce computational load
        self.chunk = 2048 # number of data points to process at a time by pyaudio stream
        self.stream = self.p.open(format=format, channels=channels, rate=self.sampling_rate_downsampled, input=True, frames_per_buffer=self.chunk)
        self.audio_chapture = False
        self.mic_input = 0.0
        self.lock = threading.Lock()
        self.audio_data = None

        self.chunks_per_plot = 16
        self.mic_input_spec = np.zeros(self.chunk * self.chunks_per_plot) # initialize mic_input buffer with zeros for spectrogram plot

        self.time_old = None
        self.time_new = None

        self.model_architectures_dir = model_architectures_dir
        self.model_weights_path = model_weights_path
        self.selected_model = selected_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model = self.load_model()

        self.detector = ClickDetector2()
        self.model = self.detector.load_model(self.model_architectures_dir, self.selected_model, self.model_weights_path)

    def start_recording(self):
        self.audio_chapture = True
        
        while self.audio_chapture:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            #amplitude = np.linalg.norm(audio_data) / len(audio_data)
            mic_input = audio_data
            print("mic_input: ", mic_input)
            
            with self.lock:
                self.mic_input = mic_input
                self.audio_data = audio_data

                mic_input_arr = np.array(self.mic_input)
                self.mic_input_spec = np.roll(self.mic_input_spec, -mic_input_arr.shape[0], axis=0)
                self.mic_input_spec[-mic_input_arr.shape[0]:] = mic_input_arr
            
            #time.sleep(0.0001)

    def get_mic_input(self):
        with self.lock:
            return self.mic_input
        
    def get_mic_input_spec(self):
        
        self.time_new = time.time()

        with self.lock:

            if self.time_old is not None:
                time_diff = self.time_new - self.time_old
                #print(f"Time difference: {time_diff} seconds")

            self.time_old = self.time_new

            return self.mic_input_spec

    def stop_recording(self):
        self.audio_chapture = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("recording stopped")

    def load_model(self):
        current_file_path = os.path.abspath(__file__)
        current_file_parent_dir = dirname(current_file_path)
        print(f"current_file_parent_dir: {current_file_parent_dir}")
        project_dir = dirname(current_file_parent_dir)
        print(f"project_dir: {project_dir}")
        model_architectures_dir_path = os.path.join(project_dir, self.model_architectures_dir)
        model_weights = os.path.join(project_dir, self.model_weights_path)
        if os.path.exists(model_architectures_dir_path):
            sys.path.append(model_architectures_dir_path)
            model_module = importlib.import_module(self.selected_model)
            ClickDetectorCNN = getattr(model_module, 'ClickDetectorCNN') #access the ClickDetectorCNN class
            model = ClickDetectorCNN(input_channels=1, output_shape=1).to(self.device)
            if os.path.exists(model_weights):
                model.load_state_dict(torch.load(model_weights))
                model.to(self.device)
                print("Model weights are loaded")
                print(f"model: {model}")
                return model
            else:
                print("Model weights file does not exist")
        else:
            print("Model architectures directory does not exist")
        return None

def signal_handler(sig, frame, click_sense):
    click_sense.stop_recording()
    sys.exit(0)

if __name__ == '__main__':
    click_sense = ClickSense2()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, click_sense))

    audio_chapture_thread = threading.Thread(target=click_sense.start_recording)
    audio_chapture_thread.start()

    #anput("Press Enter to start plotting spectrogram...")

    audio_spectrogram_plotter = AudioSpectrogramPlotter2(click_sense)

    audio_chapture_thread.join()

