import pyaudio
import numpy as np
import signal
import sys
import threading
from visualizeAudioInputAmplitude import AudioAmplitudePlotter
from visualizeAudioInputSpectrogram import AudioSpectrogramPlotter
import time


sampling_rate_orig = 48000 # original sampling rate of the microphone, defined by using "system_profiler SPAudioDataType" in macOS terminal to list connected audio devices and their properties
channels = 1
#format = pyaudio.paInt16
format = pyaudio.paFloat32 # for librosa audio data must be floating-point

class ClickSense:
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

    def start_recording(self):
        self.audio_chapture = True
        
        while self.audio_chapture:
            data = self.stream.read(self.chunk)
            audio_data = np.frombuffer(data, dtype=np.float32)
            #amplitude = np.linalg.norm(audio_data) / len(audio_data)
            mic_input = audio_data
            
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

def signal_handler(sig, frame, click_sense):
    click_sense.stop_recording()
    sys.exit(0)

if __name__ == '__main__':
    click_sense = ClickSense()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, click_sense))

    audio_chapture_thread = threading.Thread(target=click_sense.start_recording)
    audio_chapture_thread.start()

    #audio_plotter = AudioAmplitudePlotter(click_sense)
    spectrogram_plotter = AudioSpectrogramPlotter(click_sense)

    audio_chapture_thread.join()
