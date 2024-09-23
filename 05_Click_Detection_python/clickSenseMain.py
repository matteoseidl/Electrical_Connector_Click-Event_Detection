import pyaudio
import numpy as np
import signal
import sys
import threading
from visualizeAudioInputAmplitude import AudioAmplitudePlotter
from visualizeAudioInputSpectrogram import AudioSpectrogramPlotter

# Constants
sampling_rate_orig = 48000
channels = 1
#format = pyaudio.paInt16
format = pyaudio.paFloat32 # for librosa audio data must be floating-point

class ClickSense:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.chunk = 2048
        self.sampling_rate_downsampled = int(sampling_rate_orig/3) # 16000
        self.stream = self.p.open(format=format, channels=channels, rate=self.sampling_rate_downsampled, input=True, frames_per_buffer=self.chunk)
        self.recording = False
        self.mic_input = 0.0
        self.lock = threading.Lock()
        self.audio_data = None

    def start_recording(self):
        self.recording = True
        
        while self.recording:
            data = self.stream.read(self.chunk)
            audio_data = np.frombuffer(data, dtype=np.float32)
            #amplitude = np.linalg.norm(audio_data) / len(audio_data)
            mic_input = audio_data
            
            with self.lock:
                self.mic_input = mic_input
                self.audio_data = audio_data
            
            #time.sleep(0.0001)  

    def get_mic_input(self):
        with self.lock:
            #return self.audio_data
            return self.mic_input

    def stop_recording(self):
        self.recording = False
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

    recording_thread = threading.Thread(target=click_sense.start_recording)
    recording_thread.start()

    audio_plotter = AudioAmplitudePlotter(click_sense)
    #spectrogram_plotter = AudioSpectrogramPlotter(click_sense)

    recording_thread.join()
