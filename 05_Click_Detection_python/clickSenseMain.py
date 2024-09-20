import pyaudio
import numpy as np
import signal
import sys
import threading
import time
from visulizeAudioInput import AudioPlotter 

# Constants
sampling_rate = 48000
channels = 1
chunk = 4096
format = pyaudio.paInt16

class ClickSense:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format, channels=channels, rate=sampling_rate, input=True, frames_per_buffer=chunk)
        self.recording = False
        self.amplitude = 0.0
        self.lock = threading.Lock()

    def start_recording(self):
        self.recording = True
        
        while self.recording:
            data = self.stream.read(chunk)
            audio_data = np.frombuffer(data, dtype=np.int16)
            amplitude = np.linalg.norm(audio_data) / len(audio_data)
            
            with self.lock:
                self.amplitude = amplitude
            
            time.sleep(0.05)  # 50 ms

    def get_amplitude(self):
        with self.lock:
            return self.amplitude

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

    audio_plotter = AudioPlotter(click_sense)

    recording_thread.join()
