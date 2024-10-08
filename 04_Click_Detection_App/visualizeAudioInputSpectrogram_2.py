import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa
import math
import time
from scipy import signal
import matplotlib.ticker as ticker

from clickDetector_2 import ClickDetector2

# https://librosa.org/doc/main/auto_examples/plot_patch_generation.html
# https://librosa.org/doc/main/generated/librosa.power_to_db.html
# https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb
# https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html
# https://librosa.org/doc/main/generated/librosa.mel_frequencies.html

class AudioSpectrogramPlotter2:
    def __init__(self, click_sense, fig, ax):
        self.click_sense = click_sense
        self.detector = ClickDetector2()

        self.click_detected = False
        self.model = click_sense.model

        self.chunk_size = click_sense.chunk

        self.chunk_freq = self.chunk_size/click_sense.sampling_rate_downsampled
        self.chunks_per_plot = click_sense.chunks_per_plot
        self.plot_update_freq = self.chunk_freq * 1000

        self.fig = fig
        self.ax = ax
        
        # Clear existing axes content
        self.ax.clear()

        self.sr = click_sense.sampling_rate_downsampled

        self.resolution = 0.016
        self.hop_length = int(self.resolution * click_sense.sampling_rate_downsampled)
        self.n_fft = self.next_power_of_2(self.hop_length)

        self.samples_per_plot = int((self.chunk_size * self.chunks_per_plot))

        self.n_mels = 128
        self.init_spec = np.zeros((self.n_mels, int(self.samples_per_plot / self.hop_length)))
        self.melspec_full = self.init_spec

        self.dB_ref = 1
        self.amin = 1e-12
        self.top_dB_abs = 120

        self.mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=128)

        self.time_old = time.time()

        #self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=False) # interval in milliseconds

        #plt.show()

    def next_power_of_2(self, x):
        next_power_of_two = 2**(math.ceil(math.log(x, 2)))

        if next_power_of_two == x:
            next_power_of_two = next_power_of_two*2

        return next_power_of_two
    
    def power_to_db(self, S_mel, amin, dB_ref):

        S_dB = 10.0 * np.log10(np.maximum(amin, S_mel))
        S_dB -= 10.0 * np.log10(np.maximum(amin, dB_ref))
        
        return S_dB
    
    def update(self, frame):
        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk * self.chunks_per_plot)

        print(f"mic_input shape: {mic_input.shape}")

        chunk_stft = librosa.stft(mic_input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)

        #power spectral density
        S = np.abs(chunk_stft) ** 2

        #mel scale
        mel_filter_bank = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, htk=True)
        S_mel = np.dot(mel_filter_bank, S)
    
        S_dB = self.power_to_db(S_mel, amin=self.amin, dB_ref=self.dB_ref)
    
        time_now = time.time()
        time_diff = time_now - self.time_old
        self.time_old = time_now

        self.melspec_full = np.roll( self.melspec_full, -S_dB.shape[1], axis=1)
        self.melspec_full[:, -S_dB.shape[1]:] = S_dB
        
        self.mel_spec_img.set_array(self.melspec_full.ravel())
        #print(f"mel_spec_img shape: {self.melspec_full.shape}"

        # input last 4 spectrogram chunks into the click detection model (total size: 128x32)

        spectrogram_chunk = self.melspec_full[:, -32:] # size: 128x32
        spectrogram_chunk_norm = self.detector.normalize_spec_chunk(spectrogram_chunk)
        spectrogram_chunk_tensor = self.detector.convert_to_torch_tensor(spectrogram_chunk_norm) # model inpur 1 x 1 x 128 x 32
        #print(f"spectrogram_chunk_tensor shape: {spectrogram_chunk_tensor.shape}")
        prediction = self.detector.detection(self.model, spectrogram_chunk_tensor)

        if prediction == 1 and frame > self.chunks_per_plot/4: # ignore first 4 frames, as it is from spectrogram initialization
            self.click_detected = True

        if self.click_detected:
            print("Click detected!")
        else:
            print("No click detected.")

        #return self.mel_spec_img,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)