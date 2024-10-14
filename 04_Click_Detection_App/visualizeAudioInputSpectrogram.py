import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa
import math
import time
from scipy import signal
import matplotlib.ticker as ticker

from clickDetector import ClickDetector

# https://librosa.org/doc/main/auto_examples/plot_patch_generation.html
# https://librosa.org/doc/main/generated/librosa.power_to_db.html
# https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb
# https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html
# https://librosa.org/doc/main/generated/librosa.mel_frequencies.html

class AudioSpectrogramPlotter:
    def __init__(self, click_sense, fig, ax):
        self.click_sense = click_sense
        self.model = click_sense.model

        self.detector = ClickDetector()

        self.fig = fig
        self.ax = ax

        self.detection_counter = 0
        self.click_detected = False
        
        # setup parameters
        self.setup_parameters()
        
        # initialize the plot
        self.initialize_plot()

    def next_power_of_2(self, x):
        next_power_of_two = 2**(math.ceil(math.log(x, 2)))

        if next_power_of_two == x:
            next_power_of_two = next_power_of_two*2

        return next_power_of_two
        
    def setup_parameters(self):
        self.chunk_size = self.click_sense.chunk
        self.sr = self.click_sense.sampling_rate_downsampled
        self.chunk_freq = self.chunk_size / self.sr
        self.sr = self.click_sense.sampling_rate_downsampled
        self.resolution = 0.016 # in seconds, resulting in 32 frames for the 1.024 s plot duration
        self.hop_length = int(self.resolution * self.click_sense.sampling_rate_downsampled) # hop_length is the number of samples between successive frames, 0.016s * 16000 1/s = 256 samples
        self.n_fft = self.next_power_of_2(self.hop_length)
        self.dB_ref = 1 # reference value for dB conversion, log(1) = 0
        self.amin = 1e-12 # to avoid log(0)
        self.top_dB_abs = 120 # maximum dB value -> 10*log(amin) = -120
        self.n_mels = 128
        self.mel_filter_bank = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, htk=True)
        

    def initialize_plot(self):
        self.chunks_per_plot = self.click_sense.chunks_per_plot
        self.plot_update_freq = self.chunk_freq * 1000
        self.samples_per_plot = self.chunk_size * self.chunks_per_plot
        self.init_spec = np.zeros((self.n_mels, int(self.samples_per_plot / self.hop_length)))
        self.melspec_full = self.init_spec

        # initialize mel spectrogram plot
        self.mel_spec_img = self.ax.pcolormesh(
            np.linspace(0, self.samples_per_plot / self.sr, self.init_spec.shape[1]),
            np.linspace(0, self.sr // 2, self.n_mels), 
            self.init_spec, shading='auto', cmap='inferno'
        )

        self.ax.set_xlim(0 - self.resolution/2, (self.chunk_size / self.sr) * self.chunks_per_plot + self.resolution/2)

        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(base=self.chunk_freq))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))
        self.x_min, self.x_max = self.ax.get_xlim()
        self.x_min_thick = self.x_min + self.resolution/2
        self.x_max_thick = self.x_max - self.resolution/2
        
        self.mel_spec_img.set_clim(vmin=-self.top_dB_abs, vmax=0)

        self.colorbar = self.fig.colorbar(self.mel_spec_img, ax=self.ax, format="%+2.0f dB")
        self.colorbar.set_label("Decibels (dB)")

        self.ax.set(title='Mel Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

    def power_to_db(self, S_mel, amin, dB_ref):

        S_dB = 10.0 * np.log10(np.maximum(amin, S_mel))
        S_dB -= 10.0 * np.log10(np.maximum(amin, dB_ref))
        
        return S_dB

    def update(self):
        mic_input = self.click_sense.get_mic_input()
        if mic_input is None:
            return
        
        # process audio data
        S_dB = self.process_audio_data(mic_input)
        
        # update spectrogram
        self.update_spectrogram(S_dB)
        
        # perform click detection
        self.detection_res = self.detect_click()

        return self.detection_res

    def process_audio_data(self, mic_input):
        chunk_stft = librosa.stft(mic_input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
        S = np.abs(chunk_stft) ** 2
        S_mel = np.dot(self.mel_filter_bank, S)
        S_dB = self.power_to_db(S_mel, amin=self.amin, dB_ref=self.dB_ref)

        return S_dB

    def update_spectrogram(self, S_dB):
        self.melspec_full = np.roll(self.melspec_full, -S_dB.shape[1], axis=1)
        self.melspec_full[:, -S_dB.shape[1]:] = S_dB
        self.mel_spec_img.set_array(self.melspec_full.ravel())

    def detect_click(self):
        spectrogram_chunk = self.melspec_full[:, -32:]

        if self.detection_counter > self.chunks_per_plot/4:

            spectrogram_chunk_norm = self.detector.normalize_spec_chunk(spectrogram_chunk)
            spectrogram_chunk_tensor = self.detector.convert_to_torch_tensor(spectrogram_chunk_norm)
            prediction = self.detector.detection(self.model, spectrogram_chunk_tensor)
            
            if prediction == 1:
                self.click_detected = True

            if self.click_detected:
                print("Click detected!")
            else:
                print("No click detected.")

        self.detection_counter += 1

        return self.click_detected