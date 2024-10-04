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
    def __init__(self, click_sense):
        self.click_sense = click_sense
        self.detector = ClickDetector()

        self.click_detected = False
        self.model = click_sense.model
        print(f"model: {self.model}")

        self.chunk_size = click_sense.chunk

        self.chunk_freq = self.chunk_size/click_sense.sampling_rate_downsampled # in case of a chunk size of 2048 and sampling rate 16 kHz: 2048/16000 = 0.128 s, meaning that 0.128 s corresponds to one chunk
        self.chunks_per_plot = click_sense.chunks_per_plot # number of chunks to plot
        self.plot_update_freq = self.chunk_freq * 1000 # plot update frequency in milliseconds, updating for every plot

        self.fig_x = 12
        self.fig_y = 6
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.fig_x, self.fig_y))
        self.fig.canvas.manager.set_window_title('clickSense - Spectrogram Plotter')

        self.sr = click_sense.sampling_rate_downsampled

        self.resolution = 0.016 # in seconds, resulting in 32 frames for the 1.024 s plot duration
        self.hop_length = int(self.resolution * click_sense.sampling_rate_downsampled) # hop_length is the number of samples between successive frames, 0.016s * 16000 1/s = 256 samples
        self.n_fft = self.next_power_of_2(self.hop_length) # n_fft is the number of samples in each window, 512 samples, next power of 2 is 1024

        self.samples_per_plot = int((self.chunk_size * self.chunks_per_plot))

        #initialize spectrogram with zeros
        self.n_mels = 128
        self.init_spec = np.zeros((self.n_mels, int(self.samples_per_plot / self.hop_length)))
        #print(f"init_spec shape: {self.init_spec.shape}")
        self.melspec_full = self.init_spec

        self.top_dB_abs = 100 # max abs decibel value for the color map
        self.dB_ref = 1e-12 # ref level
        
        self.mel_spec_img = self.ax.pcolormesh(np.linspace(0, self.samples_per_plot / self.sr, self.init_spec.shape[1]),
                                               np.linspace(0, self.sr // 2, self.n_mels), 
                                               self.init_spec, shading='auto', cmap='inferno')
        
        self.ax.set_xlim(0 - self.resolution/2, (self.chunk_size / self.sr) * self.chunks_per_plot + self.resolution/2)
        #self.x_min, self.x_max = self.ax.get_xlim()
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(base=self.chunk_freq))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))
        self.x_min, self.x_max = self.ax.get_xlim()
        self.x_min_thick = self.x_min + self.resolution/2
        self.x_max_thick = self.x_max - self.resolution/2
        
        self.mel_spec_img.set_clim(vmin=-self.top_dB_abs, vmax=self.dB_ref)

        self.colorbar = self.fig.colorbar(self.mel_spec_img, ax=self.ax, format="%+2.0f dB")
        self.colorbar.set_label("Decibels (dB)")

        self.mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=128)

        self.ax.set(title='Mel Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        self.time_old = time.time()

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=False) # interval in milliseconds

        plt.show()

    def next_power_of_2(self, x):
        next_power_of_two = 2**(math.ceil(math.log(x, 2)))

        if next_power_of_two == x:
            next_power_of_two = next_power_of_two*2

        return next_power_of_two
    
    def power_to_db(self, S_mel, amin, top_db):
    
        S_dB = 10 * np.log10(np.maximum(S_mel, amin)) 
        #Sxx_dB -= 10 * np.log10(ref)
        # print(f" Sxx_dB max: {Sxx_dB.max()}")
        S_dB_clipped = np.maximum(S_dB, S_dB.max() - top_db)
        
        return S_dB_clipped
    
    def update(self, frame):
        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk * self.chunks_per_plot)

        #print(f"mic_input shape: {mic_input.shape}")

        #melspec_chunk  = librosa.feature.melspectrogram(y=mic_input, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        #print(f"melspec shape: {melspec.shape}")

        chunk_stft = librosa.stft(mic_input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
        """print(f"chunk_stft shape: {chunk_stft.shape}")
        print(f"chunk_stft max, min: {chunk_stft.max(), chunk_stft.min()}")"""

        #power spectral density
        S = np.abs(chunk_stft) ** 2
        #S = np.abs(chunk_stft)

        #mel scale
        mel_filter_bank = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, htk=True)
        S_mel = np.dot(mel_filter_bank, S)

        #first mel scale
        #chunk_stft_mel = librosa.hz_to_mel(chunk_stft, htk=True)

        """print(f"mel shape: {S_mel.shape}")
        print(f"S_mel max, min: {S_mel.max(), S_mel.min()}")"""

        #convert to decibels
        S_dB = self.power_to_db(S_mel, amin=self.dB_ref, top_db=self.top_dB_abs)
        """print(f"S_dB shape: {S_dB.shape}")
        print(f"S_dB max, min: {S_dB.max(), S_dB.min()}")"""

        time_now = time.time()
        time_diff = time_now - self.time_old
        self.time_old = time_now

        self.melspec_full = np.roll( self.melspec_full, -S_dB.shape[1], axis=1)
        self.melspec_full[:, -S_dB.shape[1]:] = S_dB
        
        self.mel_spec_img.set_array(self.melspec_full.ravel())
        #print(f"mel_spec_img shape: {self.melspec_full.shape}")


        self.x_min += time_diff
        self.x_max += time_diff
        self.x_min_thick += self.chunk_freq
        self.x_max_thick += self.chunk_freq
        #self.x_min, self.x_max = self.ax.get_xlim()
        #print(f"x_min_thick: {self.x_min_thick}, x_max_thick: {self.x_max_thick}")
        
        #print(f"x_min: {self.x_min}, x_max: {self.x_max}")
        #print(f"frame: {frame}")
        #self.ax.set_xlim(self.x_min, self.x_max)

        new_ticks = np.arange(self.x_min_thick, self.x_max_thick, self.chunk_freq)
        #print(f"new_ticks min: {new_ticks.min()}")
        #print(f"new_ticks length: {len(new_ticks)}")
        #self.ax.set_xticks(new_ticks)

        # input last 4 spectrogram chunks into the click detection model (total size: 128x32)

        spectrogram_chunk = self.melspec_full[:, -32:] # size: 128x32
        spectrogram_chunk_norm = self.detector.normalize_spec_chunk(spectrogram_chunk)
        spectrogram_chunk_tensor = self.detector.convert_to_torch_tensor(spectrogram_chunk_norm) # model inpur 1 x 1 x 128 x 32
        #print(f"spectrogram_chunk_tensor shape: {spectrogram_chunk_tensor.shape}")
        prediction = self.detector.detection(self.model, spectrogram_chunk_tensor)

        #print(f"model_input shape: {model_input.shape}")
        #print(model_input.min(), model_input.max())

        if prediction == 1 and frame > self.chunks_per_plot/4: # ignore first 4 frames, as it is from spectrogram initialization
            self.click_detected = True

        if self.click_detected:
            print("Click detected!")
        else:
            print("No click detected.")

        return self.mel_spec_img,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)