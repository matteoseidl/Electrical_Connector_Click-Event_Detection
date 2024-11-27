import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa
import math
import time
from scipy import signal
import matplotlib.ticker as ticker

class AudioSpectrogramPlotter:
    def __init__(self, click_sense, fig, ax):
        self.click_sense = click_sense

        self.model = click_sense.model
        self.detector = click_sense.detector

        self.window_size = click_sense.window_size

        # setup plot parameters
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

        # same parameters as used for dataset generation and model training
        self.resolution = 0.016 # in seconds, resulting in 128 frames for the 2.048s plot duration
        self.hop_length = int(self.resolution * self.click_sense.sampling_rate_downsampled) # hop_length is the number of samples between successive frames, 0.016s * 32000 1/s = 512 samples
        self.n_fft = self.next_power_of_2(self.hop_length)

        self.dB_ref = 1e3 # reference value for dB conversion
        self.a_squere_min = 1e-12 # to avoid log(0)
        self.top_dB_abs = abs(10*np.log10(self.a_squere_min)) # maximum dB value -> 10*log(a_squere_min) = -120

        self.n_mels = 128
        self.f_min = 20
        self.f_max = 14000
        self.mel_filter_bank = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max, htk=True, norm=1)

        self.old_mic_input = np.zeros(self.chunk_size).astype(np.float32)
        
    def initialize_plot(self):
        self.chunks_per_plot = self.click_sense.chunks_per_plot
        self.plot_update_freq = self.chunk_freq * 1000 # in ms
        self.samples_per_plot = self.chunk_size * self.chunks_per_plot

        self.init_spec = np.zeros((self.n_mels, int(self.samples_per_plot / self.hop_length))) # initialize mel spectrogram
        self.melspec_full = self.init_spec

        # initialize mel spectrogram plot
        self.mel_spec_img = self.ax.pcolormesh(
            np.linspace(0, self.samples_per_plot / self.sr, self.init_spec.shape[1]),
            np.linspace(self.f_min, self.f_max, self.n_mels), 
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

        self.ax.set(title='Real-Time Mel-Scaled Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
    
    def power_mel_to_db(self, D_mel, a_squere_min, dB_ref):

        D_mel_dB = 10.0 * np.log10(np.maximum(a_squere_min, np.minimum(D_mel, dB_ref)/dB_ref))

        return D_mel_dB
    
    def process_audio_data(self, signal):
        #signal = np.pad(signal, (self.hop_length//2, self.hop_length//2), 'constant', constant_values=(0, 0))
        chunk_stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, center = True)
        D = np.abs(chunk_stft) ** 2
        D_mel = np.dot(self.mel_filter_bank, D)
        D_mel_dB = self.power_mel_to_db(D_mel, a_squere_min=self.a_squere_min, dB_ref=self.dB_ref)

        return D_mel_dB

    def update(self):
        new_mic_input = self.click_sense.get_mic_input()
        if new_mic_input is None:
            return

        mid_mic_signal = np.concatenate((self.old_mic_input[(self.chunk_size//2):], new_mic_input[:(self.chunk_size//2)]))

        
        # padding
        #new_mic_input_padded = np.pad(new_mic_input, (self.hop_length//2, self.hop_length//2), 'constant', constant_values=(0, 0))

        #mid_mic_signal_padded = np.pad(mid_mic_signal, (self.hop_length//2, self.hop_length//2), 'constant', constant_values=(0, 0))
        
        
        # process audio data
        D_mel_dB_new = self.process_audio_data(new_mic_input)
        D_mel_dB_mid = self.process_audio_data(mid_mic_signal)

        new_spectrogram_chunk = np.concatenate((D_mel_dB_mid[:, 4:5], D_mel_dB_new[:, 1:8]), axis=1)
        
        # update spectrogram
        self.update_spectrogram(new_spectrogram_chunk)
        
        # perform click detection
        self.detection_res = self.detect_click()

        self.old_mic_input = new_mic_input

        # return the detection result to the gui
        return self.detection_res

    def update_spectrogram(self, new_spectrogram_chunk):
        self.melspec_full = np.roll(self.melspec_full, -new_spectrogram_chunk.shape[1], axis=1)
        self.melspec_full[:, -new_spectrogram_chunk.shape[1]:] = new_spectrogram_chunk
        self.mel_spec_img.set_array(self.melspec_full.ravel())

    def detect_click(self):
        spectrogram_chunk = self.melspec_full[:, -self.window_size:] # take only the last 32 or 64 columns for detection from the fill 128 in the plot

        update_without_detection = None
        if self.window_size == 32:
            update_without_detection = 4
        elif self.window_size == 64:
            update_without_detection = 8

        if self.detection_counter >= update_without_detection: # wait for 4 plot updates at the beginning before starting detection

            spectrogram_chunk_norm = self.detector.normalize_spec_chunk(spectrogram_chunk) # normalize the spectrogram chunk
            spectrogram_chunk_tensor = self.detector.convert_to_torch_tensor(spectrogram_chunk_norm) # convert to torch tensor
            prediction = self.detector.detection(self.model, spectrogram_chunk_tensor) # perform detection
            
            if prediction == 1:
                self.click_detected = True

            if self.click_detected:
                print("Click detected!")
            else:
                print("No click detected.")

        self.detection_counter += 1

        return self.click_detected