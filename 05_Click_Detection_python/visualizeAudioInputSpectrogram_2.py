import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import time
from matplotlib.colors import LogNorm
from scipy import signal
import librosa

# https://github.com/srrtth/Spectrogram-generator-from-live-audio/blob/main/livegram.py
# https://librosa.org/doc/main/generated/librosa.power_to_db.html
# https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#power_to_db


class AudioSpectrogramPlotter2:
    def __init__(self, click_sense):
        self.click_sense = click_sense

        self.chunk_freq = click_sense.chunk/click_sense.sampling_rate_downsampled # time for one chunk with the sampling rate, in case of a chunk size of 2048 and sampling rate 16 kHz: 2048/16000 = 0.128 s, meaning that 0.128 s corresponds to one chunk
        self.chunks_per_plot = click_sense.chunks_per_plot # number of chunks to plot
        self.plot_update_freq = self.chunk_freq * 1000 # update the plot for every new chunk

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('clickSense - Spectrogram Plotter')

        self.sr = click_sense.sampling_rate_downsampled # sampling rate of the downsampled audio signal

        self.resolution = 0.032 # in seconds, resulting in 32 frames for the 1.024 s plot duration
        self.hop_length = int(self.resolution * click_sense.sampling_rate_downsampled) # hop_length is the number of samples between successive frames, 0.032s * 16000 1/s = 512 samples
        self.n_fft = self.next_power_of_2(self.hop_length) # n_fft is the number of samples in each window, 512 samples, next power of 2 is 1024
        #self.n_fft = 2048

        self.samples_per_plot = int((click_sense.chunk * self.chunks_per_plot))

        self.mic_buffer = np.zeros(self.samples_per_plot)

        self.mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=128)

        self.mel_spec_buffer = np.zeros((128, int(self.samples_per_plot / self.hop_length)))
        
        self.mel_spec_img = self.ax.pcolormesh(np.linspace(0, self.samples_per_plot / self.sr, self.mel_spec_buffer.shape[1]),
                                               np.linspace(0, self.sr // 2, 128), 
                                               self.mel_spec_buffer, shading='auto', cmap='inferno')
        
        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_xlabel('Time [s]')
        self.ax.set(title='Mel Spectrogram')

        # Create the animation object
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True)
        plt.show()

    def next_power_of_2(self, x):
        return 2**(math.ceil(math.log(x, 2)))
    
    def power_to_db(self, Sxx, ref, amin, top_db):
    
        """Sxx_dB = 10 * np.log10(np.maximum(Sxx, amin)) 
        Sxx_dB -= 10 * np.log10(ref)
        Sxx_dB = np.maximum(Sxx_dB, Sxx_dB.max() - top_db)"""

        Sxx_dB = 10 * np.log10((np.maximum(Sxx, amin))/ref)
        Sxx_dB = np.maximum(Sxx_dB, Sxx_dB.max() - top_db)
        
        return Sxx_dB


    def update(self, frame):
        #mic_input = self.click_sense.get_mic_input_spec()
        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk)

        # Update mic buffer by rolling left and adding new mic input at the end
        mic_input = np.array(mic_input)

        #self.mic_buffer = np.roll(self.mic_buffer, -mic_input.shape[0], axis=0)
        #self.mic_buffer[-mic_input.shape[0]:] = mic_input

        # Compute the spectrogram for only the new chunk of data
        frequencies, times, Sxx = signal.spectrogram(mic_input, self.sr, nperseg=self.n_fft, noverlap=self.hop_length // 2)
        
        mel_spec_chunk = np.dot(self.mel_filter, Sxx)

        # melspec_chunk  = librosa.feature.melspectrogram(y=mic_input, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        #mel_spec_chunk_dB = 10 * np.log10(np.maximum(mel_spec_chunk, 1e-10))
        mel_spec_chunk_dB = self.power_to_db(mel_spec_chunk, ref=1.0, amin=1e-10, top_db=80.0)

        #mel_spec_chunk_dB = librosa.power_to_db(melspec_chunk, ref=1.0, amin=1e-10, top_db=80.0)

        #avarage value of the last chunk from the mel_spec_buffer
        spec_last_chunk_avg_t = np.mean(self.mel_spec_buffer[:, -2*mel_spec_chunk_dB.shape[1]:])
        

        self.mel_spec_buffer = np.roll(self.mel_spec_buffer, -mel_spec_chunk_dB.shape[1], axis=1)
        self.mel_spec_buffer[:, -mel_spec_chunk_dB.shape[1]:] = mel_spec_chunk_dB
        print(f"melspec_dB shape: {self.mel_spec_buffer.shape}")

        spec_last_chunk_avg_t_minus_1 = np.mean(self.mel_spec_buffer[:, -(3*(mel_spec_chunk_dB.shape[1])):-mel_spec_chunk_dB.shape[1]])
        #spec_last_chunk_avg_t_minus_1 = np.mean(self.mel_spec_buffer[:, -mel_spec_chunk_dB.shape[1]:])

        if spec_last_chunk_avg_t_minus_1 == spec_last_chunk_avg_t:
            print("Old values remain the same")
        else:
            print("old values changed")

        # Update the pcolormesh plot with the new mel spectrogram
        self.mel_spec_img.set_array(self.mel_spec_buffer.ravel())
        self.mel_spec_img.set_clim(vmin=np.min(self.mel_spec_buffer), vmax=np.max(self.mel_spec_buffer))

        return self.mel_spec_img,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)