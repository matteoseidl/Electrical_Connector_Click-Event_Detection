import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import time
from matplotlib.colors import LogNorm
from scipy import signal

# https://github.com/srrtth/Spectrogram-generator-from-live-audio/blob/main/livegram.py


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

        self.samples_per_plot = int((click_sense.chunk * self.chunks_per_plot))

        #initialize spectrogram with zeros
        #self.init_spec = np.zeros((128, int(self.samples_per_plot/self.hop_length)))

        self.init_audio = np.zeros(self.samples_per_plot)

         # initialize mic_input buffer with zeros
        self.mic_buffer = self.init_audio

        self.frequencies, self.times, self.Sxx = signal.spectrogram(self.init_audio, self.sr)
        self.ax.pcolormesh(self.times, self.frequencies, 10 * np.log10(self.Sxx), shading='auto', cmap='inferno')

        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_xlabel('Time [s]')

        #self.colorbar = self.fig.colorbar(self.melspec_dB_img, ax=self.ax, format="%+2.0f dB")
        self.ax.set(title='Mel Spectrogram')

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True) # interval in milliseconds

        plt.show()

    def next_power_of_2(self, x):
        return 2**(math.ceil(math.log(x, 2)))

    def update(self, frame):
        #mic_input = self.click_sense.get_mic_input_spec()
        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk)

        mic_input = np.array(mic_input)
        self.mic_buffer = np.roll(self.mic_buffer, -mic_input.shape[0], axis=0)
        self.mic_buffer[-mic_input.shape[0]:] = mic_input

        #print(f"mic_input shape: {mic_input.shape}")

        self.frequencies, self.times, self.Sxx = signal.spectrogram(self.mic_buffer, self.sr)
        self.spec = self.ax.pcolormesh(self.times, self.frequencies, 10 * np.log10(self.Sxx), shading='auto', cmap='inferno')
        
        return self.spec,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)