import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import time
from matplotlib.colors import LogNorm
from scipy import signal
import librosa
from matplotlib.mlab import window_hanning,specgram

# https://github.com/ayared/Live-Specgram/blob/master/run_specgram.py


class AudioSpectrogramPlotter3:
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

        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk)

        arr2D,freqs,bins = self.get_specgram(mic_input,self.sr)

        extent = (bins[0],bins[-1]*self.chunks_per_plot, freqs[-1],freqs[0])
        self.im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                    cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
        
        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_xlabel('Time [s]')
        self.ax.set(title='Mel Spectrogram')
        plt.gca().invert_yaxis()

        # Create the animation object
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True)
        
        try:
            plt.show()
        except:
            print("Plot Closed")

    def next_power_of_2(self, x):
        return 2**(math.ceil(math.log(x, 2)))
    
    def get_specgram(self, signal, rate):
        arr2D,freqs,bins = specgram(signal,window=window_hanning,
                                Fs = rate, NFFT=self.n_fft, noverlap=self.hop_length // 2)
        return arr2D,freqs,bins
    

    def update(self, n):
        #mic_input = self.click_sense.get_mic_input_spec()
        mic_input = self.click_sense.get_mic_input()
        
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk)

        arr2D,freqs,bins = self.get_specgram(mic_input,self.sr)

        im_data = self.im.get_array()

        if n < self.chunks_per_plot:
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1]*(self.chunks_per_plot - 1)
            im_data = np.delete(im_data,np.s_[:-keep_block],1)
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)

        return self.im,


    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)