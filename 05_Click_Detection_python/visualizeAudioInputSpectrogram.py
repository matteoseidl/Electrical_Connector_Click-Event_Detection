import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa
import math

# https://librosa.org/doc/main/auto_examples/plot_patch_generation.html
# https://librosa.org/doc/main/generated/librosa.power_to_db.html
# https://github.com/jonnor/machinehearing/blob/master/handson/badminton/BadmintonSoundEvents.ipynb


class AudioSpectrogramPlotter:
    def __init__(self, click_sense):
        self.click_sense = click_sense

        self.chunk_freq = click_sense.chunk/click_sense.sampling_rate_downsampled # in case of a chunk size of 2048 and sampling rate 16 kHz: 2048/16000 = 0.128 s, meaning that 0.128 s corresponds to one chunk
        self.chunks_per_plot = click_sense.chunks_per_plot # number of chunks to plot
        self.plot_update_freq = self.chunk_freq * 1000 * self.chunks_per_plot # in case of 8 chunks: 8 * 0.128 = 1.024 s, multiplication with 1000 to convert to milliseconds

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('clickSense - Spectrogram Plotter')

        self.sr = click_sense.sampling_rate_downsampled # sampling rate of the downsampled audio signal for librosa

        self.resolution = 0.032 # in seconds, resulting in 32 frames for the 1.024 s plot duration
        self.hop_length = int(self.resolution * click_sense.sampling_rate_downsampled) # hop_length is the number of samples between successive frames, 0.032s * 16000 1/s = 512 samples
        self.n_fft = self.next_power_of_2(self.hop_length) # n_fft is the number of samples in each window, 512 samples, next power of 2 is 1024

        self.samples_per_plot = int((click_sense.chunk * click_sense.chunks_per_plot))

        #initialize spectrogram with zeros
        #self.spectrogram = np.zeros((128, int(self.plot_update_freq/self.resolution)))

        # initialize mic_input buffer with zeros
        self.mic_buffer = np.zeros(click_sense.chunk * self.chunks_per_plot )

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True) # interval in milliseconds

        plt.show()

    def next_power_of_2(self, x):
        return 2**(math.ceil(math.log(x, 2)))

    def update(self, frame):
        mic_input = self.click_sense.get_mic_input_spec()
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk * self.chunks_per_plot)

        # convert mic_input to numpy.ndarray
        """mic_input = np.array(mic_input)
        self.mic_buffer = np.roll(self.mic_buffer, -mic_input.shape[0], axis=0)
        self.mic_buffer[-mic_input.shape[0]:] = mic_input"""

        print(f"mic_input shape: {mic_input.shape}")

        melspec  = librosa.feature.melspectrogram(y=mic_input, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        #print(f"melspec shape: {melspec.shape}")

        self.melspec_dB = librosa.power_to_db(melspec, ref=np.max)

        #print(f"melspec_dB shape: {melspec_dB.shape}")
        #print(f"melspec_dB max: {np.min(self.melspec_dB)}")

        self.ax.clear()
        
        librosa.display.specshow(self.melspec_dB, x_axis='time', y_axis='mel', ax=self.ax)
        #librosa.display.specshow(melspec, x_axis='time', y_axis='mel', ax=self.ax)

        self.ax.set(title='Mel Spectrogram')

        return self.ax.images

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)