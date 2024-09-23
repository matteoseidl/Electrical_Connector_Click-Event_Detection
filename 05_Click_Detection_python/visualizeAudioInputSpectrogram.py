import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa
import math

y_lim = 32768

# https://librosa.org/doc/main/auto_examples/plot_patch_generation.html
# https://librosa.org/doc/main/generated/librosa.power_to_db.html

class AudioSpectrogramPlotter:
    def __init__(self, click_sense):
        self.click_sense = click_sense
        self.chunk_size = click_sense.chunk
        self.chunk_freq =  self.chunk_size/click_sense.sampling_rate_downsampled
        self.plot_update_freq = self.chunk_freq*1000 # in milliseconds

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('clickSense - Spectrogram Plotter')
        #self.ax.set_xlim(left=0, right=self.chunk_freq*2)

        #self.y = None
        self.sr = click_sense.sampling_rate_downsampled

        self.resolution = 0.032 # 32 ms
        self.hop_length = int(self.resolution * click_sense.sampling_rate_downsampled)
        self.n_fft = self.next_power_of_2(self.hop_length)

        self.frames_per_plot = int((click_sense.chunk * 8) / self.hop_length)
        self.buffer = np.zeros((128, self.frames_per_plot))

        

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True) # interval in milliseconds

        plt.show()

    def next_power_of_2(self, x):
        return 2**(math.ceil(math.log(x, 2)))

    def update(self, frame):
        mic_input = self.click_sense.get_mic_input()
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk)

        # convert mic_input to numpy.ndarray
        mic_input = np.array(mic_input)

        melspec  = librosa.feature.melspectrogram(y=mic_input, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        #print(f"melspec shape: {melspec.shape}")

        melspec_dB = librosa.power_to_db(melspec, ref=np.max)

        #print(f"melspec_dB shape: {melspec_dB.shape}")
        print(f"melspec_dB max: {np.min(melspec_dB)}")

        self.buffer = np.roll(self.buffer, -melspec_dB.shape[1], axis=1)
        self.buffer[:, -melspec_dB.shape[1]:] = melspec_dB

        self.ax.clear()
        
        librosa.display.specshow(self.buffer, x_axis='time', y_axis='mel', ax=self.ax)

        self.ax.set(title='Mel Spectrogram')

        return self.ax.images

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)