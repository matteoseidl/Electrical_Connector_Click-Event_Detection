import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import librosa

y_lim = 32768

# https://librosa.org/doc/main/auto_examples/plot_patch_generation.html

class AudioSpectrogramPlotter:
    def __init__(self, click_sense):
        self.click_sense = click_sense

        x_lim = (self.click_sense.chunk)*4
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('clickSense - Amplitude Plotter')
        self.line, = self.ax.plot(np.zeros(x_lim))
        self.ax.set_ylim(-y_lim, y_lim)
        self.ax.set_xlim(0, x_lim)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.064)
        plt.show()

    def update(self, frame):
        self.time_new = time.time()
        print("update")
        print(f"Time now: {self.time_new} seconds")

        mic_input = self.click_sense.get_mic_input()
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk) 

        data = self.line.get_ydata()

        data = np.roll(data, -self.click_sense.chunk)
        data[-self.click_sense.chunk:] = mic_input
        self.line.set_ydata(data)

        """if self.time_old is not None:
            time_diff = self.time_new - self.time_old
            print(f"Time difference: {time_diff} seconds")"""

        self.time_old = self.time_new

        return self.line,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)