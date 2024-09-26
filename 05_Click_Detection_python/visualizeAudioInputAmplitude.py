import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

# range for paFloat32: -1.0 to 1.0 (https://portaudio.com/docs/v19-doxydocs-dev/portaudio_8h.html#a2f16d29916725b8791eae60ab9e0b081)
y_lim = 1.0

class AudioAmplitudePlotter:
    def __init__(self, click_sense):
        self.click_sense = click_sense

        self.chunk_freq = click_sense.chunk/click_sense.sampling_rate_downsampled
        chunks_to_plot = 8
        self.plot_update_freq = self.chunk_freq * 1000 # update plot for every new chunk

        self.time_old = None
        self.time_new = None
        
        self.x_lim = (self.click_sense.chunk) * chunks_to_plot
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('clickSense - Amplitude Plotter')
        self.line, = self.ax.plot(np.zeros(self.x_lim))
        self.ax.set_ylim(-y_lim, y_lim)
        self.ax.set_xlim(0, self.x_lim)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.plot_update_freq, blit=True)
        plt.show()

    def update(self, frame):
        self.time_new = time.time()
        #print("update")
        #print(f"Time now: {self.time_new} seconds")

        mic_input = self.click_sense.get_mic_input()
        if mic_input is None:
            mic_input = np.zeros(self.click_sense.chunk) 

        data = self.line.get_ydata()

        data = np.roll(data, -self.click_sense.chunk)
        data[-self.click_sense.chunk:] = mic_input
        self.line.set_ydata(data)

        if self.time_old is not None:
            time_diff = self.time_new - self.time_old
            print(f"Time difference: {time_diff} seconds")

        self.time_old = self.time_new

        return self.line,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)

