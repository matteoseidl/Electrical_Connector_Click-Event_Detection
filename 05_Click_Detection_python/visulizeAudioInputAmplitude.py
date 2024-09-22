import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# full range of int16 (https://learn.microsoft.com/en-us/dotnet/api/system.int16?view=net-8.0)
y_lim = 32768

class AudioAmplitudePlotter:
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
        amplitude = self.click_sense.get_amplitude()
        if amplitude is None:
            amplitude = np.zeros(self.click_sense.chunk) 

        data = self.line.get_ydata()

        data = np.roll(data, -self.click_sense.chunk)
        data[-self.click_sense.chunk:] = amplitude
        self.line.set_ydata(data)
        return self.line,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)

