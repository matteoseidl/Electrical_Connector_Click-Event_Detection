import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

x_lim = 100
y_lim = 300

class AudioPlotter:
    def __init__(self, click_sense):
        self.click_sense = click_sense
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(np.zeros(x_lim))
        self.ax.set_ylim(0, y_lim)  # Adjust based on the expected range of amplitude
        self.ax.set_xlim(0, x_lim)

        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=True)
        plt.show()

    def update(self, frame):
        amplitude = self.click_sense.get_amplitude()
        if amplitude is None:
            amplitude = 0.0 

        # Update the line data
        data = self.line.get_ydata()
        data = np.roll(data, -1)
        data[-1] = amplitude
        self.line.set_ydata(data)
        return self.line,

    def handle_close(self, event):
        print("Closing plot window. Stopping recording...")
        self.click_sense.stop_recording()
        plt.close(self.fig)

