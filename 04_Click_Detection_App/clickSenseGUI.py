import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from clickSenseMain_2 import ClickSense2
from visualizeAudioInputSpectrogram_2 import AudioSpectrogramPlotter2

class ClickDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.click_sense = ClickSense2()

        self.setWindowTitle("clickSense")
        self.setGeometry(100, 100, 1000, 800)
        
        # central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # start and stop buttons
        self.start_button = QPushButton("Start detection")
        self.stop_button = QPushButton("Stop detection")
        self.stop_button.setEnabled(False)
        
        # initialize plot
        self.fig_x = 12
        self.fig_y = 6
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.fig_x, self.fig_y))
        self.chunk_size = self.click_sense.chunk
        self.chunk_freq = self.chunk_size/self.click_sense.sampling_rate_downsampled # in case of a chunk size of 2048 and sampling rate 16 kHz: 2048/16000 = 0.128 s, meaning that 0.128 s corresponds to one chunk
        self.chunks_per_plot = self.click_sense.chunks_per_plot # number of chunks to plot
        self.plot_update_freq = self.chunk_freq * 1000
        self.samples_per_plot = int((self.chunk_size * self.chunks_per_plot))
        self.n_mels = 128
        self.sr = self.click_sense.sampling_rate_downsampled
        self.resolution = 0.016 # in seconds, resulting in 32 frames for the 1.024 s plot duration
        self.hop_length = int(self.resolution * self.click_sense.sampling_rate_downsampled)
        self.init_spec = np.zeros((self.n_mels, int(self.samples_per_plot / self.hop_length)))
        self.melspec_full = self.init_spec
        self.mel_spec_img = self.ax.pcolormesh(np.linspace(0, self.samples_per_plot / self.sr, self.init_spec.shape[1]),
                                               np.linspace(0, self.sr // 2, self.n_mels), 
                                               self.init_spec, shading='auto', cmap='inferno')
        self.ax.set_xlim(0 - self.resolution/2, (self.chunk_size / self.sr) * self.chunks_per_plot + self.resolution/2)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(base=self.chunk_freq))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))
        self.x_min, self.x_max = self.ax.get_xlim()
        self.x_min_thick = self.x_min + self.resolution/2
        self.x_max_thick = self.x_max - self.resolution/2
        self.top_dB_abs = 120
        self.mel_spec_img.set_clim(vmin=-self.top_dB_abs, vmax=0)
        self.colorbar = self.fig.colorbar(self.mel_spec_img, ax=self.ax, format="%+2.0f dB")
        self.colorbar.set_label("Decibels (dB)")
        self.ax.set(title='Mel Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        self.canvas = FigureCanvas(self.fig)
        
        # add the widgets to layout
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # connect buttons to functions
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        
        # initialize ClickSense and Plotter
        #self.click_sense = None
        self.plotter = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        self.plotter = AudioSpectrogramPlotter2(self.click_sense, self.fig, self.ax)
    
    def start_recording(self):
        
        # Start the audio capture thread
        self.click_sense.start_recording()
        
        # Start the timer for plot updates
        self.timer.start(self.plotter.plot_update_freq)
        
        # Update button states
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_recording(self):
        if self.click_sense:
            self.click_sense.stop_recording()
        self.timer.stop()
        
        # Update button states
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_plot(self):
        if self.plotter:
            self.plotter.update(None)
            self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = ClickDetectorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()