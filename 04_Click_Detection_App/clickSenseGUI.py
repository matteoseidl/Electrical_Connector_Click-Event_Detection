import sys
import signal
import threading
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from visualizeAudioInputSpectrogram_2 import AudioSpectrogramPlotter2
from clickSenseMain_2 import ClickSense2

class SpectrogramCanvas(FigureCanvas):
    def __init__(self, click_sense, plotter):
        self.fig = Figure()
        super(SpectrogramCanvas, self).__init__(self.fig)

        self.fig_x = 12
        self.fig_y = 6
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.fig_x, self.fig_y))

        self.sr = click_sense.sampling_rate_downsampled
        print(f"sr: {self.sr}")



class ClickSenseGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.click_sense = ClickSense2()
        self.spectrogram_plotter = AudioSpectrogramPlotter2(self.click_sense)
        self.plotter = AudioSpectrogramPlotter2(self.click_sense, self.spectrogram_plotter)
        
        self.setWindowTitle("ClickSense GUI")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create buttons
        self.start_button = QPushButton("Start Recording", self)
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.clicked.connect(self.stop_recording)

        # Status label
        self.status_label = QLabel("Status: Not Recording", self)

        # Matplotlib Canvas for spectrogram
        self.spec_canvas = SpectrogramCanvas(self)

        self.layout.addWidget(self.spec_canvas)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.status_label)

        # Initialize ClickSense
        #self.click_sense = ClickSense2()
        #self.plotter = AudioSpectrogramPlotter(self.click_sense)
        
        # Set up flags for recording and animation
        self.recording_thread = None
        self.is_recording = False

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.status_label.setText("Status: Recording...")
            self.recording_thread = threading.Thread(target=self.run_recording)
            self.recording_thread.start()
        
    def run_recording(self):
        #self.click_sense.audio_chapture = True
        #self.plotter.start_animation()  # Start animation
        self.click_sense.start_recording()  # record audio

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.status_label.setText("Status: Not Recording")
            self.click_sense.stop_recording()  # Stop audio recording
            #self.plotter.stop_animation()  # Stop animation
            self.recording_thread.join()  # Wait for thread to finish

    def closeEvent(self, event):
        self.stop_recording()  # Ensure recording stops
        event.accept()

def signal_handler(sig, frame):
    print("Signal handler called with signal:", sig)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    app = QApplication(sys.argv)
    gui = ClickSenseGUI()
    gui.show()
    sys.exit(app.exec_())
