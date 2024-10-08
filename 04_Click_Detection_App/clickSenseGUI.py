import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import threading

from clickSenseMain_2 import ClickSense2
from visualizeAudioInputSpectrogram_2 import AudioSpectrogramPlotter2

class ClickDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_audio()
        
    def setup_ui(self):
        self.setWindowTitle("clickSense")
        self.setGeometry(100, 100, 1400, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.start_button = QPushButton("Start detection")
        self.stop_button = QPushButton("Stop detection")
        self.stop_button.setEnabled(False)
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))
        self.canvas = FigureCanvas(self.fig)
        
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def setup_audio(self):
        self.click_sense = ClickSense2()
        self.plotter = AudioSpectrogramPlotter2(self.click_sense, self.fig, self.ax)
        self.audio_thread = None
    
    def start_recording(self):
        self.audio_thread = threading.Thread(target=self.click_sense.start_recording)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        self.timer.start(int(self.plotter.plot_update_freq))
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_recording(self):
        if self.click_sense:
            self.click_sense.stop_recording()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        self.timer.stop()
        
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
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()