import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from clickSenseMain_2 import ClickSense2
from visualizeAudioInputSpectrogram_2 import AudioSpectrogramPlotter2

class ClickDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Click Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create buttons
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # Add widgets to layout
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # Connect buttons to functions
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        
        # Initialize ClickSense and Plotter
        self.click_sense = None
        self.plotter = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
    
    def start_recording(self):
        self.click_sense = ClickSense2()
        self.plotter = AudioSpectrogramPlotter2(self.click_sense, self.fig, self.ax)
        
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