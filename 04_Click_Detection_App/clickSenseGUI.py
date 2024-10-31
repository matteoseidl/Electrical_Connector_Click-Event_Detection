import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import threading
import time

from clickSenseMain import ClickSense
from visualizeAudioInputSpectrogram import AudioSpectrogramPlotter

class ClickDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_audio()
        
    # user interface setup
    def setup_ui(self):
        self.setWindowTitle("clickSense")
        self.setGeometry(50, 50, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # buttons for strating and stopping click detection
        self.start_button = QPushButton("Start detection")
        self.stop_button = QPushButton("Stop detection")
        self.stop_button.setEnabled(False)
        
        # text widgets for displaying status and click detection results
        self.status_label = QLabel("Click on start button", self)
        self.detection_label = QLabel("No click detected", self)

        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: black; background-color: yellow; border: 1px solid black;")
        self.detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detection_label.setStyleSheet("color: black; background-color: lightblue; border: 1px solid black;")
        
        # plot for displaying the audio input spectrogram
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # add widgets to the layout
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detection_label)
        
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_recording)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def setup_audio(self):
        self.click_sense = ClickSense() # initialize ClickSense object from clickSenseMain.py
        self.plotter = AudioSpectrogramPlotter(self.click_sense, self.fig, self.ax) # initialize AudioSpectrogramPlotter object from visualizeAudioInputSpectrogram.py
        self.audio_thread = None
    
    def start_detection(self):
        self.audio_thread = threading.Thread(target=self.click_sense.start_recording)
        self.audio_thread.daemon = True # set to run in the backround
        self.audio_thread.start()
        
        self.timer.start(int(self.plotter.plot_update_freq)) # plot update frequency defined in visualizeAudioInputSpectrogram.py

        self.status_label.setText("clickSense in action")
        self.status_label.setStyleSheet("color: black; background-color: lightgreen; border: 1px solid black;")
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_recording(self):
        if self.click_sense:
            self.click_sense.stop_recording()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0) # wait 1 second for the thread to finish
        self.timer.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


    # save the detection time in a csv file
    def save_detection_time():
        with open("click_detection_times.csv", "a") as file:
            file.write(f"{time.time()}\n")
    
    def update_plot(self):
        if self.plotter:
            self.detection_result = self.plotter.update() # return value from update method in AudioSpectrogramPlotter.py
            print(f"detection_result: {self.detection_result}")

            if self.detection_result:
                self.detection_label.setText("Click detected")
                self.detection_label.setStyleSheet("color: white; background-color: green; border: 1px solid black;")

                # save detection time in a csv file
                self.save_detection_time()

            self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = ClickDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()