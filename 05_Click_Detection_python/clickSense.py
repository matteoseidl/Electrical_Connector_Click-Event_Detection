import pyaudio
import numpy as np
import signal
import sys
import threading
import time

# Global flag for running the main loop
running = False

def signal_handler(sig, frame):
    global running
    running = False
    print("\nStopping...")

def start_audio_stream():
    global running
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    running = True
    print("Recording... Press Ctrl+C to stop.")

    while running:
        data = np.frombuffer(stream.read(1024), dtype=np.int16)
        amplitude = np.linalg.norm(data) / len(data)
        print(f"Amplitude: {amplitude:.2f}")

        # Sleep for 1 ms
        time.sleep(0.001)

    stream.stop_stream()
    stream.close()
    p.terminate()

def start():
    threading.Thread(target=start_audio_stream).start()

def stop():
    global running
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    start()
    while running:
        time.sleep(0.1)
