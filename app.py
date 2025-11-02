from flask import Flask, jsonify
from flask_cors import CORS
import threading
import numpy as np
import pyaudio
import time

app = Flask(__name__)
CORS(app)  # Allows frontend to connect

# Audio config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
ENERGY_THRESH = 40000
ZCR_THRESH = 0.025
SILENT_CHUNKS = 10

status = {
    'mic_on': False,
    'energy': 0,
    'zcr': 0,
    'silence_count': 0,
    'last_update': time.time()
}

def frame_energy(audio_frame):
    return float(np.mean(audio_frame.astype(float) ** 2))

def frame_zcr(audio_frame):
    zero_crossings = np.count_nonzero(np.diff(np.sign(audio_frame)))
    return float(zero_crossings / (2 * len(audio_frame)))

def audio_monitor():
    global status
    paud = pyaudio.PyAudio()
    stream = paud.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        silence_count = 0
        mic_on = False
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)
            energy = frame_energy(audio_frame)
            zcr = frame_zcr(audio_frame)

            if mic_on:
                if (energy < ENERGY_THRESH) or (zcr > ZCR_THRESH):
                    silence_count += 1
                    if silence_count >= SILENT_CHUNKS:
                        mic_on = False
                else:
                    silence_count = 0
            else:
                if (energy > ENERGY_THRESH) and (zcr < ZCR_THRESH):
                    mic_on = True
                silence_count = 0

            # Update the status
            status = {
                'mic_on': mic_on,
                'energy': energy,
                'zcr': zcr,
                'silence_count': silence_count,
                'last_update': time.time()
            }
            time.sleep(0.08)
    finally:
        stream.stop_stream()
        stream.close()
        paud.terminate()

# Start audio monitoring thread
monitor_thread = threading.Thread(target=audio_monitor, daemon=True)
monitor_thread.start()

@app.route("/status")
def get_status():
    return jsonify(status)

if __name__ == "__main__":
    app.run(debug=True)

