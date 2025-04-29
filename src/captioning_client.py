# client for captioning client-server application
# start server first with `python captioning_server.py`, then run this client with `python captioning_client.py`

import logging
import numpy as np
import pyaudio
import printers
import socketio
import time
import captioning_utils


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


SERVER_URL = "http://localhost:5001"


USE_RICH_CAPTIONS = True
if USE_RICH_CAPTIONS:
    caption_printer = printers.RichCaptionPrinter()
else:
    caption_printer = printers.PlainCaptionPrinter()


sio = socketio.Client()


@sio.event
def connect():
    print(f"Connected to captioning server: {SERVER_URL}...")
    sio.emit('server_config_request')

@sio.event
def disconnect():
    print('Disconnected from captioning server.')

@sio.on('audio_processed')
def handle_info(data):
    logging.debug(f"Server received audio data: {data}")

@sio.on('transcription')
def handle_transcription(data):
    logging.debug(f"Received transcription data of size: {len(data['transcript'])}")
    transcription = data['transcript']
    is_partial = data['partial'] # segment or partial
    duration = data.get('duration', None)
    caption_printer.print(transcription, partial=is_partial, duration=duration)

@sio.on('server_config')
def handle_server_config(data):
    print(f"ASR server configuration:\n{data}")



sio.connect(SERVER_URL)

audio = pyaudio.PyAudio()
print("Recording started. Press Ctrl+C to stop.")


try:
    audio_stream = captioning_utils.get_audio_stream(audio)
    logging.info("Started audio stream...")

    while True:
            data = audio_stream.read(captioning_utils.AUDIO_FRAMES_TO_CAPTURE)
            sio.emit('audio_data', data)
            logging.debug(f"Emitted audio data of size: {len(data)}")
            # Brief pause to allow response to be processed
            time.sleep(0.01)
except KeyboardInterrupt:
    print("KeyboardInterrupt received, stopping audio streaming.")
    
finally:
    # Make sure we disconnect properly
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()
    sio.disconnect()
    print("Disconnected from server, all shut down")

