# client for captioning client-server application
# start server first with `python captioning_server.py`, then run this client with `python captioning_client.py`

import argparse
import logging
import numpy as np
import pyaudio
from captioning_lib import printers
import socketio
import time
from captioning_lib import captioning_utils


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Real-time audio captioning using Whisper ASR and Silero VAD.")
parser.add_argument(
    "-rc",
    "--rich_captions",
    action="store_true",
    help="Use rich captions for terminal output. Might not work on all terminals.",
)
parser.add_argument(
    "-i",
    "--audio_input_device_index",
    type=int,
    default=1,
    help="Index of the audio input device to use (default is 1).",
)
parser.add_argument(
    "--server_url",
    type=str,
    default="https://127.0.0.1:5002",
    help="Hostname where the captioning server is running (default is localhost).",
)
parser.add_argument(
    "-d",
    "--show_audio_devices",
    action="store_true",
    help="List available audio input devices and exit.",
)


args = parser.parse_args()

if args.show_audio_devices:
    captioning_utils.list_audio_devices()
    exit(0)

if args.rich_captions:
    caption_printer = printers.RichCaptionPrinter()
else:
    caption_printer = printers.PlainCaptionPrinter()


# identify the audio input device
device_index = args.audio_input_device_index
if device_index:
    print(f"Using user specified audio input device index: {device_index}")
else:
    # find default device index
    input_device = captioning_utils.find_default_input_device()
    print(f"Using default audio input device: {input_device}")
    device_index = input_device['index']


# connect to server (disable SSL verification for self-signed cert)
server_url = args.server_url
sio = socketio.Client(ssl_verify=False)


@sio.event
def connect():
    print(f"Connected to captioning server: {server_url}...")
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

sio.connect(server_url)



try:
    audio = pyaudio.PyAudio()
    audio_stream = captioning_utils.get_audio_stream(audio, input_device_index=device_index)
    print("Recording started. Press Ctrl+C to stop.")
    logging.info("Started audio stream...")
    caption_printer.start()

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
    caption_printer.stop()
    print("Disconnected from server, all shut down")

