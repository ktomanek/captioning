# client for captioning client-server application
# start server first with `python captioning_server.py`, then run this client with `python captioning_client.py`

import argparse
import logging
import numpy as np
import pyaudio
import printers
import socketio
import time
import captioning_utils


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Real-time audio captioning using Whisper ASR and Silero VAD.")
parser.add_argument(
    "--rich_captions",
    action="store_true",
    help="Use rich captions for terminal output. Might not work on all terminals.",
)
parser.add_argument(
    "--audio_input_device_index",
    type=int,
    default=1,
    help="Index of the audio input device to use (default is 1).",
)
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="Hostname where the captioning server is running (default is localhost).",
)
parser.add_argument(
    "--port",
    type=str,
    default="5001",
    help="Port of the captioning server (default is 5001).",
)
parser.add_argument(
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

server_url = f"http://{args.host}:{args.port}"

sio = socketio.Client()


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

audio = pyaudio.PyAudio()
print("Recording started. Press Ctrl+C to stop.")


try:
    audio_stream = captioning_utils.get_audio_stream(audio, input_device_index=args.audio_input_device_index)
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

