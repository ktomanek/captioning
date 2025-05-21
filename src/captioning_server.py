# server side for captioning client-server application
# start server first with `python captioning_server.py`
# supports both terminal-based and web-based clients simultaneously

from flask import Flask, render_template
from flask_socketio import SocketIO
import logging
import numpy as np
import printers
import queue
import captioning_utils
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    # Extended command line arguments for evaluation.
    parser = captioning_utils.get_argument_parser()
    parser.add_argument(
        "--server_host",
        type=str,
        default="0.0.0.0",
        help="Port for server.",
    )
    parser.add_argument(
        "--server_port",
        type=str,
        default="5002",
    )
    parser.add_argument(
        "--no_https",
        action="store_false",
        default=True,
    )

    args = parser.parse_args()
    return args

class RemotePrinter(printers.CaptionPrinter):
    """Printer that sends transcriptions via websocket to a remote client."""

    def print(self, transcript, duration=None, partial=False):
        socketio.emit('transcription', {
            'transcript': transcript,
            'partial': partial,
            'duration': duration
        })   
        logging.debug(f"Transcription sent: {transcript}, Partial: {partial}, Duration: {duration}")


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

client_connected = False  # Flag to track if a client is already connected

args = get_args()

remote_caption_printer = RemotePrinter()
vad = captioning_utils.get_vad(eos_min_silence=args.eos_min_silence)
asr_model = captioning_utils.load_asr_model(args.model, captioning_utils.SAMPLING_RATE,
                                            args.show_word_confidence_scores)
audio_queue = queue.Queue(maxsize=1000)  
stop_threads = threading.Event()  # Event to signal threads to stop
transcription_handler = captioning_utils.TranscriptionWorker(sampling_rate=captioning_utils.SAMPLING_RATE)
transcriber = threading.Thread(target=transcription_handler.transcription_worker, 
                                kwargs={'vad': vad,
                                        'asr': asr_model,
                                        'audio_queue': audio_queue,
                                        'caption_printer': remote_caption_printer,
                                        'stop_threads': stop_threads,
                                        'min_partial_duration': args.min_partial_duration,
                                        'max_segment_duration': args.max_segment_duration
                                        })
transcriber.daemon = True
transcriber.start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/web')
def web_client():
    return render_template('web_client.html')

@app.route('/translate')
def translate_client():
    return render_template('translations_client.html')

@socketio.on('connect')
def socket_connect():
    global client_connected
    if client_connected:
        logging.warning('Connection rejected: Another client is already connected')
        return False  # Reject the connection
    client_connected = True
    print('Client connected')


@socketio.on('disconnect')
def socket_disconnect():
    global client_connected
    client_connected = False
    print('Client disconnected')

@socketio.on('server_config_request')
def handle_server_config_request():
    server_config = {
        'model': args.model,
        'max_segment_duration': args.max_segment_duration,
        'min_partial_duration': args.min_partial_duration,
        'eos_min_silence': args.eos_min_silence
        }
    socketio.emit('server_config', server_config)


@socketio.on('audio_data')
def handle_audio(data):
    logging.debug(f"Received audio chunk of size: {len(data)}, queue size: {audio_queue.qsize()}")
    audio_queue.put(data)
    socketio.emit('audio_processed', {'status': 'success',
                                      'chunk_size': len(data),
                                      'queue_size': audio_queue.qsize(),
                                      })


if __name__ == '__main__':
    ssl_context = None
    if args.no_https:
        ssl_context='adhoc'
    socketio.run(app, host=args.server_host, port=args.server_port, ssl_context=ssl_context)