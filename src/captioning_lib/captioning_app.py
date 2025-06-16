# Stand-alone captioning app that can also be used to evaluate streaming performance (eval mode).
#
# For captioning from microphone:
# python captioning_app.py --model moonshine_onnx_tiny --rich_captions --eos_min_silence=200
#
# For evaluation with audio file and reference transcript:
# python captioning_app.py --model moonshine_onnx_tiny --eval --audio_file=../samples/jfk.mp3 --reference_file=../samples/jfk.txt


import json
import logging
from captioning_lib import printers
import queue
from captioning_lib import captioning_utils
from captioning_lib import evaluation_utils
import pyaudio
import threading
import time


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    # Extended command line arguments for evaluation.
    
    parser = captioning_utils.get_argument_parser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation mode with audio file input instead of microphone.",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        help="Path to audio file for evaluation (required in evaluation mode).",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        help="Path to reference transcript file for evaluation (required in evaluation mode).",
    )
    
    parser.add_argument(
        "--rtf",
        type=float,
        default=0.0,
        help="Real-Time Factor for audio processing speed. Set to 1.0 for real-time, 0.0 for no delay.",
    )
    args = parser.parse_args()

    # Validation for evaluation mode
    if args.eval and (not args.audio_file or not args.reference_file):
        parser.error("Evaluation mode requires both --audio_file and --reference_file.")

    if args.show_audio_devices:
        captioning_utils.list_audio_devices()
        exit(0)

    return args


def capture_audio_from_stream(audio_stream, audio_queue, stop_threads, caption_printer):

    print("Recording started. Press Ctrl+C to stop.")
    caption_printer.start()


    try:
        while True:
            data = audio_stream.read(captioning_utils.AUDIO_FRAMES_TO_CAPTURE)
            audio_queue.put(data, timeout=0.1)
    except queue.Full:
            logging.warning("Audio queue is full, skipping this chunk.")
    except KeyboardInterrupt:
        # print("KeyboardInterrupt received, stopping recording.")
        pass
        
    finally:
        # Signal transcription thread to stop and wait for a bit for the 
        # transcription thread to handle remaining audio chunks
        time.sleep(0.2)
        stop_threads.set()

        # Empty queue
        while not audio_queue.empty():
            logging.debug("Emptying audio queue...")
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except queue.Empty:
                break

        # Clean up audio resources
        audio_stream.stop_stream()
        audio_stream.close()
        

def capture_audio_from_file(
        audio_file, reference_file, audio_queue, stop_threads, caption_printer, rtf):

    """Simulate real-time audio streaming with specified speed factor.
    
    If `rtf` is set to 1.0, it simulates real-time audio input speed but adding delay. 
    If set to 0.0, no delay is introduced, and the audio file is processed as fast as possible.
    """

    # get audio chunks to simulate microphone
    audio_data, sample_rate = evaluation_utils.read_audio_file(audio_file)
    if sample_rate != captioning_utils.SAMPLING_RATE:
        raise ValueError(f"Sample rate mismatch: expected {captioning_utils.SAMPLING_RATE}Hz, got {sample_rate}Hz. ")
    audio_chunks = evaluation_utils.chunk_audio(audio_data, chunk_size=captioning_utils.AUDIO_FRAMES_TO_CAPTURE)
    
    print(f"Audio file split into {len(audio_chunks)} chunks of {captioning_utils.AUDIO_FRAMES_TO_CAPTURE} frames each")
    print("Audio file duration: {:.2f} seconds".format(len(audio_data) / captioning_utils.SAMPLING_RATE))
    
    # Read reference transcript
    reference_text = evaluation_utils.read_reference_file(reference_file)

    # chunk duration
    if rtf <= 0:
        sleep_time = 0
    else:
        chunk_duration = captioning_utils.AUDIO_FRAMES_TO_CAPTURE / captioning_utils.SAMPLING_RATE
        sleep_time = chunk_duration / rtf
        print(f"RTF: {rtf:.2f}")
        print(f">> Sleep time: {sleep_time:.2f} seconds per chunk")
        print(f">> Total wait time: {len(audio_chunks) * sleep_time:.2f} seconds")

    start_time = time.time()
    for chunk in audio_chunks:
        # Simulate real-time audio input by waiting between chunks
        time.sleep(sleep_time)
        try:
            audio_queue.put(chunk)
        except queue.Full:
            logging.warning("Audio queue is full, skipping this chunk.")

    # wait until all audio from queue is processed
    while not audio_queue.empty():
        time.sleep(0.05)

    # send stop signal to transcription thread and give smoe time to finish
    stop_threads.set()
    time.sleep(1.0)

    time_elapsed = time.time() - start_time
    print(f"Total processing time for audio file: {time_elapsed:.2f} seconds")

    full_transcript = caption_printer.get_complete_caption()
    wer = evaluation_utils.get_wer(reference_text, full_transcript, normalized=True)

    audio_duration = len(audio_data) / captioning_utils.SAMPLING_RATE
    results = {
        "audio_duration_seconds": audio_duration,
        "processing_time_seconds": time_elapsed,
        "rtf": rtf,
        "normalized_wer": wer,        
        # "transcript": full_transcript,
        # " reference": reference_text,
        }
    
    print("\n>>> Evaluation Results:\n", json.dumps(results, indent=2))


def main():
    """Main function supporting both live captioning and evaluation modes."""
    args = get_args()

    if args.eval:
        caption_printer = evaluation_utils.EvaluationPrinter()
    else:
        if args.rich_captions:
            caption_printer = printers.RichCaptionPrinter()
        else:
            caption_printer = printers.PlainCaptionPrinter()

    
    vad = captioning_utils.get_vad(eos_min_silence=args.eos_min_silence)    
    asr_model = captioning_utils.load_asr_model(model_name=args.model, 
                                                language=args.language,
                                                sampling_rate=captioning_utils.SAMPLING_RATE, 
                                                show_word_confidence_scores=args.show_word_confidence_scores)
    audio_queue = queue.Queue(maxsize=1000)

    # Start transcription thread
    stop_threads = threading.Event()  # Event to signal threads to stop    
    transcription_handler = captioning_utils.TranscriptionWorker(sampling_rate=captioning_utils.SAMPLING_RATE)
    transcriber = threading.Thread(target=transcription_handler.transcription_worker, 
                                   kwargs={'vad': vad,
                                           'asr': asr_model,
                                           'audio_queue': audio_queue,
                                           'caption_printer': caption_printer,
                                           'stop_threads': stop_threads,
                                           'min_partial_duration': args.min_partial_duration,
                                           'max_segment_duration': args.max_segment_duration})
    transcriber.daemon = True
    transcriber.start()


    if args.eval:
        capture_audio_from_file(args.audio_file, args.reference_file, 
                                audio_queue, stop_threads,
                                caption_printer, args.rtf)
    else:
        audio = pyaudio.PyAudio()
        device_index = args.audio_input_device_index
        if device_index:
            print(f"Using user specified audio input device index: {device_index}")
        else:
            # find default device index
            input_device = captioning_utils.find_default_input_device()
            print(f"Using default audio input device: {input_device}")
            device_index = input_device['index']
        audio_stream = captioning_utils.get_audio_stream(audio, input_device_index=device_index)
        capture_audio_from_stream(audio_stream, audio_queue, stop_threads, caption_printer)

        audio.terminate()
        caption_printer.stop()
        print("\nRecording stopped.")
        

    print("\n>>> Model stats:")
    asr_model.get_stats()


if __name__ == "__main__":
    main()

