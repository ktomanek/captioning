# utilities for real-time audio captioning with different ASR models and Silero VAD

import argparse
import logging
import numpy as np
import pyaudio
import queue
import time

from silero_vad import VADIterator, load_silero_vad
import printers
import transcribers

########## configurations ##########
def get_argument_parser():
    parser = argparse.ArgumentParser(description="Real-time audio captioning using Whisper ASR and Silero VAD.")
    parser.add_argument(
        "--model",
        type=str,
        default="whisper_tiny",
        choices=list(transcribers.WhisperTranscriber.AVAILABLE_MODELS.keys()) + list(transcribers.NemoTranscriber.AVAILABLE_MODELS.keys()) + list(transcribers.MoonshineTranscriber.AVAILABLE_MODELS.keys()) + list(transcribers.RemoteGPUTranscriber.AVAILABLE_MODELS.keys()),
        help="ASR model to use.",
    )
    parser.add_argument(
        "--rich_captions",
        action="store_true",
        help="Use rich captions for terminal output. Might not work on all terminals.",
    )
    parser.add_argument(
        "--min_partial_duration",
        type=float,
        default=0.1,
        help="Minimum duration in seconds for partial transcriptions to be displayed.",
    )
    parser.add_argument(
        "--max_segment_duration",
        type=float,
        default=10.0,
        help="Maximum duration in seconds for a segment before it is transcribed.",
    )
    parser.add_argument(
        "--eos_min_silence",
        type=int,
        default=100,
        help="Minimum silence duration in milliseconds to consider the end of a segment.",
    )
    parser.add_argument(
        "--audio_input_device_index",
        type=int,
        default=1,
        help="Index of the audio input device to use (default is 1).",
    )
    parser.add_argument(
        "--show_audio_devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    
    return parser
########## configurations ##########

# audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLING_RATE = 16000
AUDIO_FRAMES_TO_CAPTURE = 512 # VAD strictly needs this number
INPUT_DEVICE_INDEX = 1 # use default device

# VAD settings
VAD_THRESHOLD = 0.5
EOS_MIN_SILENCE = 100 

# how many seconds we need to record to transcribe
MINIMUM_PARTIAL_DURATION = 0.1
MAXIMUM_SEGMENT_DURATION = 10.0

######################################


def load_asr_model(model_name, sampling_rate=SAMPLING_RATE):
    logging.debug("Loading ASR model...")
    if model_name.startswith('whisper'):
        asr_model = transcribers.WhisperTranscriber(model_name, sampling_rate)
    elif model_name.startswith('nemo'):
        asr_model = transcribers.NemoTranscriber(model_name, sampling_rate)
    elif model_name.startswith('moonshine'):
        asr_model = transcribers.MoonshineTranscriber(model_name, sampling_rate)
    elif model_name.startswith('remote'):
        asr_model = transcribers.RemoteGPUTranscriber(model_name, sampling_rate)
    print(f"ASR model {model_name} loaded.")
    return asr_model

def get_vad(eos_min_silence=EOS_MIN_SILENCE, vad_threshold=VAD_THRESHOLD, sampling_rate=SAMPLING_RATE):
    # Silero VAD now requires fixed sample windows (512 for 16khz sampling rate)
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=sampling_rate,
        threshold=vad_threshold,
        min_silence_duration_ms=eos_min_silence,
    )
    print('VAD loaded.')
    return vad_iterator

def transcription_worker(
        asr,
        audio_queue,
        caption_printer,
        vad,
        stop_threads,
        min_partial_duration=MINIMUM_PARTIAL_DURATION,
        max_segment_duration=MAXIMUM_SEGMENT_DURATION):
    """Worker thread that processes audio chunks for transcription"""

    # transcription logic inspired by 
    # https://github.com/usefulsensors/moonshine/blob/main/demo/moonshine-onnx/live_captions.py

    speech_buffer = np.empty(0, dtype=np.float32)
    is_speech_recording = False
    time_since_last_transcription = time.time()

    while not stop_threads.is_set():
        try:
            # read new chunk from queue and add to buffer
            chunk = audio_queue.get(timeout=0.05)
            chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            speech_buffer = np.concatenate((speech_buffer, chunk_np))
            speech_buffer_duration = len(speech_buffer) / SAMPLING_RATE
            current_recording_duration = time.time() - time_since_last_transcription            

            # process speech in buffer depending on VAD event
            vad_event = vad(chunk_np)
            if vad_event:
                logging.debug(f"VAD event detected: {vad_event}")
                if "start" in vad_event:
                    is_speech_recording = True
                elif "end" in vad_event:
                    # finish the segment by processing all so far and then flushing buffer
                    is_speech_recording = False
                    text = asr.transcribe(speech_buffer, segment_end=True)
                    caption_printer.print(text, duration=speech_buffer_duration, partial=False)
                    speech_buffer = np.empty(0, dtype=np.float32)  
                    time_since_last_transcription = time.time()
            else:
                # no VAD event means recording state hasn't changed
                if is_speech_recording:
                    # force end a segment if it is getting too long even if no EOS detected by VAD
                    if speech_buffer_duration > max_segment_duration:  # e.g., 5 seconds
                        logging.debug(f"Max segment duration reached, ending segment: {speech_buffer_duration:.2f} sec")
                        text = asr.transcribe(speech_buffer, segment_end=True)
                        caption_printer.print(text, duration=speech_buffer_duration, partial=False)
                        speech_buffer = np.empty(0, dtype=np.float32)  
                        time_since_last_transcription = time.time()

                    # if we have enough data in the buffer, transcribe a partial
                    elif current_recording_duration > min_partial_duration:
                        logging.debug(f"Transcribing partial segment: {current_recording_duration:.2f} sec")
                        text = asr.transcribe(speech_buffer, segment_end=False)
                        d = len(speech_buffer) / SAMPLING_RATE
                        caption_printer.print(text, duration=d, partial=True)
                        time_since_last_transcription = time.time()
                else:
                    empty_frames_to_keep = int(0.1 * SAMPLING_RATE)
                    speech_buffer = speech_buffer[-empty_frames_to_keep:]

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\nTranscription error: {e}")

    if len(speech_buffer) > 0:
        logging.debug("Flushing remaining speech buffer...")
        text = asr.transcribe(speech_buffer, segment_end=True)
        caption_printer.print(text, duration=len(speech_buffer) / SAMPLING_RATE, partial=False)
        speech_buffer = np.empty(0, dtype=np.float32)



def get_audio_stream(audio, input_device_index=INPUT_DEVICE_INDEX):
    print('Using audio input device:', audio.get_device_info_by_index(input_device_index).get('name'))
    audio_stream = audio.open(format=FORMAT, 
                        channels=CHANNELS,
                        rate=SAMPLING_RATE, 
                        input=True,
                        frames_per_buffer=AUDIO_FRAMES_TO_CAPTURE,
                        input_device_index=input_device_index)

    return audio_stream


def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"* Device [{i}]: {device_info['name']} \t input channels: {device_info['maxInputChannels']}, output channels: {device_info['maxOutputChannels']}")
    p.terminate()