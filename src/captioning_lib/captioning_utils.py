# utilities for real-time audio captioning with different ASR models and Silero VAD

import argparse
import logging
import numpy as np
import pyaudio
import queue
import time

from silero_vad import VADIterator, load_silero_vad
from captioning_lib import printers
from captioning_lib import transcribers

########## configurations ##########
def get_argument_parser():
    parser = argparse.ArgumentParser(description="Real-time audio captioning using Whisper ASR and Silero VAD.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="moonshine_onnx_tiny",
        choices=list(transcribers.FasterWhisperTranscriber.AVAILABLE_MODELS.keys()) + 
        list(transcribers.NemoTranscriber.AVAILABLE_MODELS.keys()) + 
        list(transcribers.MoonshineTranscriber.AVAILABLE_MODELS.keys()) + 
        list(transcribers.RemoteGPUTranscriber.AVAILABLE_MODELS.keys()) + 
        list(transcribers.TranslationTranscriber.AVAILABLE_MODELS.keys()) +
        list(transcribers.VoskTranscriber.AVAILABLE_MODELS.keys()) +
        list(transcribers.ONNXWhisperTranscriber.AVAILABLE_MODELS.keys()),
        help="ASR model to use.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=transcribers.DEFAULT_LANGUAGE,
        help="Language to transcribe in (not all models support all languages).",
    )
    parser.add_argument(
        "-c",
        "--show_word_confidence_scores",
        action="store_true",
        default=False,
        help="Calculate and show per-word confidence scores.",
    )
    parser.add_argument(
        "-rc",
        "--rich_captions",
        action="store_true",
        default=False,
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
        default=15.0,
        help="Maximum duration in seconds for a segment before it is transcribed.",
    )
    parser.add_argument(
        "--eos_min_silence",
        type=int,
        default=100,
        help="Minimum silence duration in milliseconds to consider the end of a segment.",
    )
    parser.add_argument(
        "-i",
        "--audio_input_device_index",
        type=int,
        help="Index of the audio input device to use (default is 1).",
    )
    parser.add_argument(
        "-d",
        "--show_audio_devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to custom model file (required for whisperonnx model type).",
    )
    parser.add_argument(
        "--recent_chunk_mode",
        action="store_true",
        default=False,
        help="Use recent-chunk mode for partials instead of retranscribing all accumulated audio. More efficient for longer min_partial_duration (> 2s). Enables token streaming for supported models.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed transcription information including mode and timing details.",
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

# default language
LANGUAGE = 'en'
######################################


def load_asr_model(model_name, language, sampling_rate=SAMPLING_RATE, show_word_confidence_scores=False, model_path=None, output_streaming=True):
    logging.debug("Loading ASR model...")
    if model_name.startswith('fasterwhisper'):
        asr_model = transcribers.FasterWhisperTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('nemo'):
        asr_model = transcribers.NemoTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('moonshine'):
        asr_model = transcribers.MoonshineTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('remote'):
        asr_model = transcribers.RemoteGPUTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('translation'):
        asr_model = transcribers.TranslationTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('vosk'):
        asr_model = transcribers.VoskTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, output_streaming=output_streaming)
    elif model_name.startswith('whisperonnx'):
        asr_model = transcribers.ONNXWhisperTranscriber(model_name, sampling_rate, show_word_confidence_scores, language, model_path=model_path, output_streaming=output_streaming)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

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

class TranscriptionWorker():

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.is_speech_recording = False
        self.had_speech = False
        self.frames_since_last_speech = 0
        self.transcribed_segments = []
        self.last_partial_transcribed_length = 0  # Track position for recent-chunk mode
        self.accumulated_partial_text = ""  # Store accumulated partial text for display


    def reset(self):
        self.is_speech_recording = False
        self.had_speech = False
        self.frames_since_last_speech = 0
        self.transcribed_segments = []
        self.last_partial_transcribed_length = 0
        self.accumulated_partial_text = ""

    def time_since_last_speech(self):
        # seconds since last recording
        return float(self.frames_since_last_speech) / self.sampling_rate


    def transcription_worker(
            self,
            asr,
            audio_queue,
            caption_printer,
            vad,
            stop_threads,
            min_partial_duration=MINIMUM_PARTIAL_DURATION,
            max_segment_duration=MAXIMUM_SEGMENT_DURATION,
            recent_chunk_mode=False):
        """Worker thread that processes audio chunks for transcription"""

        # transcription logic inspired by 
        # https://github.com/usefulsensors/moonshine/blob/main/demo/moonshine-onnx/live_captions.py

        speech_buffer = np.empty(0, dtype=np.float32)
        self.is_speech_recording = False
        time_since_last_transcription = time.time()

        while not stop_threads.is_set():
            try:
                # read new chunk from queue and add to buffer
                chunk = audio_queue.get(timeout=0.05)
                chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                speech_buffer = np.concatenate((speech_buffer, chunk_np))
                speech_buffer_duration = len(speech_buffer) / self.sampling_rate
                current_recording_duration = time.time() - time_since_last_transcription            

                # process speech in buffer depending on VAD event
                vad_event = vad(chunk_np)
                if vad_event:
                    logging.debug(f"VAD event detected: {vad_event}")
                    if "start" in vad_event:
                        self.is_speech_recording = True
                        self.had_speech = True
                        self.frames_since_last_speech = 0
                        self.last_partial_transcribed_length = 0  # Reset partial tracking
                        self.accumulated_partial_text = ""  # Reset accumulated text
                        time_since_last_transcription = time.time()  # Reset timer when speech starts
                    elif "end" in vad_event:
                        # finish the segment by processing all so far and then flushing buffer
                        self.is_speech_recording = False
                        self.frames_since_last_speech += len(chunk_np)
                        
                        complete_text = ""
                        for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                            if text_chunk:
                                complete_text += text_chunk
                        complete_text = complete_text.strip()
                        if complete_text:
                            # Only display the final complete segment once
                            caption_printer.print(complete_text, duration=speech_buffer_duration, partial=False)
                            self.transcribed_segments.append(complete_text)
                        speech_buffer = np.empty(0, dtype=np.float32)
                        # Reset partial tracking for new segment
                        self.last_partial_transcribed_length = 0
                        self.accumulated_partial_text = ""
                        time_since_last_transcription = time.time()
                else:
                    # no VAD event means recording state hasn't changed
                    if self.is_speech_recording:
                        # force end a segment if it is getting too long even if no EOS detected by VAD
                        if speech_buffer_duration > max_segment_duration:  # e.g., 5 seconds
                            logging.debug(f"Max segment duration reached, ending segment: {speech_buffer_duration:.2f} sec")
                            
                            complete_text = ""
                            for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                                if text_chunk:
                                    complete_text += text_chunk
                            
                            complete_text = complete_text.strip()
                            if complete_text:
                                # Only display the final complete segment once
                                caption_printer.print(complete_text, duration=speech_buffer_duration, partial=False)
                                self.transcribed_segments.append(complete_text)
                            speech_buffer = np.empty(0, dtype=np.float32)
                            # Reset partial tracking for new segment
                            self.last_partial_transcribed_length = 0
                            self.accumulated_partial_text = ""
                            time_since_last_transcription = time.time()

                        # if we have enough data in the buffer, transcribe a partial
                        elif current_recording_duration > min_partial_duration:
                            logging.debug(f"Transcribing partial segment: {current_recording_duration:.2f} sec")
                            
                            if not recent_chunk_mode:
                                # Mode 1: Retranscribe all accumulated audio (better quality for short durations)
                                self.accumulated_partial_text = ""
                                for text_chunk in asr.transcribe(speech_buffer, segment_end=False):
                                    if text_chunk:
                                        self.accumulated_partial_text += text_chunk
                                        d = len(speech_buffer) / self.sampling_rate
                                        caption_printer.print(self.accumulated_partial_text, duration=d, partial=True, 
                                                             is_recent_chunk_mode=False, recent_chunk_duration=None)
                            else:
                                # Mode 2: Transcribe only recent chunk (efficient for long durations)
                                recent_chunk = speech_buffer[self.last_partial_transcribed_length:]
                                if len(recent_chunk) > 0:
                                    recent_text = ""
                                    for text_chunk in asr.transcribe(recent_chunk, segment_end=False):
                                        if text_chunk:
                                            recent_text += text_chunk
                                            # Update accumulated display text
                                            self.accumulated_partial_text += text_chunk
                                            d = len(speech_buffer) / self.sampling_rate
                                            recent_chunk_duration = len(recent_chunk) / self.sampling_rate
                                            caption_printer.print(self.accumulated_partial_text, duration=d, partial=True, 
                                                                 is_recent_chunk_mode=True, recent_chunk_duration=recent_chunk_duration)
                                    
                                    # Update tracking position
                                    self.last_partial_transcribed_length = len(speech_buffer)
                            
                            time_since_last_transcription = time.time()
                    else:
                        empty_frames_to_keep = int(0.1 * self.sampling_rate)
                        speech_buffer = speech_buffer[-empty_frames_to_keep:]

                        self.frames_since_last_speech += len(chunk_np)

            except queue.Empty:
                if stop_threads.is_set():
                    break
                continue
            except Exception as e:
                if stop_threads.is_set():
                    break
                print(f"\nTranscription error: {e}")
                continue

        if len(speech_buffer) > 0:
            logging.debug("Flushing remaining speech buffer...")
            
            complete_text = ""
            for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                if text_chunk:
                    complete_text += text_chunk
            
            complete_text = complete_text.strip()
            if complete_text:
                # Only display the final complete segment once
                caption_printer.print(complete_text, duration=len(speech_buffer) / self.sampling_rate, partial=False)
                self.transcribed_segments.append(complete_text)
            speech_buffer = np.empty(0, dtype=np.float32)
            # Reset partial tracking
            self.last_partial_transcribed_length = 0
            self.accumulated_partial_text = ""


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


# find best audio input device by picking the one with at least one input channel and name 'default'
def find_best_audio_input_device():
    p = pyaudio.PyAudio()
    best_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print('device', device_info['name'], 'input channels', device_info['maxInputChannels'])
        if device_info['maxInputChannels'] > 0 and 'default' in device_info['name'].lower():
            best_device_index = i
            break
    p.terminate()
    return best_device_index


def find_default_input_device():
    """Find the default microphone device using PyAudio. If no default device is found, list all available input devices."""
    p = pyaudio.PyAudio()
    default_info = p.get_default_input_device_info()
    p.terminate()
    if default_info:
        print(f"Default input device: {default_info['name']} (index: {default_info['index']})")
        return {
            'name': default_info['name'],
            'index': default_info['index']
        }

    if not default_info:
        print("\nAll available input devices:")
        list_audio_devices()
        return None

if __name__ == "__main__":
    default_mic = find_default_input_device()
    print(default_mic)#['name'], default_mic['index'])
