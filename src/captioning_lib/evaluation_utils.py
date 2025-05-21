# used for captioning evaluation

import jiwer
import numpy as np
from typing import Dict, List, Tuple
from captioning_lib import printers

import soundfile as sf
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
transcript_normalizer = BasicTextNormalizer()

def get_wer(reference_text: str, transcript_text: str, normalized: bool = True) -> float:
    """Calculate Word Error Rate (WER) between reference and transcript."""
    
    if normalized:
        reference_text = transcript_normalizer(reference_text)
        transcript_text = transcript_normalizer(transcript_text)
    
    wer = jiwer.wer(reference_text, transcript_text)
    return wer

class EvaluationPrinter(printers.CaptionPrinter):
    """Printer that collects transcription data for evaluation.
    
    While segments and partials are shown, only full segments are stored for evaluation.
    This allows for both real-time display and later evaluation of full transcriptions.
    """
    
    def __init__(self):
        self.transcripts = []
        self.durations = []
        
    def print(self, transcript, duration=None, partial=False):

        if not transcript.strip():
            return
        if not partial:
            self.transcripts.append(transcript)
            self.durations.append(duration)
            if duration:
                print(f"\nSEGMENT: {transcript} ({duration:.2f} sec)")
            else:
                print(f"\nSEGMENT: {transcript}")
        else:
            print(f"\rPARTIAL: {transcript}", flush=True, end='')
    
    def get_complete_caption(self):
        """Return full caption as a single string."""
        c = " ".join(self.transcripts) if self.transcripts else ""
        return c.strip()


def read_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Read audio file and return data as float32 and sample rate."""
    data, sample_rate = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # # Convert to float32 in range [-1, 1]
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0 
    return data, sample_rate

def chunk_audio(audio_data: np.ndarray, chunk_size: int) -> List[bytes]:
    """Split audio into chunks to simulate streaming."""
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]

        # If the last chunk is smaller than chunk_size, pad it with zeros
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

        # Convert to int16 bytes for queue insertion (mimicking microphone input)
        chunk_bytes = (chunk * 32768).astype(np.int16).tobytes()
        chunks.append(chunk_bytes)
    return chunks



def read_reference_file(file_path: str) -> str:
    """Read reference transcript file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().replace("\n", " ").strip()


