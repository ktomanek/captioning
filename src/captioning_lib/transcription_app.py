# for evaluation of offline/non-streaming ASR with same models
import argparse
from captioning_lib import evaluation_utils
import jiwer
from transcribers import FasterWhisperTranscriber, NemoTranscriber, MoonshineTranscriber, VoskTranscriber, ONNXWhisperTranscriber
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
transcript_normalizer = BasicTextNormalizer()


def get_wer(reference_text: str, transcript_text: str, normalized: bool = True) -> float:
    """Calculate Word Error Rate (WER) between reference and transcript."""
    
    if normalized:
        reference_text = transcript_normalizer(reference_text)
        transcript_text = transcript_normalizer(transcript_text)
    
    wer = jiwer.wer(reference_text, transcript_text)
    return wer

def get_transcriber(model_name, sampling_rate, model_path=None, use_raspberry_pi_session_config=True):
    """Get the appropriate transcriber based on model name"""
    if model_name in FasterWhisperTranscriber.AVAILABLE_MODELS:
        return FasterWhisperTranscriber(model_name, sampling_rate)
    elif model_name in NemoTranscriber.AVAILABLE_MODELS:
        return NemoTranscriber(model_name, sampling_rate)
    elif model_name in MoonshineTranscriber.AVAILABLE_MODELS:
        return MoonshineTranscriber(model_name, sampling_rate)
    elif model_name in VoskTranscriber.AVAILABLE_MODELS:
        return VoskTranscriber(model_name, sampling_rate)
    elif model_name in ONNXWhisperTranscriber.AVAILABLE_MODELS:
        return ONNXWhisperTranscriber(model_name, sampling_rate, model_path=model_path, use_raspberry_pi_session_config=use_raspberry_pi_session_config)

    else:
        available_models = list(FasterWhisperTranscriber.AVAILABLE_MODELS.keys()) + \
                          list(NemoTranscriber.AVAILABLE_MODELS.keys()) + \
                          list(MoonshineTranscriber.AVAILABLE_MODELS.keys()) + \
                          list(VoskTranscriber.AVAILABLE_MODELS.keys()) + \
                          list(ONNXWhisperTranscriber.AVAILABLE_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(available_models)}")

def main():
    parser = argparse.ArgumentParser(description='Transcribe a WAV file using various ASR models')
    parser.add_argument('--audio_file', required=True, help='Path to the WAV file to transcribe')
    parser.add_argument('--reference_file', help='Path to the reference transcript file for evaluation')
    parser.add_argument('--model', required=True, help='Model to use for transcription')
    parser.add_argument('--model_path', help='Path to custom ONNX model files (required for whisperonnx model)')
    parser.add_argument('--use_raspberry_pi_session_config', action='store_true', default=True, help='Use Raspberry Pi optimized session configuration for ONNX models')
    args = parser.parse_args()
    
   
    # Read the WAV file
    audio_data, sampling_rate = evaluation_utils.read_audio_file(args.audio_file)
    audio_duration = len(audio_data) / sampling_rate
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # read references file
    if args.reference_file:
        print(f"Reading reference transcript file: {args.reference_file}")
        reference_text = evaluation_utils.read_reference_file(args.reference_file)
    else:
        reference_text = None

    # Initialize the transcriber
    print(f"Initializing transcriber with model: {args.model}")
    transcriber = get_transcriber(args.model, sampling_rate, args.model_path, args.use_raspberry_pi_session_config)
    
    # Transcribe the entire audio file as a single segment
    transcript_generator = transcriber.transcribe(audio_data, segment_end=True)
    # Collect all yielded results into a single transcript
    transcript = "".join(transcript_generator)
    
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULT:")
    print(transcript)
    if reference_text:
        print("REFERENCE:")
        print(reference_text)
    
    print("="*80 + "\n")
    
    print("\nDetailed Stats:")
    transcriber.get_stats()

    if reference_text:
        wer = get_wer(reference_text, transcript, normalized=True)
        print(f"Word Error Rate (WER): {wer:.4f}")

if __name__ == "__main__":
    exit(main())