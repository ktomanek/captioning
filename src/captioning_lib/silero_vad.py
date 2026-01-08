"""
Torch-free ONNX loader for Silero VAD. 
Can be used to replace silero_vad package to avoid PyTorch dependency.

Fully offline, doesn't try to download model.
Hence, you need to download the silero VAD ONNX model separately.
(see helpers/download_silero_vad_model.py)


This is based on https://github.com/snakers4/silero-vad/tree/master/src/silero_vad
but replaces torch with numpy.
"""
import onnxruntime
import numpy as np

class SileroVADOnnx:
    """ONNX-only wrapper for Silero VAD model."""

    def __init__(self, model_path, force_cpu=True):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        providers = ['CPUExecutionProvider'] if force_cpu else onnxruntime.get_available_providers()

        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers
        )

        # Check model input/output names to handle different ONNX model versions
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.reset_states()
        self._sample_rate = 16000

    def reset_states(self, batch_size=1):
        """Reset internal model states."""
        # Support both state formats (separate h/c or combined state)
        self._h = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._c = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        # Context buffer for maintaining continuity between chunks
        self._context = np.zeros((0,), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        """
        Run VAD inference.

        Args:
            x: Audio chunk as numpy array (int16 or float32)
            sr: Sample rate

        Returns:
            float: Speech probability [0.0, 1.0]
        """
        # Normalize int16 to float32
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0

        # Ensure 2D shape (batch, samples)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if len(x.shape) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk: {len(x.shape)}")

        # Validate sample rate
        if sr != 16000 and sr != 8000:
            raise ValueError(f"Unsupported sampling rate: {sr}. Supported: 8000, 16000")

        # Expected chunk sizes
        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(f"Expected {num_samples} samples for {sr}Hz, got {x.shape[-1]}")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        # Reset states if batch size or sample rate changed
        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        # Initialize context buffer if needed
        if len(self._context) == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        # Concatenate context with input
        x = np.concatenate([self._context, x], axis=1)

        # Build inputs based on model's expected input names
        if 'state' in self.input_names:
            # Combined state version
            ort_inputs = {
                'input': x,
                'state': self._state,
                'sr': np.array(sr, dtype=np.int64)
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, self._state = ort_outs
        else:
            # Separate h/c version
            ort_inputs = {
                'input': x,
                'h': self._h,
                'c': self._c,
                'sr': np.array(sr, dtype=np.int64)
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs

        # Update context with last context_size samples
        self._context = x[:, -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out.item()


class VADIterator:
    """
    Numpy-only VAD iterator for streaming speech detection.
    Compatible replacement for silero_vad.VADIterator.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Initialize VAD iterator.

        Args:
            model: SileroVADOnnx model instance
            threshold: Speech probability threshold (0.0-1.0)
            sampling_rate: Audio sample rate (8000 or 16000)
            min_silence_duration_ms: Minimum silence to end speech segment
            speech_pad_ms: Padding added to speech boundaries
        """
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        # Convert durations to samples
        self.min_silence_samples = sampling_rate * min_silence_duration_ms // 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms // 1000

        self.reset_states()

    def reset_states(self, full_reset=True):
        """Reset all iterator states."""
        if full_reset:
            self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        Process audio chunk and detect speech boundaries.

        Args:
            x: Audio chunk as numpy array (float32, normalized to [-1, 1])
            return_seconds: If True, return timestamps in seconds instead of samples

        Returns:
            dict or None: {'start': timestamp} when speech begins,
                         {'end': timestamp} when speech ends,
                         None otherwise
        """
        if isinstance(x, np.ndarray):
            if x.dtype == np.int16:
                x = x.astype(np.float32) / 32768.0

        window_size_samples = len(x)

        # Get speech probability from model
        speech_prob = self.model(x, self.sampling_rate)

        if speech_prob >= self.threshold and self.temp_end != 0:
            # Reset temp_end if speech resumes during silence
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            # Speech start detected
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            speech_start = max(0, speech_start)

            self.current_sample += window_size_samples

            if return_seconds:
                return {'start': speech_start / self.sampling_rate}
            else:
                return {'start': speech_start}

        if speech_prob < self.threshold - 0.15 and self.triggered:
            # Potential silence detected
            if self.temp_end == 0:
                # Mark the start of silence
                self.temp_end = self.current_sample

            # Check if silence has lasted long enough
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                # Speech end confirmed
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False

                self.current_sample += window_size_samples

                if return_seconds:
                    return {'end': speech_end / self.sampling_rate}
                else:
                    return {'end': speech_end}

        self.current_sample += window_size_samples
        return None


def load_silero_vad(onnx_model_path):
    return SileroVADOnnx(onnx_model_path, force_cpu=True)

# # Old version allowing to also download the model if not found locally.
# from pathlib import Path
# def load_silero_vad_onnx(model_path, model_name='silero_vad.onnx', force_cpu=True):
#     """
#     Load Silero VAD using ONNX runtime only (no torch dependency).

#     Args:
#         model_path: Path to ONNX model file. If provided, this takes precedence.
#                    Can be absolute or relative path. If None, will search for model.
#         model_name: Name of ONNX model file to search for if model_path not provided.
#                    Options: 'silero_vad.onnx' (default), 'silero_vad_half.onnx'
#         force_cpu: If True, force CPU execution (default: True); otherwise might use GPU or other providers like 
#                     - TensorrtExecutionProvider (NVIDIA TensorRT)
#                     - CoreMLExecutionProvider (Apple Silicon GPU)
#                     - OpenVINOExecutionProvider (Intel optimizations)
#                    But not worth it since model is small

#     Returns:
#         SileroVADOnnx: Model wrapper compatible with VADIteratorOnnx

#     Examples:
#         # Load from custom path
#         model = load_silero_vad_onnx(model_path='./models/silero_vad/silero_vad.onnx')

#         # Search for model in default locations
#         model = load_silero_vad_onnx()
#     """
#     # If explicit path provided, use it directly
#     if model_path is not None:
#         model_path = Path(model_path)
#         if not model_path.exists():
#             raise FileNotFoundError(f"Model file not found: {model_path}")
#         print(f"Loading Silero VAD ONNX model from: {model_path}")
#         return SileroVADOnnx(str(model_path), force_cpu=force_cpu)

#     # Otherwise, search for model in multiple locations
#     search_paths = []

#     # 1. Local models directory (for downloaded models)
#     local_model_dir = Path(__file__).parent.parent.parent / 'models' / 'silero_vad'
#     search_paths.append(local_model_dir / model_name)

#     # 2. Relative to this file
#     search_paths.append(Path(__file__).parent / 'models' / model_name)

#     # 3. Try silero_vad package data (if installed)
#     try:
#         import importlib.resources as pkg_resources
#         try:
#             # Python 3.9+
#             files = pkg_resources.files('silero_vad.data')
#             search_paths.append(files.joinpath(model_name))
#         except AttributeError:
#             # Python 3.8
#             with pkg_resources.path('silero_vad.data', model_name) as p:
#                 search_paths.append(Path(p))
#     except (ImportError, ModuleNotFoundError):
#         pass

#     # Try each path
#     for path in search_paths:
#         if Path(path).exists():
#             print(f"Loading Silero VAD ONNX model from: {path}")
#             return SileroVADOnnx(str(path), force_cpu=force_cpu)

#     # If nothing found, provide helpful error
#     raise FileNotFoundError(
#         f"Could not find '{model_name}' in any of these locations:\n" +
#         "\n".join(f"  - {p}" for p in search_paths) +
#         f"\n\nTo download the model, run:\n  python download_silero_vad_model.py"
#     )
