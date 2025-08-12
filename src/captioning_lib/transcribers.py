# different ASR models as transcribers with settings for transcribing partial and full segments

import logging
import numpy as np
import os
import psutil
import time
import json

DEFAULT_LANGUAGE = 'en'

class Transcriber():

    def __init__(self, model_name_or_path, sampling_rate, show_word_confidence_scores=False, language=DEFAULT_LANGUAGE, output_streaming=True):
        self.number_of_partials_transcribed = 0
        self.speech_segments_transcribed = 0
        self.speech_frames_transcribed = 0
        self.compute_time = 0.0
        self.memory_used = 0

        self.sampling_rate = sampling_rate
        self.model_name = model_name_or_path
        self.show_word_confidence_scores = show_word_confidence_scores
        self.output_streaming = output_streaming

        self.language = language
        print(f"Setting model language to: {self.language}")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss  # in bytes
        self._load_model(model_name_or_path)
        mem_after = process.memory_info().rss
        self.memory_used = mem_after - mem_before
        
        self._warmup_model()
        pass

    def _load_model(self, model_name_or_path):
        raise NotImplementedError("Subclasses should implement this method to load the model.")

    def _warmup_model(self):
        """Warm up the model by running a dummy transcription."""
        data = np.zeros((self.sampling_rate,), dtype=np.float32)  # 1 second of silence
        _ = self._transcribe(data, segment_end=True)
        logging.debug('Model warmed up...')

    def _transcribe(self, audio_data, segment_end):
        raise NotImplementedError("Subclasses should implement this method to load the model.")

    def transcribe(self, audio_data, segment_end):
        """Generator that yields transcription results as they become available."""
        t1 = time.time()
        
        yielded_any = False
        for result in self._transcribe(audio_data, segment_end):
            yielded_any = True
            yield result
        
        # Only update stats after all results have been yielded
        self.compute_time += (time.time() - t1)
        self.speech_frames_transcribed += len(audio_data)
        if segment_end:
            self.speech_segments_transcribed += 1
        else:
            self.number_of_partials_transcribed += 1
        
        # Handle case where _transcribe yields nothing
        if not yielded_any:
            yield ""
    
    def get_stats(self):
        speech_time_transcribes = self.speech_frames_transcribed / self.sampling_rate
        rtfx = speech_time_transcribes / self.compute_time
        print(f"Model uses {self.memory_used / (1024 * 1024):.2f} MB of RAM")
        print(f"Number of inference calls total: {self.speech_segments_transcribed + self.number_of_partials_transcribed}")
        print(f"Number of partial segments transcribed: {self.number_of_partials_transcribed}")
        print(f"Number of full segments transcribed: {self.speech_segments_transcribed}")
        print(f"Number of frames transcribed: {self.speech_frames_transcribed}")
        print(f"Total speech time transcribed: {speech_time_transcribes:.2f} sec")
        print(f"Total inference time: {self.compute_time:.2f} sec")
        print(f"Inverse real-time factor (RTFx): {rtfx:.2f}")
        
class FasterWhisperTranscriber(Transcriber):
    AVAILABLE_MODELS = {'fasterwhisper_tiny': 'tiny',
                        'fasterwhisper_base': 'base',
                        'fasterwhisper_small': 'small'}
    
    def _load_model(self, model_name):
        if model_name not in self.AVAILABLE_MODELS.keys():
            raise ValueError(f"Model {model_name} is not supported by WhisperTranscriber.")
        
        from faster_whisper import WhisperModel
        
        full_model_name = self.AVAILABLE_MODELS[model_name]
        self.model = WhisperModel(full_model_name, device="cpu", compute_type="int8")

        logging.info(f"Loaded FasterWhisper model: {model_name} --> {full_model_name}")

    def _transcribe(self, audio_data, segment_end):
        # for partial transcriptions, we are using smaller beam size
        beam_size = 5 if segment_end else 1

        # use VAD filter only for full transcriptions
        use_vad_filter = segment_end

        # use word probabilities only for full transcriptions
        use_word_probabilities = self.show_word_confidence_scores and segment_end
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=beam_size,
            language=self.language,
            task='transcribe',
            condition_on_previous_text=False,
            vad_filter=use_vad_filter,
            word_timestamps=use_word_probabilities,
        )
        
        if self.output_streaming:
            # Stream segments one by one
            for segment in segments:
                if use_word_probabilities:
                    segment_text = ''
                    for word in segment.words:
                        segment_text += word.word + '/' + str(word.probability) + ' '
                    segment_text = segment_text.strip()
                    if segment_text:
                        yield segment_text
                else:
                    if segment.text.strip():
                        yield segment.text.strip()
        else:
            # Accumulate all segments and yield as one result
            complete_text = ""
            for segment in segments:
                if use_word_probabilities:
                    for word in segment.words:
                        complete_text += word.word + '/' + str(word.probability) + ' '
                else:
                    complete_text += segment.text + ' '
            if complete_text.strip():
                yield complete_text.strip()

class VoskTranscriber(Transcriber):
    AVAILABLE_MODELS = {'vosk_tiny': 'tiny'}

    def _load_model(self, model_name):
        from vosk import KaldiRecognizer, Model
        lang_str = None
        if self.language == 'en':
            lang_str = 'en-us'
        else:
            raise ValueError(f"Language {self.language} is not supported by VoskTranscriber.")
        self.model = Model(lang=lang_str)
        self.rec = KaldiRecognizer(self.model, self.sampling_rate)

    def _transcribe(self, audio_data, segment_end):

        # Vosk expects audio data as int16
        audio_data = (audio_data * 32767).astype(np.int16)

        # we're sort of not using vosk the way it is intended here (ie, having it do
        # the streaming and VAD steps, hence not handling PartialResults here and FinalResult
        # will likely not have anything either)
        rec_end = self.rec.AcceptWaveform(audio_data.tobytes())
        transcript = json.loads(self.rec.Result())["text"]

        if segment_end:
            final_result = json.loads(self.rec.FinalResult())
            if final_result['text']:
                t = final_result['text']
                transcript += ' ' + t
            self.rec.Reset()
        
        if transcript.strip():
            yield transcript.strip()

class TranslationTranscriber(Transcriber):
    """Model for partial transcripts show the source language, for final segments
    the transcript is translated to English using a larger model running on Modal remotely.
    """

    AVAILABLE_MODELS = {'translation_from_es': 'es',
                        'translation_from_de': 'de',
                        'translation_from_fr': 'fr'
                        }

    def _load_model(self, model_name):
        from faster_whisper import WhisperModel
        import modal
        import deploy_modal_transcriber

        # set language
        self.source_language = self.AVAILABLE_MODELS[model_name]
        print(f"Set source language to: {self.source_language}")
        
        # load models
        self.model_for_partials = WhisperModel('tiny', device="cpu", compute_type="int8")
        logging.info(f"Loaded Whisper model: tiny")
        self.model_for_segments = modal.Cls.from_name(deploy_modal_transcriber.MODAL_APP_NAME, "FasterWhisper")
        print(f"Connecting to remote model on Modal: {self.model_for_segments} -- this may take up to 2 minutes if service needs to be started.")
        # send random audio to trigger model loading
        random_audio_np = np.random.rand(16000).astype(np.float32)
        _ = self.model_for_segments().transcribe.remote(random_audio_np)
        print(f"Remote model loaded!")

        
    def _transcribe(self, audio_data, segment_end):

        if segment_end:
            text = self.model_for_segments().transcribe.remote(
                audio_data, translate_from_source_language=self.source_language)
            if text and text.strip():
                yield text.strip()
        else:
            segments, _ = self.model_for_partials.transcribe(
                audio_data,
                beam_size=1,
                language=self.source_language,
                task='transcribe',
                condition_on_previous_text=False,
                vad_filter=True,
                word_timestamps=False,
            )
            for segment in segments:
                if segment.text.strip():
                    yield segment.text.strip()

class NemoTranscriber(Transcriber):

    # TODO use other nemo models
    # stt_en_fastconformer_hybrid_large_pc
    # stt_de_fastconformer_hybrid_large_pc, stt_de_conformer_ctc_large, stt_de_conformer_transducer_large
    # stt_es_fastconformer_hybrid_large_pc, stt_enes_conformer_ctc_large, stt_enes_conformer_transducer_large

    CTC_MODEL = 'nvidia/stt_en_fastconformer_ctc_large'
    RNNT_MODEL = 'nvidia/stt_en_fastconformer_transducer_large'
    E2E_MODEL = 'nvidia/canary-180m-flash'


    AVAILABLE_MODELS = {'nemo_ctc': CTC_MODEL,
                        'nemo_rnnt': RNNT_MODEL,
                        'nemo_canary': E2E_MODEL}


    # TODO grab word probabilities from the model to colorize outputs 
    # as done in WhisperTranscriber

    def _load_model(self, model_name):

        if self.language != DEFAULT_LANGUAGE:
            raise ValueError(f"Language {self.language} is not supported by NemoTranscriber.")

        if model_name not in self.AVAILABLE_MODELS.keys():
            raise ValueError(f"Model {model_name} is not supported by NemoTranscriber.")
        full_model_name = self.AVAILABLE_MODELS[model_name]

        if full_model_name == self.E2E_MODEL:
            from nemo.collections.asr.models import EncDecMultiTaskModel
            self.model = EncDecMultiTaskModel.from_pretrained(self.E2E_MODEL)
            # update decode params
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self.model.change_decoding_strategy(decode_cfg)   
        elif full_model_name == self.CTC_MODEL:
            from nemo.collections.asr.models import EncDecCTCModelBPE
            self.model = EncDecCTCModelBPE.from_pretrained(self.CTC_MODEL)
        elif full_model_name == self.RNNT_MODEL:
            from nemo.collections.asr.models import EncDecRNNTBPEModel
            self.model = EncDecRNNTBPEModel.from_pretrained(self.RNNT_MODEL)

        # make nemo models less verbose
        nemo_logger = logging.getLogger("nemo_logger")
        nemo_logger.setLevel(logging.ERROR)  # Only show errors and critical issues

        logging.info(f"Loaded Nemo model: {model_name} --> {full_model_name}")

    def _transcribe(self, audio_data, segment_end):
        if self.model_name == self.E2E_MODEL:
            output = self.model.transcribe(
                audio_data, batch_size=1,
                return_hypotheses=False,
                source_lang="en",
                target_lang="en",
                task="asr",
                pnc="yes", # "yes" for punctuation and capitalization 
                verbose=False # don't show progress bar
                )
        else:
            output = self.model.transcribe(
                audio_data, batch_size=1,
                return_hypotheses=False,
                verbose=False # don't show progress bar
                )

        result = output[0].text if output else ""
        if result.strip():
            yield result.strip()

class MoonshineTranscriber(Transcriber):
    AVAILABLE_MODELS = {'moonshine_onnx_tiny': 'tiny',
                        'moonshine_onnx_base': 'base'}

    def _load_model(self, model_name):

        if self.language != DEFAULT_LANGUAGE:
            raise ValueError(f"Language {self.language} is not supported by MoonshineTranscriber.")

        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        if model_name not in self.AVAILABLE_MODELS.keys():
            raise ValueError(f"Model {model_name} is not supported by MoonshineTranscriber.")
        
        full_model_name = self.AVAILABLE_MODELS[model_name]
        self.tokenizer = load_tokenizer()
        self.model = MoonshineOnnxModel(model_name=full_model_name)

        logging.info(f"Loaded Moonshine ONNX model: {full_model_name}")

    def _transcribe(self, audio_data, segment_end):
        tokens = self.model.generate(audio_data[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]
        if text and text.strip():
            yield text.strip()

class CustomWhisperONNXTranscriber(Transcriber):
    """Custom Whisper ONNX transcriber that uses user-provided ONNX model files."""
    AVAILABLE_MODELS = {'whisperonnx': 'custom'}

    # TODO
    BASE_MODEL_NAME = "openai/whisper-tiny"  # Constant for tokenizer

    def __init__(self, model_name_or_path, sampling_rate, show_word_confidence_scores=False, language=DEFAULT_LANGUAGE, model_path=None, output_streaming=True):
        self.model_path = model_path
        if not model_path:
            raise ValueError("model_path is required for CustomWhisperONNXTranscriber")
        super().__init__(model_name_or_path, sampling_rate, show_word_confidence_scores, language, output_streaming)

    def _load_model(self, model_name_or_path):
        if self.language != DEFAULT_LANGUAGE:
            raise ValueError(f"Language {self.language} is not supported by CustomWhisperONNXTranscriber yet.")

        try:
            import onnxruntime as ort
            from transformers import WhisperProcessor
        except ImportError as e:
            raise ImportError(f"Required libraries not installed: {e}. Please install: pip install onnxruntime transformers")

        # Construct model file paths
        encoder_path = os.path.join(self.model_path, "encoder_model.onnx")
        decoder_path = os.path.join(self.model_path, "decoder_model.onnx")
        decoder_with_past_path = os.path.join(self.model_path, "decoder_with_past_model.onnx")

        # Verify files exist
        for path in [encoder_path, decoder_path, decoder_with_past_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required ONNX model file not found: {path}")

        # Optimize for CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 0  # Use all available cores
        sess_options.intra_op_num_threads = 0

        # Load ONNX sessions
        print(f"Loading ONNX models from: {self.model_path}")
        print(f"  - Encoder: {os.path.basename(encoder_path)}")
        print(f"  - Decoder: {os.path.basename(decoder_path)}")  
        print(f"  - Decoder with past: {os.path.basename(decoder_with_past_path)}")
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options=sess_options)
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options=sess_options)
        self.decoder_with_past_session = ort.InferenceSession(decoder_with_past_path, sess_options=sess_options)

        # Detect model size from encoder dimensions
        self.detected_model_size = self._detect_model_size()
        
        # Load processor for audio preprocessing and tokenizer
        # Use detected model size for appropriate tokenizer
        base_model_name = f"openai/whisper-{self.detected_model_size}"
        self.processor = WhisperProcessor.from_pretrained(base_model_name)
        self.tokenizer = self.processor.tokenizer
        print(f"Using processor: {base_model_name}")

        # Special tokens
        self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.no_timestamps_token = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.transcribe_token = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.english_token = self.tokenizer.convert_tokens_to_ids("<|en|>")

        # Get input/output names
        self.decoder_outputs = [out.name for out in self.decoder_session.get_outputs()]
        self.decoder_with_past_outputs = [out.name for out in self.decoder_with_past_session.get_outputs()]

        logging.info(f"Loaded CustomWhisperONNX model from: {self.model_path}")
        logging.info(f"Detected model size: {self.detected_model_size}")

    def _detect_model_size(self):
        """Detect Whisper model size from encoder output dimensions"""
        # Get encoder output shape - typically [batch_size, seq_len, hidden_dim]
        encoder_outputs = self.encoder_session.get_outputs()
        
        # Get hidden dimension from the first output
        output_shape = encoder_outputs[0].shape
        hidden_dim = output_shape[-1]  # Last dimension is hidden size
        
        # Map hidden dimensions to model sizes
        size_map = {
            384: "tiny",
            512: "base", 
            768: "small",
            1024: "medium",
            1280: "large-v3"  # Also covers large-v2, large, and large-v3-turbo
        }
        
        detected_size = size_map.get(hidden_dim)
        if detected_size:
            logging.info(f"Detected model size from hidden dimension {hidden_dim}: {detected_size}")
            return detected_size
        else:
            raise ValueError(f"Unknown hidden dimension {hidden_dim}. Supported dimensions: {list(size_map.keys())}")

    def _preprocess_audio(self, audio_data):
        """Preprocess audio data for ONNX model"""
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Whisper expects 30-second chunks, pad if necessary
        expected_length = 30 * self.sampling_rate  # 30 seconds * 16000 = 480000 samples
        if len(audio_data) < expected_length:
            # Pad with zeros
            padding = expected_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        elif len(audio_data) > expected_length:
            # Truncate to 30 seconds
            audio_data = audio_data[:expected_length]

        # Use processor to create input features
        inputs = self.processor(
            audio_data,
            sampling_rate=self.sampling_rate,
            return_tensors="np"
        )
        return inputs.input_features

    def _encode_audio(self, input_features):
        """Encode audio features using ONNX encoder"""
        encoder_outputs = self.encoder_session.run(
            None,
            {"input_features": input_features}
        )
        return encoder_outputs[0]  # encoder hidden states

    def _transcribe(self, audio_data, segment_end):
        """Perform transcription using ONNX models"""
        try:
            # Preprocess audio
            input_features = self._preprocess_audio(audio_data)
            
            # Encode audio
            encoder_hidden_states = self._encode_audio(input_features)
            
            if self.output_streaming:
                # Stream decode tokens one by one
                yield from self._decode_streaming(encoder_hidden_states, max_length=448)
            else:
                # Accumulate all tokens and yield complete result
                complete_text = ""
                for token in self._decode_streaming(encoder_hidden_states, max_length=448):
                    complete_text += token
                if complete_text.strip():
                    yield complete_text.strip()
            
        except Exception as e:
            logging.error(f"ONNX transcription error: {e}")
            yield f"[Error: {str(e)}]"

    def _decode_streaming(self, encoder_hidden_states, max_length=448):
        """Streaming decoding with ONNX models"""
        # Initialize decoder input with start tokens
        decoder_input_ids = np.array([
            [self.sot_token, self.english_token, self.transcribe_token, self.no_timestamps_token]
        ], dtype=np.int64)
        
        past_key_values_dict = {}
        
        for i in range(max_length):
            if not past_key_values_dict:
                # First iteration - use full decoder
                inputs = {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_hidden_states
                }
                
                outputs = self.decoder_session.run(None, inputs)
                logits = outputs[0]
                
                # Store past key values
                for idx, output_name in enumerate(self.decoder_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]
            else:
                # Subsequent iterations - use decoder with past
                current_input_ids = decoder_input_ids[:, -1:].astype(np.int64)
                
                inputs = {"input_ids": current_input_ids}
                inputs.update(past_key_values_dict)
                
                outputs = self.decoder_with_past_session.run(None, inputs)
                logits = outputs[0]
                
                # Update past key values
                for idx, output_name in enumerate(self.decoder_with_past_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]
            
            # Get next token (greedy decoding)
            next_token_logits = logits[0, -1, :]
            next_token_id = np.argmax(next_token_logits)
            
            # Check for end of transcript
            if next_token_id == self.eot_token:
                break
            
            # Add to sequence
            decoder_input_ids = np.concatenate([
                decoder_input_ids, 
                np.array([[next_token_id]], dtype=np.int64)
            ], axis=1)
            
            # Decode token to text for streaming output
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            if token_text.strip():  # Only yield non-empty tokens
                if self.show_word_confidence_scores:
                    # Calculate confidence if requested
                    exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
                    token_probs = exp_logits / np.sum(exp_logits)
                    confidence = token_probs[next_token_id]
                    yield f"{token_text}/{confidence:.2f}"
                else:
                    yield token_text


class RemoteGPUTranscriber(Transcriber):
    """Runs a model on GPU via Modal functions.
    
    Finished segments are always processed remotely, partials can optionally
    be processed locally using the tiny Moonshine ONNX model for lower latency.
    """
    AVAILABLE_MODELS = {'remote_and_local': 'remote_and_local',
                        'remote_only': 'remote_only'}

    def _load_model(self, model_name_or_path):

        if self.language != DEFAULT_LANGUAGE:
            raise ValueError(f"Language {self.language} is not supported by RemoteGPUTranscriber.")


        # load local model: Moonshine ONNX
        logging.info("Loading local Moonshine ONNX model...")
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        self.local_tokenizer = load_tokenizer()
        self.local_model = MoonshineOnnxModel(model_name='tiny')
        logging.debug(f"Loaded local model: Moonshine ONNX tiny")

        # load remote model
        import modal
        import deploy_modal_transcriber

        logging.info("Loading remote ASR model from Modal...")

        self.remote_asr_cls = modal.Cls.from_name(deploy_modal_transcriber.MODAL_APP_NAME, "FasterWhisper")
        
        # startup time for the required image is surprisingly slow on Modal
        # self.remote_asr_cls = modal.Cls.from_name(deploy_modal_transcriber.MODAL_APP_NAME, "NemoASR")
        print(f"Connecting to remote model on Modal: {self.remote_asr_cls} -- this may take up to 2 minutes if service needs to be started.")
        # send random audio to trigger model loading
        random_audio_np = np.random.rand(16000).astype(np.float32)
        _ = self.remote_asr_cls().transcribe.remote(random_audio_np)
        print(f"Remote model loaded!")


    def _transcribe(self, audio_data, segment_end):
        if self.model_name == 'remote_only':
            use_remote_model = True
        else:
            use_remote_model = segment_end

        if use_remote_model:
            text = self.remote_asr_cls().transcribe.remote(audio_data)
        else:
            tokens = self.local_model.generate(audio_data[np.newaxis, :].astype(np.float32))
            text = self.local_tokenizer.decode_batch(tokens)[0]
        
        if text and text.strip():
            yield text.strip()


