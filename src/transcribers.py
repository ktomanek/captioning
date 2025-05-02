# different ASR models as transcribers with settings for transcribing partial and full segments

import logging
import numpy as np
import os
import psutil
import time

class Transcriber():

    def __init__(self, model_name_or_path, sampling_rate):
        self.number_of_partials_transcribed = 0
        self.speech_segments_transcribed = 0
        self.speech_frames_transcribed = 0
        self.compute_time = 0.0
        self.memory_used = 0

        self.sampling_rate = sampling_rate
        self.model_name = model_name_or_path

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
        t1 = time.time()
        t = self._transcribe(audio_data, segment_end)
        self.compute_time += (time.time() - t1)
        self.speech_frames_transcribed += len(audio_data)
        if segment_end:
            self.speech_segments_transcribed += 1
        else:
            self.number_of_partials_transcribed += 1
        return t
    
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
        
class WhisperTranscriber(Transcriber):
    AVAILABLE_MODELS = {'whisper_tiny': 'tiny',
                        'whisper_base': 'base',
                        'whisper_small': 'small'}
    
    USE_WORD_PROBABILITIES = True  

    def _load_model(self, model_name):
        if model_name not in self.AVAILABLE_MODELS.keys():
            raise ValueError(f"Model {model_name} is not supported by WhisperTranscriber.")
        
        from faster_whisper import WhisperModel
        
        full_model_name = self.AVAILABLE_MODELS[model_name]
        self.model = WhisperModel(full_model_name, device="cpu", compute_type="int8")

        logging.info(f"Loaded Whisper model: {model_name} --> {full_model_name}")

    def _transcribe(self, audio_data, segment_end):
        # for partial transcriptions, we are using smaller beam size
        beam_size = 5 if segment_end else 1

        use_word_probabilities = self.USE_WORD_PROBABILITIES and segment_end
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=beam_size,
            language='en',
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=use_word_probabilities,
        )
        pred = ''
        for segment in segments:
            if use_word_probabilities:
                for word in segment.words:
                    pred += word.word + '/' + str(word.probability) + ' '
            else:
                pred += segment.text + ' '
        return pred.strip()


class NemoTranscriber(Transcriber):

    CTC_MODEL = 'nvidia/stt_en_fastconformer_ctc_large'
    RNNT_MODEL = 'nvidia/stt_en_fastconformer_transducer_large'
    E2E_MODEL = 'nvidia/canary-180m-flash'


    AVAILABLE_MODELS = {'nemo_ctc': CTC_MODEL,
                        'nemo_rnnt': RNNT_MODEL,
                        'nemo_canary': E2E_MODEL}


    # TODO grab word probabilities from the model to colorize outputs 
    # as done in WhisperTranscriber

    def _load_model(self, model_name):
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

        return output[0].text if output else ""


class MoonshineTranscriber(Transcriber):
    AVAILABLE_MODELS = {'moonshine_onnx_tiny': 'tiny',
                        'moonshine_onnx_base': 'base'}

    def _load_model(self, model_name):
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
        return text.strip() if text else ""


class RemoteGPUTranscriber(Transcriber):
    """Runs a model on GPU via Modal functions.
    
    Finished segments are always processed remotely, partials can optionally
    be processed locally using the tiny Moonshine ONNX model for lower latency.
    """
    AVAILABLE_MODELS = {'remote_and_local': 'remote_and_local',
                        'remote_only': 'remote_only'}

    def _load_model(self, model_name_or_path):

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
        return text.strip() if text else ""


