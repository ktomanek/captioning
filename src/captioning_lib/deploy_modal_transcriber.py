# Deploy a Model as service on Modal
# First, modal needs to be installed locally. Follow the instructions for how to set up modal:
#  * follow https://modal.com/docs/reference/cli/setup
#  * https://modal.com/docs/guide
# (Modal provides freebie credits)

# The launch, run:
# modal deploy deploy_modal_transcriber.py

import modal
from pathlib import Path
import numpy as np

MODAL_APP_NAME = "asr-service"

nemo_image = modal.Image.debian_slim().apt_install("ffmpeg").pip_install(
    "torch",
    "nemo_toolkit[asr]",
)

# latest version of ctranslate2 (4.5.0) requires cuda 12 and cudnn 9
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "numpy",
        "huggingface_hub[hf_transfer]==0.26.2",        
        "torch",
        "ctranslate2",
        "faster_whisper",
        "transformers",
    )
)

app = modal.App(MODAL_APP_NAME)

volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)
MODEL_DIR = Path("/models")

@app.cls(
    image=cuda_image, 
    gpu="L4", 
    scaledown_window=60 * 2, 
    enable_memory_snapshot=True,
    volumes={MODEL_DIR: volume})
@modal.concurrent(max_inputs=2)
class FasterWhisper:

    sample_rate = 16000
    model_id = 'large-v3-turbo'
    model_path = None

    def download_model(self):
        from faster_whisper.utils import download_model
        from pathlib import Path

        self.model_path = MODEL_DIR / self.model_id

        if not self.model_path.exists():
            print(f"Downloading model to {self.model_path} ...")            
            self.model_path.mkdir(parents=True)
            download_model(self.model_id, output_dir=str(self.model_path))
            print(f"Model downloaded successfully.")
        else:
            print(f"Model already available on {self.model_path}.")

    @modal.enter()
    def enter(self):
        from faster_whisper import WhisperModel
        self.download_model()

        self.model = WhisperModel(str(self.model_path), device="cuda", compute_type="float16")
        print(f"FasterWhisper model loaded.")

    @modal.method()
    def transcribe(self, audio_chunk, translate_from_source_language=None):
        """Process audio chunks for transcription"""
        import time
        import numpy as np

        t1 = time.time()
        print(f"Transcription starting...")
        
        # Process audio
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)

        task = 'transcribe'
        language = 'en'
        if translate_from_source_language:
            task = 'translate'
            language = translate_from_source_language

        print(f"running {task} on {language}...")

        segments, _ = self.model.transcribe(
            audio_array,
            beam_size=5,
            language=language,
            task=task,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        transcription = ''
        for segment in segments:
                for word in segment.words:
                    transcription += word.word + '/' + str(word.probability) + ' '
        print(f"{task.upper()}: {transcription}")
        return transcription.strip()


@app.cls(image=cuda_image, gpu="L4", scaledown_window=60 * 2)
class NemoASR:
    @modal.enter()
    def enter(self):
        import torch
        from nemo.collections.asr.models import EncDecMultiTaskModel
        if torch.cuda.is_available():
            device = torch.device(f'cuda:0')
        else:
            device = torch.device(f'cpu')
        print(f"Using device: {device}")

        model_name = 'nvidia/canary-1b'
        # model_name = 'nvidia/canary-180m-flash'

        self.model = EncDecMultiTaskModel.from_pretrained(model_name, map_location=device)
        print("Nemo model loaded.")

    @modal.method()
    def transcribe(self, audio_chunk):
        """Process audio chunks for transcription"""
        import numpy as np
        
        # Process audio
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
        pred = self.model.transcribe(
            audio_array, 
            batch_size=1,
            source_lang="en",
            target_lang="en",
            task="asr",
            pnc="yes" # "yes" for punctuation and capitalization 
        )
        print('PRED:', pred)
        transcription = pred[0].text.strip() if pred else ""
        print(f"Transcription: {transcription}")
        return transcription
