# Deploy a Model as service on Modal
# First, modal needs to be installed locally. Follow the instructions for how to set up modal:
#  * follow https://modal.com/docs/reference/cli/setup
#  * https://modal.com/docs/guide
# (Modal provides freebie credits)

# The launch, run:
# modal deploy deploy_modal_transcriber.py

# TODO use scaledown_window instead of container_idle_timeout for later versions of modal
# see https://modal.com/docs/reference/modal.App#cls


import modal

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

@app.cls(image=cuda_image, gpu="L4", container_idle_timeout=180)
class FasterWhisper:
    @modal.enter()
    def enter(self):
        import torch
        from faster_whisper import WhisperModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model = WhisperModel('large', device="cuda", compute_type="float16")
        print("FasterWhisper model loaded.")

    @modal.method()
    def transcribe(self, audio_chunk):
        """Process audio chunks for transcription"""
        import numpy as np
        
        # Process audio
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)

        segments, _ = self.model.transcribe(
            audio_array,
            beam_size=5,
            language='en',
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        transcription = ''
        for segment in segments:
                for word in segment.words:
                    transcription += word.word + '/' + str(word.probability) + ' '
        print(f"Transcription: {transcription}")
        return transcription.strip()


@app.cls(image=cuda_image, gpu="L4", container_idle_timeout=180)
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
