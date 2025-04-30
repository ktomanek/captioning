# Deploy a Model as service on Modal
# First, modal needs to be installed locally. Follow the instructions for how to set up modal:
#  * follow https://modal.com/docs/reference/cli/setup
#  * https://modal.com/docs/guide
# (Modal provides freebie credits)

# The launch, run:
# modal deploy deploy_modal_transcriber.py

import modal

MODAL_APP_NAME = "asr-service"

image = modal.Image.debian_slim().apt_install("ffmpeg").pip_install(
    "numpy",
    "rich",
    "torch",
    "huggingface_hub[hf_transfer]==0.26.2",
    "transformers",
)
app = modal.App(MODAL_APP_NAME)


# TODO set scaledown_window and other handling
# see https://modal.com/docs/reference/modal.App#cls
@app.cls(
    image=image,
    gpu="L4", 
    # timeout=5, # seconds,
    # scaledown_window=20,
)
class WhisperLarge:
    @modal.enter()
    def enter(self):
        """This runs once when the container starts"""
        import torch
        from transformers import pipeline
        
        # TODO use fasterwhisper

        # model_name = "openai/whisper-small"
        model_name = "openai/whisper-large-v3"
        #model_name = "openai/whisper-large-v3-turbo"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.whisper_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=device)
        print("Model loaded and ready")

    @modal.method()
    def transcribe(self, audio_chunk):
        """Process audio chunks for transcription"""
        import numpy as np
        
        # Process audio
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
        pred = self.whisper_pipeline(audio_array)
        print(f"Transcription: {pred}")
        transcription = pred.get("text", "")        
        return transcription
